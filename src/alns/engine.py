"""ALNS engine: runs the optimization loop and tracks the best solution."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.alns.acceptance import GreedyAcceptance, SimulatedAnnealing
from src.alns.weights import AdaptiveWeights
from src.config import ALNSConfig
from src.operators.destroy import DestroyOperator
from src.operators.repair import RepairOperator
from src.problem.instance import Instance
from src.problem.solution import Solution

logger = logging.getLogger(__name__)


@dataclass
class ALNSResult:
    best_solution: Solution
    best_cost: float
    best_distance: float
    best_n_vehicles: int
    iterations_run: int
    elapsed_seconds: float
    history: list[float] = field(default_factory=list)
    destroy_weights: list[float] = field(default_factory=list)
    repair_weights: list[float] = field(default_factory=list)


def initial_solution(instance: Instance, rng: np.random.Generator) -> Solution:
    """Construct a feasible initial solution via greedy insertion.

    Ensures feasibility by opening new routes when capacity/time prevents insertion.
    """
    sol = Solution(instance=instance)
    customers = list(range(1, instance.n + 1))
    rng.shuffle(customers)
    for customer in customers:
        best = sol.best_insertion(customer, noise=0.0, rng=rng)
        if best is None:
            # force open new route
            sol.routes.append(
                type(sol.routes[0])(nodes=[0, customer, 0]) if sol.routes else None
            )
            if sol.routes[-1] is None:
                from src.problem.solution import Route

                sol.routes[-1] = Route(nodes=[0, customer, 0])
        else:
            delta, r_idx, pos = best
            sol.insert_customer(customer, delta, r_idx, pos)
    return sol


class ALNSEngine:
    def __init__(
        self,
        config: ALNSConfig,
        destroy_ops: list[DestroyOperator],
        repair_ops: list[RepairOperator],
        destroy_weights: list[float],
        repair_weights: list[float],
    ):
        self.config = config
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.destroy_weights = AdaptiveWeights(
            n_operators=len(destroy_ops),
            initial_weights=destroy_weights,
            reaction_factor=config.reaction_factor,
            segment_length=config.segment_length,
            score_new_best=config.score_new_best,
            score_better=config.score_better,
            score_accepted=config.score_accepted,
            score_rejected=config.score_rejected,
        )
        self.repair_weights = AdaptiveWeights(
            n_operators=len(repair_ops),
            initial_weights=repair_weights,
            reaction_factor=config.reaction_factor,
            segment_length=config.segment_length,
            score_new_best=config.score_new_best,
            score_better=config.score_better,
            score_accepted=config.score_accepted,
            score_rejected=config.score_rejected,
        )
        self.acceptance = SimulatedAnnealing(
            initial_temp=config.sa_initial_temp, cooling=config.sa_cooling
        )

    def solve(self, instance: Instance) -> ALNSResult:
        rng = np.random.default_rng(self.config.seed)
        start_time = time.time()

        current = initial_solution(instance, rng)
        current_cost = current.cost()
        best = current.copy()
        best_cost = current_cost
        history: list[float] = [best_cost]

        logger.info(
            "ALNS start: initial vehicles=%s, distance=%.2f",
            current.n_vehicles(),
            current.total_distance(),
        )

        for it in range(self.config.iterations):
            # selection
            d_idx = self.destroy_weights.select(rng)
            r_idx = self.repair_weights.select(rng)
            destroy = self.destroy_ops[d_idx]
            repair = self.repair_ops[r_idx]

            # destroy
            candidate = current.copy()
            removed = destroy.destroy(candidate, rng)
            if not removed:
                continue

            # repair
            repair.repair(candidate, removed, rng)
            if not candidate.is_feasible():
                # skip infeasible (rare fallback path)
                self.acceptance.cool()
                continue

            candidate_cost = candidate.cost()

            # scoring
            score_d = 0.0
            score_r = 0.0
            if candidate_cost < best_cost - 1e-9:
                score_d = score_r = self.config.score_new_best
                best = candidate.copy()
                best_cost = candidate_cost
            elif candidate_cost < current_cost - 1e-9:
                score_d = score_r = self.config.score_better
            elif self.acceptance.accept(current_cost, candidate_cost, rng):
                score_d = score_r = self.config.score_accepted

            if score_d == 0.0:
                self.destroy_weights.record(d_idx, self.config.score_rejected)
                self.repair_weights.record(r_idx, self.config.score_rejected)
            else:
                self.destroy_weights.record(d_idx, score_d)
                self.repair_weights.record(r_idx, score_r)

            if self.acceptance.accept(current_cost, candidate_cost, rng):
                current = candidate
                current_cost = candidate_cost

            self.acceptance.cool()

            # segment end
            if (it + 1) % self.config.segment_length == 0:
                self.destroy_weights.end_segment()
                self.repair_weights.end_segment()

            history.append(best_cost)

            if (it + 1) % 5000 == 0:
                logger.info(
                    "Iter %s: best=%.2f (veh=%s, dist=%.2f)",
                    it + 1,
                    best_cost,
                    best.n_vehicles(),
                    best.total_distance(),
                )

        elapsed = time.time() - start_time
        logger.info(
            "ALNS done in %.1fs: vehicles=%s, distance=%.2f",
            elapsed,
            best.n_vehicles(),
            best.total_distance(),
        )
        return ALNSResult(
            best_solution=best,
            best_cost=best_cost,
            best_distance=best.total_distance(),
            best_n_vehicles=best.n_vehicles(),
            iterations_run=self.config.iterations,
            elapsed_seconds=elapsed,
            history=history,
            destroy_weights=self.destroy_weights.weights.tolist(),
            repair_weights=self.repair_weights.weights.tolist(),
        )
