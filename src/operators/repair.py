"""Repair operators: re-insert removed customers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.problem.instance import Instance
from src.problem.solution import Solution


class RepairOperator(ABC):
    name: str = "base"

    @abstractmethod
    def repair(
        self, solution: Solution, removed: list[int], rng: np.random.Generator
    ) -> None:
        """Insert all removed customers into the solution (in place)."""

    def _force_insert(
        self, solution: Solution, customer: int, rng: np.random.Generator
    ) -> None:
        """Insert even if infeasible — fallback to avoid dead-ends.

        Tries best feasible insertion first, then opens a new route if capacity allows,
        finally appends to the least-loaded route.
        """
        inst = solution.instance
        best = solution.best_insertion(customer, noise=0.0, rng=rng)
        if best is not None:
            delta, r_idx, pos = best
            solution.insert_customer(customer, delta, r_idx, pos)
            return
        # new route as fallback (accepts capacity violation minimally)
        solution.routes.append(
            type(solution.routes[0])(nodes=[0, customer, 0]) if solution.routes else None
        )
        if solution.routes[-1] is None:
            from src.problem.solution import Route

            solution.routes[-1] = Route(nodes=[0, customer, 0])


class GreedyInsertion(RepairOperator):
    """Insert customers one at a time at their globally best position."""

    name = "greedy_insertion"

    def __init__(self, noise: float = 0.0):
        self.noise = noise

    def repair(
        self, solution: Solution, removed: list[int], rng: np.random.Generator
    ) -> None:
        # process in random order to diversify
        order = list(removed)
        rng.shuffle(order)
        for customer in order:
            best = solution.best_insertion(customer, noise=self.noise, rng=rng)
            if best is None:
                self._force_insert(solution, customer, rng)
            else:
                delta, r_idx, pos = best
                solution.insert_customer(customer, delta, r_idx, pos)


class RegretInsertion(RepairOperator):
    """Regret-k: at each step insert the customer with max regret = sum of
    (k-th best cost - best cost)."""

    name = "regret_insertion"

    def __init__(self, k: int = 2, noise: float = 0.0):
        self.k = max(2, k)
        self.noise = noise

    def _candidate_costs(
        self, solution: Solution, customer: int, rng: np.random.Generator
    ) -> list[tuple[float, int, int]]:
        """Return sorted list of feasible (delta_cost, route_idx, position) options.

        Includes the option of opening a new route (route_idx = -1).
        """
        inst = solution.instance
        cust = inst.node(customer)
        options: list[tuple[float, int, int]] = []

        for r_idx, route in enumerate(solution.routes):
            if solution.route_load(route) + cust.demand > inst.capacity + 1e-6:
                continue
            for pos in range(len(route.nodes) - 1):
                prev = route.nodes[pos]
                nxt = route.nodes[pos + 1]
                added = (
                    inst.dist(prev, customer)
                    + inst.dist(customer, nxt)
                    - inst.dist(prev, nxt)
                )
                if self.noise > 0.0:
                    added *= 1.0 + self.noise * (rng.random() - 0.5) * 2.0
                if not solution._insertion_time_feasible(route, pos, customer):
                    continue
                options.append((added, r_idx, pos + 1))

        # open new route
        added_new = inst.dist(0, customer) + inst.dist(customer, 0)
        if self.noise > 0.0:
            added_new *= 1.0 + self.noise * (rng.random() - 0.5) * 2.0
        options.append((added_new, -1, 0))

        options.sort(key=lambda t: t[0])
        return options

    def repair(
        self, solution: Solution, removed: list[int], rng: np.random.Generator
    ) -> None:
        pending = list(removed)
        while pending:
            best_customer: int | None = None
            best_regret: float = -1e18
            best_option: tuple[float, int, int] | None = None

            for customer in pending:
                options = self._candidate_costs(solution, customer, rng)
                if not options:
                    # no feasible insertion; force on next pass
                    continue
                best_cost = options[0][0]
                # regret = sum over positions 2..k of (cost_k - cost_1)
                k = min(self.k, len(options))
                regret = sum(options[i][0] - best_cost for i in range(1, k))
                # prefer customers with HIGHER regret (more urgent to insert)
                if regret > best_regret:
                    best_regret = regret
                    best_customer = customer
                    best_option = options[0]

            if best_customer is None or best_option is None:
                # no customer has feasible insertion -> force-insert the first pending
                self._force_insert(solution, pending[0], rng)
                pending.remove(pending[0])
                continue

            delta, r_idx, pos = best_option
            solution.insert_customer(best_customer, delta, r_idx, pos)
            pending.remove(best_customer)


class NoisyGreedy(RepairOperator):
    """Greedy insertion with multiplicative noise to escape local minima."""

    name = "noisy_greedy"

    def __init__(self, noise: float = 0.3):
        self.noise = noise

    def repair(
        self, solution: Solution, removed: list[int], rng: np.random.Generator
    ) -> None:
        order = list(removed)
        rng.shuffle(order)
        for customer in order:
            best = solution.best_insertion(customer, noise=self.noise, rng=rng)
            if best is None:
                self._force_insert(solution, customer, rng)
            else:
                delta, r_idx, pos = best
                solution.insert_customer(customer, delta, r_idx, pos)
