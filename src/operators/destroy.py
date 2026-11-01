"""Destroy operators: remove a subset of customers from the solution."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.problem.instance import Instance
from src.problem.solution import Route, Solution


class DestroyOperator(ABC):
    name: str = "base"

    @abstractmethod
    def destroy(self, solution: Solution, rng: np.random.Generator) -> list[int]:
        """Remove customers in-place; return list of removed customer IDs."""


def _removal_size(inst: Instance, rng: np.random.Generator, frac: float, lo: int = 4, hi: int = 40) -> int:
    upper = max(lo, min(hi, int(inst.n * frac)))
    return int(rng.integers(lo, upper + 1))


class RandomRemoval(DestroyOperator):
    name = "random_removal"

    def __init__(self, max_removal_frac: float = 0.15, min_removal: int = 4, max_removal: int = 40):
        self.frac = max_removal_frac
        self.lo = min_removal
        self.hi = max_removal

    def destroy(self, solution: Solution, rng: np.random.Generator) -> list[int]:
        n_remove = _removal_size(solution.instance, rng, self.frac, self.lo, self.hi)
        all_cust = list(solution.all_customers())
        if len(all_cust) <= n_remove:
            to_remove = set(all_cust)
        else:
            to_remove = set(rng.choice(all_cust, size=n_remove, replace=False).tolist())
        return solution.remove_customers(to_remove)


class WorstRemoval(DestroyOperator):
    name = "worst_removal"

    def __init__(self, max_removal_frac: float = 0.15, p_worst: float = 3.0):
        self.frac = max_removal_frac
        self.p_worst = p_worst

    def _customer_savings(self, solution: Solution) -> dict[int, float]:
        inst = solution.instance
        savings: dict[int, float] = {}
        for route in solution.routes:
            nodes = route.nodes
            for k in range(1, len(nodes) - 1):
                prev, cur, nxt = nodes[k - 1], nodes[k], nodes[k + 1]
                removed = inst.dist(prev, cur) + inst.dist(cur, nxt)
                added = inst.dist(prev, nxt)
                savings[cur] = removed - added
        return savings

    def destroy(self, solution: Solution, rng: np.random.Generator) -> list[int]:
        n_remove = _removal_size(solution.instance, rng, self.frac, 4, 40)
        savings = self._customer_savings(solution)
        if not savings:
            return []
        customers = list(savings.keys())
        values = np.array([savings[c] for c in customers])
        # rank: worst = highest savings
        order = np.argsort(-values)
        removed: list[int] = []
        # pick with randomness
        while len(removed) < n_remove and len(order) > 0:
            idx = int(rng.integers(0, len(order)))
            # bias toward worst via exponent
            pos = int((rng.random() ** self.p_worst) * len(order))
            cust = customers[order[pos]]
            if cust in removed:
                order = np.delete(order, pos)
                continue
            removed.append(cust)
            order = np.delete(order, pos)
        return solution.remove_customers(set(removed))


class ShawRemoval(DestroyOperator):
    """Remove a cluster of similar customers (relatedness in space + time + demand)."""

    name = "shaw_removal"

    def __init__(self, max_removal_frac: float = 0.15, p_shaw: float = 4.0, c_bar: float = 20.0):
        self.frac = max_removal_frac
        self.p_shaw = p_shaw
        self.c_bar = c_bar

    def _relatedness(self, inst: Instance, a: int, b: int) -> float:
        ca, cb = inst.node(a), inst.node(b)
        spatial = inst.dist(a, b) / max(1.0, self.c_bar)
        time_diff = abs(ca.ready - cb.ready) / max(1.0, inst.depot.due)
        demand_diff = abs(ca.demand - cb.demand) / max(1.0, inst.capacity)
        return spatial + time_diff + demand_diff

    def destroy(self, solution: Solution, rng: np.random.Generator) -> list[int]:
        inst = solution.instance
        all_cust = list(solution.all_customers())
        if not all_cust:
            return []
        n_remove = _removal_size(inst, rng, self.frac, 4, 40)
        seed = int(rng.choice(all_cust))
        # sort by relatedness to seed
        related = sorted(all_cust, key=lambda c: self._relatedness(inst, seed, c))
        removed: list[int] = [seed]
        while len(removed) < n_remove and len(related) > 0:
            pos = int((rng.random() ** self.p_shaw) * len(related))
            candidate = related[pos]
            if candidate in removed:
                related.pop(pos)
                continue
            removed.append(candidate)
            related.pop(pos)
        return solution.remove_customers(set(removed))


class RouteRemoval(DestroyOperator):
    """Remove entire routes (typically the ones with fewest customers / shortest distance)."""

    name = "route_removal"

    def __init__(self, max_remove_routes: int = 2):
        self.max_remove_routes = max_remove_routes

    def destroy(self, solution: Solution, rng: np.random.Generator) -> list[int]:
        n_routes = len(solution.routes)
        if n_routes <= 1:
            return []
        k = int(min(n_routes - 1, rng.integers(1, self.max_remove_routes + 1)))
        # pick k routes with smallest customer counts (destroys the weakest routes)
        sorted_routes = sorted(range(n_routes), key=lambda r: len(solution.routes[r].customers))
        to_remove_customers: set[int] = set()
        # allow randomness in which routes to remove
        pool = sorted_routes[: min(n_routes, max(k, k * 2))]
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        for r_idx in chosen:
            to_remove_customers.update(solution.routes[r_idx].customers)
        return solution.remove_customers(to_remove_customers)
