"""Solution representation and evaluation.

A route is [0, c1, c2, ..., ck, 0]; depot is index 0. Customers are 1..n.
Solution = list of routes; feasibility checked incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.problem.instance import Instance


@dataclass
class Route:
    nodes: list[int] = field(default_factory=list)  # includes leading/trailing depot 0

    def __post_init__(self) -> None:
        if not self.nodes:
            self.nodes = [0, 0]

    @property
    def customers(self) -> list[int]:
        return self.nodes[1:-1]

    def copy(self) -> "Route":
        return Route(nodes=list(self.nodes))


@dataclass
class Solution:
    instance: Instance
    routes: list[Route] = field(default_factory=list)

    def copy(self) -> "Solution":
        return Solution(instance=self.instance, routes=[r.copy() for r in self.routes])

    # ---- Evaluation ----------------------------------------------------
    def total_distance(self) -> float:
        total = 0.0
        inst = self.instance
        for route in self.routes:
            for k in range(len(route.nodes) - 1):
                total += inst.dist(route.nodes[k], route.nodes[k + 1])
        return total

    def n_vehicles(self) -> int:
        return len(self.routes)

    def cost(self) -> float:
        """Objective: primarily minimize vehicles, then distance.

        Standard VRPTW scoring: lexicographic (vehicles, distance).
        We use a large constant to prioritize vehicle count.
        """
        return self.n_vehicles() * 1e6 + self.total_distance()

    # ---- Feasibility ---------------------------------------------------
    def is_feasible(self) -> bool:
        for route in self.routes:
            if not self.route_is_feasible(route):
                return False
        return True

    def route_is_feasible(self, route: Route) -> bool:
        inst = self.instance
        load = 0.0
        time = 0.0
        prev = route.nodes[0]
        for node in route.nodes[1:]:
            cust = inst.node(node)
            travel = inst.dist(prev, node)
            arrival = time + travel
            if node == 0:
                # return to depot: must arrive before depot due
                if arrival > inst.depot.due + 1e-6:
                    return False
                time = arrival
                prev = node
                continue
            load += cust.demand
            if load > inst.capacity + 1e-6:
                return False
            start_service = max(arrival, cust.ready)
            if start_service > cust.due + 1e-6:
                return False
            time = start_service + cust.service
            prev = node
        return True

    # ---- Route metrics -------------------------------------------------
    def route_distance(self, route: Route) -> float:
        inst = self.instance
        return sum(
            inst.dist(route.nodes[k], route.nodes[k + 1])
            for k in range(len(route.nodes) - 1)
        )

    def route_load(self, route: Route) -> float:
        inst = self.instance
        return sum(inst.node(n).demand for n in route.customers)

    def route_arrival_times(self, route: Route) -> list[float]:
        inst = self.instance
        times = [0.0]
        time = 0.0
        prev = route.nodes[0]
        for node in route.nodes[1:]:
            arrival = time + inst.dist(prev, node)
            times.append(arrival)
            if node != 0:
                time = max(arrival, inst.node(node).ready) + inst.node(node).service
                prev = node
            else:
                time = arrival
                prev = node
        return times

    # ---- Insertion helpers --------------------------------------------
    def best_insertion(
        self, customer: int, noise: float = 0.0, rng: np.random.Generator | None = None
    ) -> tuple[float, int, int] | None:
        """Find best feasible (delta_cost, route_idx, position).

        Returns None if no feasible insertion exists.
        """
        inst = self.instance
        best: tuple[float, int, int] | None = None
        cust = inst.node(customer)
        for r_idx, route in enumerate(self.routes):
            # skip routes that would exceed capacity
            if self.route_load(route) + cust.demand > inst.capacity + 1e-6:
                continue
            for pos in range(len(route.nodes) - 1):
                prev = route.nodes[pos]
                nxt = route.nodes[pos + 1]
                added = inst.dist(prev, customer) + inst.dist(customer, nxt) - inst.dist(prev, nxt)
                if noise > 0.0 and rng is not None:
                    added *= 1.0 + noise * (rng.random() - 0.5) * 2.0
                # time feasibility check
                if not self._insertion_time_feasible(route, pos, customer):
                    continue
                if best is None or added < best[0]:
                    best = (added, r_idx, pos + 1)
        # also consider opening a new route if vehicles not exceeded
        if len(self.routes) < inst.n_vehicles or True:  # allow extra if needed
            # delta cost = dist(0,c) + dist(c,0)
            added = inst.dist(0, customer) + inst.dist(customer, 0)
            if noise > 0.0 and rng is not None:
                added *= 1.0 + noise * (rng.random() - 0.5) * 2.0
            if best is None or added < best[0]:
                best = (added, -1, 0)
        return best

    def _insertion_time_feasible(self, route: Route, pos: int, customer: int) -> bool:
        inst = self.instance
        cust = inst.node(customer)
        if cust.demand > inst.capacity + 1e-6:
            return False
        times = self.route_arrival_times(route)
        # arrival at prev and nxt
        prev = route.nodes[pos]
        nxt = route.nodes[pos + 1]
        arrive_at_prev = times[pos]
        start_prev = (
            max(arrive_at_prev, inst.node(prev).ready) + inst.node(prev).service
            if prev != 0
            else arrive_at_prev
        )
        arrive_cust = start_prev + inst.dist(prev, customer)
        if arrive_cust > cust.due + 1e-6:
            return False
        start_cust = max(arrive_cust, cust.ready) + cust.service
        arrive_nxt = start_cust + inst.dist(customer, nxt)
        # check downstream propagation
        # simplified: only check immediate next
        if nxt == 0:
            if arrive_nxt > inst.depot.due + 1e-6:
                return False
        else:
            nxt_node = inst.node(nxt)
            if max(arrive_nxt, nxt_node.ready) > nxt_node.due + 1e-6:
                return False
        return True

    def insert_customer(
        self, customer: int, delta: float, route_idx: int, position: int
    ) -> None:
        if route_idx == -1:
            self.routes.append(Route(nodes=[0, customer, 0]))
        else:
            self.routes[route_idx].nodes.insert(position, customer)

    def remove_customers(self, customers: set[int]) -> list[int]:
        """Remove customers from routes; remove empty routes. Return removed in order."""
        removed: list[int] = []
        new_routes: list[Route] = []
        for route in self.routes:
            kept = [route.nodes[0]]
            for n in route.nodes[1:-1]:
                if n in customers:
                    removed.append(n)
                else:
                    kept.append(n)
            kept.append(route.nodes[-1])
            if len(kept) > 2:  # has at least one customer
                new_routes.append(Route(nodes=kept))
        self.routes = new_routes
        return removed

    def all_customers(self) -> set[int]:
        out: set[int] = set()
        for route in self.routes:
            out.update(route.customers)
        return out

    def customers_per_route(self) -> list[list[int]]:
        return [route.customers for route in self.routes]
