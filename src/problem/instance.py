"""VRPTW instance: depot, customers, vehicle capacity."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


@dataclass
class Instance:
    name: str
    depot: Customer
    customers: list[Customer]
    capacity: float
    n_vehicles: int
    distance_matrix: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        if self.distance_matrix is None:
            self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        nodes = [self.depot] + self.customers
        n = len(nodes)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mat[i, j] = math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
        return mat

    @property
    def n(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.customers)

    @property
    def n_nodes(self) -> int:
        """Depot + customers."""
        return len(self.customers) + 1

    def dist(self, i: int, j: int) -> float:
        return float(self.distance_matrix[i, j])  # type: ignore[index]

    def node(self, idx: int) -> Customer:
        if idx == 0:
            return self.depot
        return self.customers[idx - 1]
