"""Acceptance criteria: simulated annealing or greedy."""
from __future__ import annotations

import math

import numpy as np


class SimulatedAnnealing:
    """SA acceptance with geometric cooling."""

    def __init__(self, initial_temp: float = 100.0, cooling: float = 0.99975):
        self.temp = float(initial_temp)
        self.initial_temp = float(initial_temp)
        self.cooling = float(cooling)

    def accept(self, current_cost: float, candidate_cost: float, rng: np.random.Generator) -> bool:
        if candidate_cost <= current_cost:
            return True
        delta = candidate_cost - current_cost
        if self.temp <= 1e-12:
            return False
        p = math.exp(-delta / self.temp)
        return bool(rng.random() < p)

    def cool(self) -> None:
        self.temp *= self.cooling


class GreedyAcceptance:
    """Accept only improvements."""

    def accept(self, current_cost: float, candidate_cost: float, rng: np.random.Generator) -> bool:
        return candidate_cost <= current_cost

    def cool(self) -> None:
        pass
