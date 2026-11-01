"""Adaptive operator weights (roulette-wheel selection with segment updates)."""
from __future__ import annotations

import numpy as np


class AdaptiveWeights:
    """Segment-based adaptive weights as in Ropke & Pisinger (2006)."""

    def __init__(
        self,
        n_operators: int,
        initial_weights: list[float],
        reaction_factor: float = 0.1,
        segment_length: int = 100,
        score_new_best: float = 33.0,
        score_better: float = 9.0,
        score_accepted: float = 13.0,
        score_rejected: float = 0.0,
    ):
        if len(initial_weights) != n_operators:
            raise ValueError("initial_weights length mismatch")
        self.n = n_operators
        self.weights = np.array(initial_weights, dtype=np.float64)
        self.scores = np.zeros(n_operators, dtype=np.float64)
        self.uses = np.zeros(n_operators, dtype=np.int64)
        self.reaction_factor = reaction_factor
        self.segment_length = segment_length
        self.score_new_best = score_new_best
        self.score_better = score_better
        self.score_accepted = score_accepted
        self.score_rejected = score_rejected

    def select(self, rng: np.random.Generator) -> int:
        total = self.weights.sum()
        if total <= 0:
            self.weights[:] = 1.0
            total = float(self.n)
        probs = self.weights / total
        return int(rng.choice(self.n, p=probs))

    def record(self, op_idx: int, score: float) -> None:
        self.scores[op_idx] += score
        self.uses[op_idx] += 1

    def end_segment(self) -> None:
        """Update weights based on accumulated segment scores."""
        for i in range(self.n):
            if self.uses[i] > 0:
                avg = self.scores[i] / self.uses[i]
                self.weights[i] = (
                    (1.0 - self.reaction_factor) * self.weights[i]
                    + self.reaction_factor * avg
                )
        # reset segment accumulators
        self.scores[:] = 0.0
        self.uses[:] = 0
        # clamp small weights
        self.weights = np.clip(self.weights, 0.05, None)
