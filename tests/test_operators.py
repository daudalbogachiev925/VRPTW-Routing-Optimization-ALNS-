"""Destroy/repair operator tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.alns.engine import initial_solution
from src.operators.destroy import (
    RandomRemoval,
    RouteRemoval,
    ShawRemoval,
    WorstRemoval,
)
from src.operators.repair import GreedyInsertion, NoisyGreedy, RegretInsertion
from src.problem.parser import parse_solomon

C101_PATH = Path(__file__).resolve().parents[1] / "data" / "solomon" / "c101.txt"


@pytest.fixture(scope="module")
def instance():
    return parse_solomon(C101_PATH)


@pytest.fixture(scope="module")
def initial(instance):
    rng = np.random.default_rng(0)
    return initial_solution(instance, rng)


def test_initial_solution_feasible(instance, initial) -> None:
    assert initial.is_feasible()
    assert len(initial.all_customers()) == instance.n


def test_random_removal(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(1)
    op = RandomRemoval(max_removal_frac=0.1)
    removed = op.destroy(sol, rng)
    assert len(removed) > 0
    assert len(sol.all_customers()) == instance.n - len(removed)


def test_worst_removal(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(2)
    op = WorstRemoval(max_removal_frac=0.1)
    removed = op.destroy(sol, rng)
    assert len(removed) > 0
    assert len(sol.all_customers()) == instance.n - len(removed)


def test_shaw_removal(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(3)
    op = ShawRemoval(max_removal_frac=0.1)
    removed = op.destroy(sol, rng)
    assert len(removed) > 0


def test_route_removal(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(4)
    op = RouteRemoval(max_remove_routes=1)
    removed = op.destroy(sol, rng)
    assert len(removed) > 0


def test_greedy_repair(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(5)
    removed = RandomRemoval(max_removal_frac=0.1).destroy(sol, rng)
    GreedyInsertion().repair(sol, removed, rng)
    assert len(sol.all_customers()) == instance.n


def test_regret_repair(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(6)
    removed = RandomRemoval(max_removal_frac=0.1).destroy(sol, rng)
    RegretInsertion(k=3).repair(sol, removed, rng)
    assert len(sol.all_customers()) == instance.n


def test_noisy_repair(instance, initial) -> None:
    sol = initial.copy()
    rng = np.random.default_rng(7)
    removed = RandomRemoval(max_removal_frac=0.1).destroy(sol, rng)
    NoisyGreedy(noise=0.3).repair(sol, removed, rng)
    assert len(sol.all_customers()) == instance.n
