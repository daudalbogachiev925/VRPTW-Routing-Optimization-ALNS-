"""ALNS engine end-to-end tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.alns.acceptance import GreedyAcceptance, SimulatedAnnealing
from src.alns.engine import ALNSEngine, initial_solution
from src.alns.weights import AdaptiveWeights
from src.config import ALNSConfig, get_app_config
from src.operators.registry import build_operators
from src.problem.parser import parse_solomon

C101_PATH = Path(__file__).resolve().parents[1] / "data" / "solomon" / "c101.txt"


@pytest.fixture(scope="module")
def instance():
    return parse_solomon(C101_PATH)


def test_sa_accepts_improvement() -> None:
    sa = SimulatedAnnealing(initial_temp=100.0)
    rng = __import__("numpy").random.default_rng(0)
    assert sa.accept(100.0, 90.0, rng)


def test_sa_rejects_large_worsening_cold() -> None:
    sa = SimulatedAnnealing(initial_temp=0.01)
    rng = __import__("numpy").random.default_rng(0)
    assert not sa.accept(100.0, 1000.0, rng)


def test_sa_cools() -> None:
    sa = SimulatedAnnealing(initial_temp=100.0, cooling=0.5)
    sa.cool()
    assert sa.temp == pytest.approx(50.0)


def test_greedy_acceptance() -> None:
    g = GreedyAcceptance()
    rng = __import__("numpy").random.default_rng(0)
    assert g.accept(100.0, 90.0, rng)
    assert not g.accept(100.0, 110.0, rng)


def test_adaptive_weights_selection() -> None:
    w = AdaptiveWeights(
        n_operators=4,
        initial_weights=[1.0, 1.0, 1.0, 1.0],
        reaction_factor=0.1,
        segment_length=10,
    )
    rng = __import__("numpy").random.default_rng(0)
    picks = [w.select(rng) for _ in range(200)]
    assert all(0 <= p < 4 for p in picks)
    # all operators should get picked at least once
    assert len(set(picks)) >= 3


def test_adaptive_weights_segment_update() -> None:
    w = AdaptiveWeights(
        n_operators=2,
        initial_weights=[1.0, 1.0],
        reaction_factor=0.5,
        segment_length=10,
    )
    w.record(0, 100.0)
    w.record(0, 100.0)
    w.record(1, 1.0)
    old_w0 = w.weights[0]
    w.end_segment()
    # op 0 got much higher score, weight should increase
    assert w.weights[0] > w.weights[1]


def test_initial_solution_feasible(instance) -> None:
    rng = __import__("numpy").random.default_rng(0)
    sol = initial_solution(instance, rng)
    assert sol.is_feasible()
    assert len(sol.all_customers()) == instance.n


def test_alns_short_run(instance) -> None:
    """Quick smoke test: run 200 iterations, expect feasible non-empty solution."""
    cfg = get_app_config()
    alns_cfg = cfg.alns.model_copy(update={"iterations": 200, "seed": 42})
    destroy_ops, repair_ops, d_w, r_w = build_operators(cfg.operators)
    engine = ALNSEngine(
        config=alns_cfg,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        destroy_weights=d_w,
        repair_weights=r_w,
    )
    result = engine.solve(instance)
    assert result.best_solution.is_feasible()
    assert len(result.best_solution.all_customers()) == instance.n
    assert result.best_distance > 0
    assert result.best_n_vehicles > 0
    assert result.iterations_run == 200


def test_alns_improves_over_iterations(instance) -> None:
    """Best cost after 1000 iterations should be <= initial cost."""
    cfg = get_app_config()
    alns_cfg = cfg.alns.model_copy(update={"iterations": 1000, "seed": 123})
    destroy_ops, repair_ops, d_w, r_w = build_operators(cfg.operators)
    engine = ALNSEngine(
        config=alns_cfg,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        destroy_weights=d_w,
        repair_weights=r_w,
    )
    result = engine.solve(instance)
    assert result.best_cost <= result.history[0] + 1e-6
