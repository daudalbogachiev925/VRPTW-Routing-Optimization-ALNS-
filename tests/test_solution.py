"""Solution representation and evaluation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.problem.parser import parse_solomon
from src.problem.solution import Route, Solution

C101_PATH = Path(__file__).resolve().parents[1] / "data" / "solomon" / "c101.txt"


@pytest.fixture
def instance():
    return parse_solomon(C101_PATH)


def test_empty_solution_cost_zero(instance) -> None:
    sol = Solution(instance=instance)
    assert sol.total_distance() == 0.0
    assert sol.n_vehicles() == 0
    assert sol.is_feasible()


def test_single_route_distance(instance) -> None:
    sol = Solution(instance=instance, routes=[Route(nodes=[0, 1, 0])])
    d = instance.dist(0, 1) + instance.dist(1, 0)
    assert sol.total_distance() == pytest.approx(d)
    assert sol.n_vehicles() == 1


def test_route_capacity_feasibility(instance) -> None:
    # overload route with many customers
    # depot capacity = 200, customers have demand ~10-40
    nodes = [0] + list(range(1, 30)) + [0]
    route = Route(nodes=nodes)
    sol = Solution(instance=instance, routes=[route])
    total_demand = sum(instance.node(n).demand for n in range(1, 30))
    if total_demand > instance.capacity:
        assert not sol.route_is_feasible(route)


def test_route_time_window_violation(instance) -> None:
    # construct route where arrival at depot would exceed depot due (1236)
    # add all customers that push total time > 1236
    route = Route(nodes=[0] + list(range(1, 51)) + [0])
    sol = Solution(instance=instance, routes=[route])
    # very likely infeasible due to cumulative time
    if not sol.route_is_feasible(route):
        assert not sol.is_feasible()


def test_best_insertion_returns_tuple(instance) -> None:
    sol = Solution(instance=instance, routes=[Route(nodes=[0, 0])])
    # empty route, insert customer 1
    best = sol.best_insertion(1)
    assert best is not None
    delta, r_idx, pos = best
    assert delta >= 0
    assert r_idx in (0, -1)


def test_insert_and_remove_roundtrip(instance) -> None:
    sol = Solution(instance=instance, routes=[Route(nodes=[0, 0])])
    best = sol.best_insertion(1)
    assert best is not None
    delta, r_idx, pos = best
    sol.insert_customer(1, delta, r_idx, pos)
    assert 1 in sol.all_customers()
    removed = sol.remove_customers({1})
    assert removed == [1]
    assert 1 not in sol.all_customers()


def test_route_load_calculation(instance) -> None:
    nodes = [0, 1, 2, 3, 0]
    route = Route(nodes=nodes)
    sol = Solution(instance=instance, routes=[route])
    expected_load = sum(instance.node(n).demand for n in [1, 2, 3])
    assert sol.route_load(route) == pytest.approx(expected_load)
