"""Instance parsing tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.problem.parser import parse_solomon

C101_PATH = Path(__file__).resolve().parents[1] / "data" / "solomon" / "c101.txt"


def test_parse_c101() -> None:
    inst = parse_solomon(C101_PATH)
    assert inst.name == "c101"
    assert inst.n == 100
    assert inst.capacity == 200.0
    assert inst.n_vehicles == 25
    assert inst.depot.id == 0
    assert inst.depot.x == 40
    assert inst.depot.y == 50


def test_distance_matrix() -> None:
    inst = parse_solomon(C101_PATH)
    assert inst.distance_matrix is not None
    assert inst.distance_matrix.shape == (101, 101)
    # diagonal zero
    assert inst.distance_matrix[0, 0] == 0.0
    # symmetry (Euclidean)
    assert inst.distance_matrix[1, 2] == pytest.approx(inst.distance_matrix[2, 1])


def test_node_access() -> None:
    inst = parse_solomon(C101_PATH)
    depot = inst.node(0)
    c1 = inst.node(1)
    assert depot.id == 0
    assert c1.id == 1
    assert c1.demand == 10.0


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_solomon("/nonexistent/path.txt")
