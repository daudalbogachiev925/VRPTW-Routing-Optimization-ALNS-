"""FastAPI tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # point config to the repo's data directory
    repo_data = Path(__file__).resolve().parents[1] / "data" / "solomon"
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        f"""
alns:
  iterations: 300
  segment_length: 50
  reaction_factor: 0.1
  seed: 42
operators:
  destroy:
    - name: random_removal
      weight: 1.0
    - name: worst_removal
      weight: 1.0
  repair:
    - name: greedy_insertion
      weight: 1.0
    - name: regret_2_insertion
      weight: 1.0
benchmark:
  instances_dir: "{repo_data}"
  instances: ["c101"]
  default_iterations: 300
  output_dir: "{tmp_path / 'artifacts'}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(cfg_yaml))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    from src.config import get_settings, get_app_config
    from src.api import main as api_main

    get_settings.cache_clear()
    get_app_config.cache_clear()
    api_main.get_instance_cache.cache_clear()
    api_main.get_engine_factory.cache_clear()

    with TestClient(api_main.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "ALNS" in data["algorithm"]
    assert data["n_destroy_ops"] >= 2
    assert data["n_repair_ops"] >= 2


def test_instances(client: TestClient) -> None:
    r = client.get("/instances")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert any(i["name"] == "c101" for i in data)


def test_solve_ok(client: TestClient) -> None:
    body = {"instance_name": "c101", "iterations": 300}
    r = client.post("/solve", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["instance_name"] == "c101"
    assert data["n_customers"] == 100
    assert data["n_vehicles"] > 0
    assert data["total_distance"] > 0
    assert data["feasible"] is True
    assert len(data["routes"]) == data["n_vehicles"]


def test_solve_unknown_instance(client: TestClient) -> None:
    r = client.post("/solve", json={"instance_name": "zzz999", "iterations": 100})
    assert r.status_code == 404


def test_solve_invalid_iterations(client: TestClient) -> None:
    r = client.post("/solve", json={"instance_name": "c101", "iterations": 10})
    assert r.status_code == 422
