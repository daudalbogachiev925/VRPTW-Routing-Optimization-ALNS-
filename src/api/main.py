"""FastAPI: solve VRPTW instances on demand."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    InstanceInfo,
    ModelInfoResponse,
    RouteOut,
    SolveRequest,
    SolveResponse,
)
from src.alns.engine import ALNSEngine
from src.config import AppConfig, get_app_config, get_settings
from src.operators.registry import build_operators
from src.problem.instance import Instance
from src.problem.parser import parse_solomon

logger = logging.getLogger(__name__)
logging.basicConfig(level=get_settings().log_level)


class InstanceCache:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._cache: dict[str, Instance] = {}

    def get(self, name: str) -> Instance:
        if name in self._cache:
            return self._cache[name]
        path = self.base_dir / f"{name}.txt"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Instance '{name}' not found")
        inst = parse_solomon(path)
        self._cache[name] = inst
        return inst

    def list_available(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(p.stem for p in self.base_dir.glob("*.txt"))


@lru_cache(maxsize=1)
def get_instance_cache() -> InstanceCache:
    cfg = get_app_config()
    return InstanceCache(Path(cfg.benchmark.instances_dir))


@lru_cache(maxsize=1)
def get_engine_factory() -> tuple[AppConfig, list, list, list, list]:
    cfg = get_app_config()
    destroy_ops, repair_ops, d_w, r_w = build_operators(cfg.operators)
    return cfg, destroy_ops, repair_ops, d_w, r_w


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # warm up cache
    get_engine_factory()
    get_instance_cache()
    yield


app = FastAPI(title="VRPTW ALNS API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/instances", response_model=list[InstanceInfo])
def list_instances(
    cache: InstanceCache = Depends(get_instance_cache),
) -> list[InstanceInfo]:
    out: list[InstanceInfo] = []
    for name in cache.list_available():
        try:
            inst = cache.get(name)
        except Exception:
            continue
        out.append(
            InstanceInfo(
                name=inst.name,
                n_customers=inst.n,
                capacity=inst.capacity,
                n_vehicles=inst.n_vehicles,
            )
        )
    return out


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    cfg, destroy_ops, repair_ops, _, _ = get_engine_factory()
    return ModelInfoResponse(
        algorithm="ALNS (Adaptive Large Neighborhood Search)",
        n_destroy_ops=len(destroy_ops),
        n_repair_ops=len(repair_ops),
        config_iterations=cfg.alns.iterations,
    )


@app.post("/solve", response_model=SolveResponse)
def solve(
    req: SolveRequest,
    cache: InstanceCache = Depends(get_instance_cache),
) -> SolveResponse:
    cfg, destroy_ops, repair_ops, d_w, r_w = get_engine_factory()
    inst = cache.get(req.instance_name)

    # override iterations for this request
    alns_cfg = cfg.alns.model_copy(update={"iterations": req.iterations})
    engine = ALNSEngine(
        config=alns_cfg,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        destroy_weights=d_w,
        repair_weights=r_w,
    )
    try:
        result = engine.solve(inst)
    except Exception as exc:
        logger.exception("Solve failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Solve failed: {exc}") from exc

    routes_out = [
        RouteOut(
            customers=list(r.customers),
            distance=float(result.best_solution.route_distance(r)),
            load=float(result.best_solution.route_load(r)),
            n_customers=len(r.customers),
        )
        for r in result.best_solution.routes
    ]
    return SolveResponse(
        instance_name=inst.name,
        n_customers=inst.n,
        n_vehicles=result.best_n_vehicles,
        total_distance=result.best_distance,
        feasible=result.best_solution.is_feasible(),
        routes=routes_out,
        elapsed_seconds=result.elapsed_seconds,
    )
