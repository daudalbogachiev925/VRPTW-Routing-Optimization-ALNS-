"""Run ALNS on multiple Solomon instances and produce a comparison table."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.alns.engine import ALNSEngine
from src.config import ensure_dirs, get_app_config, get_settings
from src.operators.registry import build_operators
from src.problem.parser import parse_solomon
from src.viz.routes import plot_convergence, plot_solution

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vrptw.benchmark")

# Best-known solutions (from Solomon 1987 / CVRPLIB).
# Format: (n_vehicles, total_distance)
BEST_KNOWN: dict[str, tuple[int, float]] = {
    "c101": (10, 828.94),
    "c201": (3, 591.56),
    "r101": (19, 1650.80),
    "rc101": (14, 1696.95),
    "c102": (10, 828.94),
    "r201": (4, 1252.37),
    "rc201": (4, 1406.94),
}


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()
    out_dir = Path(cfg.benchmark.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in cfg.benchmark.instances:
        path = Path(cfg.benchmark.instances_dir) / f"{name}.txt"
        if not path.exists():
            logger.warning("Skip %s: file missing", name)
            continue
        logger.info("=== Solving %s ===", name)
        inst = parse_solomon(path)

        alns_cfg = cfg.alns.model_copy(update={"iterations": cfg.benchmark.default_iterations})
        destroy_ops, repair_ops, d_w, r_w = build_operators(cfg.operators)
        engine = ALNSEngine(
            config=alns_cfg,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            destroy_weights=d_w,
            repair_weights=r_w,
        )
        t0 = time.time()
        result = engine.solve(inst)
        elapsed = time.time() - t0

        best = BEST_KNOWN.get(name)
        if best is not None:
            bk_veh, bk_dist = best
            gap_veh = (result.best_n_vehicles - bk_veh) / max(1, bk_veh) * 100
            gap_dist = (result.best_distance - bk_dist) / bk_dist * 100
        else:
            bk_veh, bk_dist = None, None
            gap_veh = None
            gap_dist = None

        rows.append(
            {
                "instance": name,
                "n_customers": inst.n,
                "vehicles": result.best_n_vehicles,
                "distance": round(result.best_distance, 2),
                "best_known_vehicles": bk_veh,
                "best_known_distance": bk_dist,
                "gap_vehicles_pct": round(gap_veh, 2) if gap_veh is not None else None,
                "gap_distance_pct": round(gap_dist, 2) if gap_dist is not None else None,
                "feasible": result.best_solution.is_feasible(),
                "iterations": result.iterations_run,
                "elapsed_s": round(elapsed, 1),
            }
        )

        # plots
        instance_dir = out_dir / name
        plot_solution(result.best_solution, out_path=instance_dir / "routes.png")
        plot_convergence(result.history, out_path=instance_dir / "convergence.png")

    table = pl.DataFrame(rows)
    print()
    print("=== Benchmark Summary ===")
    print(table)
    table.write_csv(out_dir / "benchmark.csv")
    (out_dir / "benchmark.json").write_text(json.dumps(rows, indent=2))
    logger.info("Results saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
