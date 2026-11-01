"""Solve a single Solomon instance with ALNS."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alns.engine import ALNSEngine
from src.config import ensure_dirs, get_app_config, get_settings
from src.operators.registry import build_operators
from src.problem.parser import parse_solomon
from src.viz.routes import plot_convergence, plot_solution

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vrptw.solve")


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a Solomon VRPTW instance with ALNS")
    parser.add_argument("--instance", required=True, help="Instance name, e.g. c101")
    parser.add_argument("--iterations", type=int, default=None, help="Override iterations")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    ensure_dirs()
    cfg = get_app_config()
    path = Path(cfg.benchmark.instances_dir) / f"{args.instance}.txt"
    if not path.exists():
        logger.error("Instance not found: %s", path)
        return 1

    inst = parse_solomon(path)
    alns_cfg = cfg.alns.model_copy(
        update={"iterations": args.iterations or cfg.alns.iterations}
    )
    destroy_ops, repair_ops, d_w, r_w = build_operators(cfg.operators)
    engine = ALNSEngine(
        config=alns_cfg,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        destroy_weights=d_w,
        repair_weights=r_w,
    )
    result = engine.solve(inst)

    print()
    print(f"=== Result: {inst.name} ===")
    print(f"  Customers:     {inst.n}")
    print(f"  Vehicles:      {result.best_n_vehicles}")
    print(f"  Distance:      {result.best_distance:.2f}")
    print(f"  Feasible:      {result.best_solution.is_feasible()}")
    print(f"  Iterations:    {result.iterations_run}")
    print(f"  Time:          {result.elapsed_seconds:.1f}s")
    print()
    print("Destroy weights:", [f"{w:.2f}" for w in result.destroy_weights])
    print("Repair weights: ", [f"{w:.2f}" for w in result.repair_weights])

    if not args.no_plot:
        out_dir = Path(get_settings().artifacts_dir) / inst.name
        plot_solution(result.best_solution, out_path=out_dir / "routes.png")
        plot_convergence(result.history, out_path=out_dir / "convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
