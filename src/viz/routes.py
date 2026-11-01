"""Plot VRPTW solution: depot, customers, colored routes."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.problem.solution import Solution

logger = logging.getLogger(__name__)


def plot_solution(
    solution: Solution,
    title: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    inst = solution.instance
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot all customers as faint dots
    xs = [c.x for c in inst.customers]
    ys = [c.y for c in inst.customers]
    ax.scatter(xs, ys, s=15, color="lightgray", zorder=1)

    # depot
    ax.scatter(inst.depot.x, inst.depot.y, s=200, marker="s", color="black", zorder=3)
    ax.annotate("Depot", (inst.depot.x, inst.depot.y), xytext=(8, 8), textcoords="offset points")

    # routes
    cmap = plt.get_cmap("tab20")
    for k, route in enumerate(solution.routes):
        color = cmap(k % 20)
        coords = [(inst.node(n).x, inst.node(n).y) for n in route.nodes]
        xs_r = [c[0] for c in coords]
        ys_r = [c[1] for c in coords]
        ax.plot(xs_r, ys_r, "-o", color=color, markersize=4, linewidth=1.2, zorder=2)

    if title is None:
        title = (
            f"{inst.name}: {solution.n_vehicles()} vehicles, "
            f"distance={solution.total_distance():.2f}"
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        logger.info("Saved route plot -> %s", out_path)
    if show:
        plt.show()
    return fig


def plot_convergence(history: list[float], out_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    # plot log of cost for readability when vehicles term dominates
    ys = np.array(history, dtype=float)
    ax.plot(np.arange(len(ys)), ys, color="steelblue", linewidth=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost (vehicles*1e6 + distance)")
    ax.set_title("ALNS convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig
