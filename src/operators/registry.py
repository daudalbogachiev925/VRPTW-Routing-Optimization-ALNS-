"""Operator registry: build operator instances from config specs."""
from __future__ import annotations

import logging

from src.config import OperatorSpec, OperatorsConfig
from src.operators.destroy import (
    DestroyOperator,
    RandomRemoval,
    RouteRemoval,
    ShawRemoval,
    WorstRemoval,
)
from src.operators.repair import (
    GreedyInsertion,
    NoisyGreedy,
    RegretInsertion,
    RepairOperator,
)

logger = logging.getLogger(__name__)


def _build_destroy(spec: OperatorSpec) -> DestroyOperator:
    name = spec.name
    params = spec.params
    if name == "random_removal":
        return RandomRemoval(
            max_removal_frac=float(params.get("max_removal_frac", 0.15)),
            min_removal=int(params.get("min_removal", 4)),
            max_removal=int(params.get("max_removal", 40)),
        )
    if name == "worst_removal":
        return WorstRemoval(
            max_removal_frac=float(params.get("max_removal_frac", 0.15)),
            p_worst=float(params.get("p_worst", 3.0)),
        )
    if name == "shaw_removal":
        return ShawRemoval(
            max_removal_frac=float(params.get("max_removal_frac", 0.15)),
            p_shaw=float(params.get("p_shaw", 4.0)),
            c_bar=float(params.get("c_bar", 20.0)),
        )
    if name == "route_removal":
        return RouteRemoval(max_remove_routes=int(params.get("max_remove_routes", 2)))
    raise ValueError(f"Unknown destroy operator: {name}")


def _build_repair(spec: OperatorSpec) -> RepairOperator:
    name = spec.name
    params = spec.params
    if name == "greedy_insertion":
        return GreedyInsertion(noise=float(params.get("noise", 0.0)))
    if name == "regret_2_insertion":
        return RegretInsertion(k=2, noise=float(params.get("noise", 0.0)))
    if name == "regret_3_insertion":
        return RegretInsertion(k=3, noise=float(params.get("noise", 0.0)))
    if name == "noisy_greedy":
        return NoisyGreedy(noise=float(params.get("noise", 0.3)))
    raise ValueError(f"Unknown repair operator: {name}")


def build_operators(
    cfg: OperatorsConfig,
) -> tuple[list[DestroyOperator], list[RepairOperator], list[float], list[float]]:
    """Build operator instances with their initial weights."""
    if not cfg.destroy:
        cfg.destroy = [
            OperatorSpec(name="random_removal"),
            OperatorSpec(name="worst_removal"),
            OperatorSpec(name="shaw_removal"),
            OperatorSpec(name="route_removal"),
        ]
    if not cfg.repair:
        cfg.repair = [
            OperatorSpec(name="greedy_insertion"),
            OperatorSpec(name="regret_2_insertion"),
            OperatorSpec(name="regret_3_insertion"),
            OperatorSpec(name="noisy_greedy"),
        ]
    destroy_ops = [_build_destroy(s) for s in cfg.destroy]
    repair_ops = [_build_repair(s) for s in cfg.repair]
    destroy_weights = [float(s.weight) for s in cfg.destroy]
    repair_weights = [float(s.weight) for s in cfg.repair]
    logger.info(
        "Built %s destroy, %s repair operators",
        len(destroy_ops),
        len(repair_ops),
    )
    return destroy_ops, repair_ops, destroy_weights, repair_weights
