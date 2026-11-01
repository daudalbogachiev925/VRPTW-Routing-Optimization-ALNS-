"""Solomon CVRPTW instance parser.

Format:
    NAME
    VEHICLE
    NUMBER CAPACITY
    ...
    CUSTOMER
    CUST NO. XCOORD YCOORD DEMAND READY TIME DUE DATE SERVICE TIME
    0 depot_x depot_y 0 0 due 0
    1 ... ...
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.problem.instance import Customer, Instance

logger = logging.getLogger(__name__)


def parse_solomon(path: str | Path) -> Instance:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Instance file not found: {p}")
    lines = [line.rstrip() for line in p.read_text(encoding="utf-8").splitlines()]

    name = p.stem
    capacity = 0.0
    n_vehicles = 0
    customers: list[Customer] = []
    depot: Customer | None = None
    in_customer_section = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        upper = line.upper()
        if upper.startswith("VEHICLE"):
            # next non-empty line: NUMBER CAPACITY
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                parts = lines[i].split()
                if len(parts) >= 2:
                    n_vehicles = int(parts[0])
                    capacity = float(parts[1])
        elif upper.startswith("CUSTOMER"):
            in_customer_section = True
        elif in_customer_section:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    cust_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    demand = float(parts[3])
                    ready = float(parts[4])
                    due = float(parts[5])
                    service = float(parts[6])
                except ValueError:
                    i += 1
                    continue
                cust = Customer(
                    id=cust_id,
                    x=x,
                    y=y,
                    demand=demand,
                    ready=ready,
                    due=due,
                    service=service,
                )
                if cust_id == 0 and depot is None:
                    depot = cust
                else:
                    customers.append(cust)
        i += 1

    if depot is None:
        raise ValueError(f"No depot found in {p}")
    if capacity <= 0:
        raise ValueError(f"Invalid capacity in {p}")
    if not customers:
        raise ValueError(f"No customers found in {p}")

    instance = Instance(
        name=name,
        depot=depot,
        customers=customers,
        capacity=capacity,
        n_vehicles=n_vehicles,
    )
    logger.info(
        "Parsed %s: %s customers, capacity=%.1f, n_vehicles=%s",
        name,
        len(customers),
        capacity,
        n_vehicles,
    )
    return instance
