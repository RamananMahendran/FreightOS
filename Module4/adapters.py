"""Adapters: Module 3 output -> Module 4 Plan."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .schemas import Plan, Segment

DEFAULT_SPEED_LIMIT_KMH = 80.0


def segment_geometry(seg: Segment) -> Tuple[float, float]:
    """Returns (length_m, v_limit_mps) for a segment.
    Uses traversing_time_hrs * speed_limit to derive physical track length.
    """
    v_mps = DEFAULT_SPEED_LIMIT_KMH / 3.6
    length_m = v_mps * (seg.traversing_time_hrs * 3600.0)
    return length_m, v_mps


def plan_from_solver_result(
    solver_result: Dict,
    trains: List[Dict],
    cargos: List[Dict],
    segments: List[Dict],
    convoys: Optional[List[Dict]] = None,
) -> Plan:
    """Build a Plan from `FreightHeuristicSolverImproved.solve()` output.

    `solver_result` is the dict returned by solve(): it holds `best_solution`
    (train_sequence / allocation / remaining_capacity) and `train_arrivals`.
    `convoys` is the list from `VirtualCouplingPlanner.plan_convoys(...)`.
    """
    sol = solver_result["best_solution"]
    return Plan(
        trains=trains,
        cargos=cargos,
        segments=segments,
        train_sequence=list(sol["train_sequence"]),
        allocation=dict(sol["allocation"]),
        remaining_capacity=dict(sol["remaining_capacity"]),
        claimed_arrivals=dict(solver_result.get("train_arrivals") or {}) or None,
        convoys=convoys or [],
    )


def save_plan(plan: Plan, path: str | Path) -> None:
    Path(path).write_text(plan.model_dump_json(indent=2))


def load_plan(path: str | Path) -> Plan:
    return Plan.model_validate(json.loads(Path(path).read_text()))
