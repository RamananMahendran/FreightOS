"""What-if perturbations applied by simulate_plan(scenario=...)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Delay:
    """Train reaches its yard / becomes ready `hours` later than planned."""
    train_id: str
    hours: float


@dataclass(frozen=True)
class Breakdown:
    """Train fails at chainage `at_km`: emergency stop, stalls `duration_hrs`, then restarts."""
    train_id: str
    at_km: float
    duration_hrs: float


@dataclass(frozen=True)
class LineClosure:
    """No train may ENTER the segment during [start_hrs, end_hrs). Trains already on the line proceed."""
    segment_id: str
    start_hrs: float
    end_hrs: float


@dataclass(frozen=True)
class Scenario:
    name: str = "scenario"
    delays: Tuple[Delay, ...] = ()
    breakdowns: Tuple[Breakdown, ...] = ()
    closures: Tuple[LineClosure, ...] = ()
