"""Module 4 schemas. Mirrors Module 3 dict shapes (spec 6.1) and the Gatekeeper contract (spec 6.7)."""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Module 3 dicts carry extra keys (e.g. _fitness)


# ---------- Module 3 inputs ----------
class Train(_Base):
    id: str
    origin: str
    destination: str
    segment_id: str = "CORRIDOR_MAIN"
    ready_time_hrs: float
    max_capacity_tons: float
    priority: float = 0.5
    yard_arrival_hrs: Optional[float] = None


class Cargo(_Base):
    id: str
    origin: str
    destination: str
    weight: float = Field(ge=0)
    priority: float = 0.5
    due_date_offset_hrs: float
    commodity: Optional[str] = None


class Segment(_Base):
    id: str
    traversing_time_hrs: float = Field(gt=0)
    safe_headway_hrs: float = Field(ge=0)


class Convoy(_Base):
    convoy_id: str
    leader_train: str
    follower_trains: List[str] = []
    uncoupling_maneuver_required: bool = False


class Plan(_Base):
    """A complete Module 3 dispatch plan, ready for validation."""
    trains: List[Train]
    cargos: List[Cargo]
    segments: List[Segment]
    train_sequence: List[str]
    allocation: Dict[str, Optional[str]]              # cargo_id -> train_id | None
    remaining_capacity: Optional[Dict[str, float]] = None
    claimed_arrivals: Optional[Dict[str, float]] = None  # Module 3 `train_arrivals`
    convoys: List[Convoy] = []


# ---------- Tier 1 output ----------
class Severity(str, Enum):
    ERROR = "ERROR"      # plan is rejected
    WARNING = "WARNING"  # plan passes but needs attention / Tier 2


class Violation(BaseModel):
    code: str
    severity: Severity
    message: str
    subject: Optional[str] = None     # train/cargo/segment id
    location: Optional[str] = None    # segment id -> becomes `hazard_location`
    details: Dict[str, float | str | int | bool | None] = {}


class Tier1Report(BaseModel):
    passed: bool
    violations: List[Violation]
    recomputed_arrivals: Dict[str, float]
    train_load_tons: Dict[str, float]
    train_utilization: Dict[str, float]
    cargo_tardiness_hrs: Dict[str, float]
    weighted_tardiness: float
    needs_tier2: List[str]            # convoy ids etc. that Tier 1 cannot decide
    elapsed_ms: float

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]


# ---------- Gatekeeper contract (spec 6.7) ----------
class SimRequest(BaseModel):
    train_agent: str
    route_override: str
    wagons: List[str]
    max_sim_seconds: int = 20


class SimResult(BaseModel):
    headway_safety_violations: int
    hazard_location: Optional[str] = None
    delay_reduction_percentage: float
    estimated_arrival_time: float
