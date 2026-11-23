"""Track segments for the SimPy engine.

`entry` is a capacity-1 SimPy Resource: a departing train holds it for the safe headway,
so the next train cannot enter before `ST_pd` has elapsed (spec 5.1 headway constraint).
Multi-train moving-block logic is added in Week 21."""
from __future__ import annotations
from dataclasses import dataclass, field
import simpy
from .adapters import segment_geometry
from .schemas import Segment


@dataclass
class TrackSegment:
    id: str
    length_m: float
    v_limit_mps: float
    headway_s: float
    entry: simpy.Resource = field(repr=False, default=None)

    @classmethod
    def from_schema(cls, env: simpy.Environment, seg: Segment) -> "TrackSegment":
        length_m, v = segment_geometry(seg)
        return cls(seg.id, length_m, v, seg.safe_headway_hrs * 3600.0, simpy.Resource(env, capacity=1))

    def hold_entry(self, env: simpy.Environment, req, duration_s: Optional[float] = None):
        """Keep the entry lock for the safe headway after a departure, then release."""
        hold = duration_s if duration_s is not None else self.headway_s
        yield env.timeout(hold)
        self.entry.release(req)
