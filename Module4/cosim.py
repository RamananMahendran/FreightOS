"""Orchestrator (Week 19 scope): run a Plan through SimPy + Mesa and report departure/arrival.
Trains depart in Module 3's `train_sequence` order, each after max(ready, predecessor+headway)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union
import mesa
import simpy
from .params import DEFAULT_LOCO, IDEAL, LOCOS, LocoParams, TARE_FRACTION_OF_CAPACITY, YardParams
from .yard import Yard
from .physics import SpeedProfile, build_profile, with_breakdown
from .schemas import Plan, Segment
from .scenarios import Scenario
from .track import TrackSegment
from .train_agent import TrainAgent

HRS = 3600.0


@dataclass
class TrainRun:
    train_id: str
    ready_hrs: float            # actual (after yard handling when a yard is simulated)
    depart_hrs: float
    arrive_hrs: float
    mass_t: float
    peak_speed_kmh: float
    profile: SpeedProfile
    scheduled_ready_hrs: float = 0.0     # Module 3's ready_time
    yard_arrival_hrs: Optional[float] = None
    load_start_hrs: Optional[float] = None
    load_end_hrs: Optional[float] = None
    load_tons: float = 0.0
    utilization: float = 0.0
    segment_id: str = "CORRIDOR_MAIN"
    v_limit_mps: float = 80.0 / 3.6

    @property
    def dwell_hrs(self) -> Optional[float]:
        return None if self.yard_arrival_hrs is None else self.ready_hrs - self.yard_arrival_hrs

    @property
    def yard_wait_hrs(self) -> Optional[float]:    # time queued for a slot
        return None if self.load_start_hrs is None else self.load_start_hrs - self.yard_arrival_hrs

    @property
    def handling_delay_hrs(self) -> float:         # departure later than Module 3 assumed
        return self.depart_hrs - self.scheduled_ready_hrs


@dataclass
class SimRun:
    trains: Dict[str, TrainRun]
    seed: int
    yard_busy_hrs: Dict[str, float] = None
    locos: Dict[str, LocoParams] = field(default_factory=dict)

    @property
    def arrivals_hrs(self) -> Dict[str, float]:
        return {k: r.arrive_hrs for k, r in self.trains.items()}


def simulate_plan(
    plan: Plan,
    loco: Union[str, LocoParams, Dict[str, Union[str, LocoParams]]] = DEFAULT_LOCO,
    seed: int = 0,
    dt: float = 0.5,
    yard: Optional[YardParams] = None,
    v2v_headway_hrs: Optional[float] = None,
    scenario: Optional[Scenario] = None,
) -> SimRun:
    """Simulates a dispatch plan through SimPy + Mesa.
    Supports locomotive physics, yard workflows, V2V dynamic headway compression,
    and scenario perturbations (delays, breakdowns, closures).
    """
    env = simpy.Environment()
    model = mesa.Model(rng=seed)
    segs_in = {s.id: s for s in plan.segments}
    tracks: Dict[str, TrackSegment] = {}
    cargo_w = {c.id: c.weight for c in plan.cargos}
    load = {t.id: 0.0 for t in plan.trains}
    for c_id, t_id in plan.allocation.items():
        if t_id in load and c_id in cargo_w:
            load[t_id] += cargo_w[c_id]
    trains = {t.id: t for t in plan.trains}

    # Scenario perturbations
    delays = {d.train_id: d.hours for d in scenario.delays} if scenario else {}
    breakdowns = {b.train_id: b for b in scenario.breakdowns} if scenario else {}
    closures_by_seg = {}
    if scenario and scenario.closures:
        for cl in scenario.closures:
            closures_by_seg.setdefault(cl.segment_id, []).append(cl)

    # Convoy mapping for V2V headway compression
    convoy_of = {
        m: cv.convoy_id
        for cv in plan.convoys
        for m in [cv.leader_train, *cv.follower_trains]
    }

    locos_resolved: Dict[str, LocoParams] = {}
    agents: Dict[str, TrainAgent] = {}
    yards: Dict[str, Yard] = {}

    seq = plan.train_sequence
    for idx, tid in enumerate(seq):
        t = trains[tid]
        # Resolve locomotive
        if isinstance(loco, dict):
            loco_spec = loco.get(tid, DEFAULT_LOCO)
        else:
            loco_spec = loco
        loco_p = loco_spec if isinstance(loco_spec, LocoParams) else (
            IDEAL if loco_spec == "IDEAL" else LOCOS[loco_spec]
        )
        locos_resolved[tid] = loco_p

        # Track segment
        if t.segment_id not in tracks:
            seg = segs_in.get(t.segment_id) or Segment(
                id=t.segment_id, traversing_time_hrs=2.2, safe_headway_hrs=0.25
            )
            tracks[t.segment_id] = TrackSegment.from_schema(env, seg)
        trk = tracks[t.segment_id]

        mass_kg = (load[tid] + TARE_FRACTION_OF_CAPACITY * t.max_capacity_tons) * 1000.0
        prof = build_profile(trk.length_m, trk.v_limit_mps, mass_kg, loco_p, dt=dt)

        # Apply breakdown if scheduled
        if tid in breakdowns:
            bk = breakdowns[tid]
            t_fail = prof.time_at_position(bk.at_km * 1000.0)
            prof = with_breakdown(
                prof, t_fail, bk.duration_hrs * HRS, mass_kg, loco_p, trk.v_limit_mps, dt=dt
            )

        y = None
        if yard is not None:
            y = yards.setdefault(t.origin, Yard(env, t.origin, yard))

        ready_h = t.ready_time_hrs + delays.get(tid, 0.0)
        arr_h = (t.yard_arrival_hrs if t.yard_arrival_hrs is not None else t.ready_time_hrs) + delays.get(tid, 0.0)

        # Dynamic V2V headway hold: if next train in sequence is in the same convoy
        next_tid = seq[idx + 1] if idx + 1 < len(seq) else None
        same_convoy = (
            v2v_headway_hrs is not None
            and next_tid is not None
            and tid in convoy_of
            and convoy_of[tid] == convoy_of.get(next_tid)
        )
        hw_hold_s = (v2v_headway_hrs * HRS) if same_convoy else trk.headway_s

        seg_closures = closures_by_seg.get(t.segment_id, [])

        agents[tid] = TrainAgent(
            model,
            tid,
            ready_h * HRS,
            mass_kg,
            trk,
            prof,
            priority=t.priority,
            load_tons=load[tid],
            yard=y,
            yard_arrival_s=arr_h * HRS,
            headway_hold_s=hw_hold_s,
            closures=seg_closures,
        )

    last_departed: Dict[str, simpy.Event] = {}
    for tid in plan.train_sequence:
        a = agents[tid]
        ev = env.event()
        env.process(a.run(env, last_departed.get(a.segment.id), ev))
        last_departed[a.segment.id] = ev
    env.run()

    def _h(x): return None if x is None else x / HRS
    runs = {
        tid: TrainRun(
            tid,
            a.ready_s / HRS,
            a.depart_s / HRS,
            a.arrive_s / HRS,
            a.mass_kg / 1000.0,
            a.profile.peak_speed_mps * 3.6,
            a.profile,
            scheduled_ready_hrs=a.scheduled_ready_s / HRS,
            yard_arrival_hrs=_h(a.yard_arrive_s),
            load_start_hrs=_h(a.load_start_s),
            load_end_hrs=_h(a.load_end_s),
            load_tons=a.load_tons,
            utilization=a.load_tons / trains[tid].max_capacity_tons,
            segment_id=trains[tid].segment_id,
            v_limit_mps=tracks[trains[tid].segment_id].v_limit_mps,
        )
        for tid, a in agents.items()
    }
    return SimRun(runs, seed, {k: y.busy_s / HRS for k, y in yards.items()}, locos_resolved)

