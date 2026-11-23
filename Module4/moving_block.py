"""Moving-block following controller + V2V assessment.

A follower obeys the CONSERVATIVE rule from kpi.py: it must always be able to stop behind the
leader's rear, assuming the leader may stop at its full EMERGENCY deceleration, using only its own
SERVICE deceleration after comms latency + brake delay. If that needs more than the follower can
physically deliver, a hazard event is raised.  Otherwise the follower simply slows / waits, and the
extra delay it suffers is reported as `intervention_delay_s`.

Departure times are taken from the SimPy run; this pass shapes in-line motion only (single segment,
one direction). Trains with an imposed breakdown follow their given trajectory (not controlled).
Opposing-direction conflicts are detected (kpi.evaluate_safety), not controlled.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, List
import numpy as np
from .cosim import SimRun, TrainRun
from .kpi import SafetyConfig, SafetyEvent, SafetyReport, _decel, evaluate_safety
from .physics import Phase, SpeedProfile, accel_mps2
from .schemas import Plan

HRS = 3600.0


def _controlled_trajectory(rb: TrainRun, ra: TrainRun, loco_b, loco_a, cfg: SafetyConfig, dt: float,
                           max_hours: float):
    L, v_lim = rb.profile.length_m, rb.v_limit_mps
    dep_a, dep_b, arr_a = ra.depart_hrs * HRS, rb.depart_hrs * HRS, ra.arrive_hrs * HRS
    n_lead = int(max(arr_a - dep_b, 0.0) / dt) + 2
    tg = dep_b + np.arange(n_lead) * dt
    xl, vl = ra.profile.positions(tg - dep_a), ra.profile.speeds(tg - dep_a)
    on_line = tg < arr_a

    b, b_em = _decel(loco_b, cfg.follower_decel), loco_b.emergency_decel_mps2
    b_l = _decel(loco_a, cfg.leader_decel)
    tau_r = cfg.comm_latency_s + loco_b.brake_delay_s
    mass = rb.mass_t * 1000.0

    ts, xs, vs = [0.0], [0.0], [0.0]
    x = v = 0.0
    events: List[SafetyEvent] = []
    in_margin = in_coll = False
    k = 0
    while L - x > 0.5 and k * dt < max_hours * HRS:
        t_abs = dep_b + k * dt
        lead = k < n_lead and on_line[k]
        v_safe = np.inf
        if lead:
            avail = xl[k] - cfg.train_length_m - x - cfg.standstill_margin_m + vl[k] ** 2 / (2 * b_l)
            v_safe = 0.0 if avail <= 0 else -b * tau_r + np.sqrt((b * tau_r) ** 2 + 2 * b * avail)
        v_target = min(v_lim, np.sqrt(2 * b * max(L - x, 0.0)))
        a = accel_mps2(v, mass, loco_b)
        v_new = min(v + max(a, 0.0) * dt, v_target, v_safe)
        v_new = max(v_new, v - b_em * dt, 0.0)
        if lead:
            gap_new = xl[k] - cfg.train_length_m - (x + 0.5 * (v + v_new) * dt)
            short = v_new > v_safe + 1e-6          # cannot shed speed fast enough
            if gap_new < 0 and not in_coll:
                # follower has not moved: it was released while the leader's body still covers the
                # departure point -> a clearance violation, not a rear-end impact
                kind = "DEPARTURE_CLEARANCE" if x < 1.0 else "REAR_END_COLLISION"
                events.append(SafetyEvent(kind, rb.segment_id, (ra.train_id, rb.train_id),
                                          t_abs / HRS, x / 1000, float(gap_new), 0.0))
            elif short and not in_margin and gap_new >= 0:
                events.append(SafetyEvent("BRAKING_MARGIN", rb.segment_id, (ra.train_id, rb.train_id),
                                          t_abs / HRS, x / 1000, float(gap_new), float(cfg.standstill_margin_m)))
            in_coll, in_margin = gap_new < 0, short
        x += 0.5 * (v + v_new) * dt
        v = v_new
        k += 1
        ts.append(k * dt); xs.append(min(x, L)); vs.append(v)
    t_fin = k * dt + (L - x) / max(v, 0.1)
    ts.append(t_fin); xs.append(L); vs.append(0.0)
    return np.array(ts), np.array(xs), np.array(vs), events


@dataclass
class MBResult:
    run: SimRun                          # same trains, followers' trajectories shaped by moving-block
    events: List[SafetyEvent]
    intervention_delay_s: Dict[str, float]   # extra delay vs the free (unreactive) profile


def apply_moving_block(run: SimRun, plan: Plan, cfg: SafetyConfig = SafetyConfig(), dt: float = 0.5,
                       max_hours: float = 48.0) -> MBResult:
    tmap = {t.id: t for t in plan.trains}
    new = dict(run.trains)
    events: List[SafetyEvent] = []
    delay: Dict[str, float] = {tid: 0.0 for tid in run.trains}
    by_seg: Dict[str, List[str]] = {}
    for tid in run.trains:
        by_seg.setdefault(tmap[tid].segment_id, []).append(tid)
    for ids in by_seg.values():
        ids.sort(key=lambda i: run.trains[i].depart_hrs)
        for i, b in enumerate(ids):
            tb = tmap[b]
            lead = next((a for a in reversed(ids[:i])
                         if not (tmap[a].origin == tb.destination and tmap[a].destination == tb.origin)), None)
            rb = new[b]
            if lead is None or any(p.kind == "stall" for p in rb.profile.phases):
                continue
            ts, xs, vs, ev = _controlled_trajectory(rb, new[lead], run.locos[b], run.locos[lead], cfg, dt, max_hours)
            events += ev
            prof = SpeedProfile(rb.profile.length_m, [Phase("track", 0.0, float(ts[-1]), 0.0, rb.profile.length_m,
                                                           0.0, 0.0, ts, xs, vs)], float(vs.max()))
            arrive = rb.depart_hrs + ts[-1] / HRS
            delay[b] = (arrive - rb.arrive_hrs) * HRS
            new[b] = replace(rb, arrive_hrs=arrive, profile=prof, peak_speed_kmh=float(vs.max()) * 3.6)
    events.sort(key=lambda e: e.t_hrs)
    return MBResult(replace(run, trains=new), events, delay)


@dataclass
class Assessment:
    verdict: str                     # ACCEPT | DEGRADED | REJECT
    events: List[SafetyEvent]        # hazards: head-on, headway, insufficient braking, collisions
    mitigated: SimRun
    intervention_delay_s: Dict[str, float]

    @property
    def violation_count(self) -> int:
        return len(self.events)

    @property
    def hazard_location(self):
        return min(self.events, key=lambda e: e.t_hrs).location if self.events else None

    @property
    def max_intervention_delay_s(self) -> float:
        return max(self.intervention_delay_s.values(), default=0.0)


def assess(run: SimRun, plan: Plan, cfg: SafetyConfig = SafetyConfig(),
           delay_tolerance_s: float = 30.0) -> Assessment:
    """REJECT  = a hazard exists (head-on, headway breach, braking impossible, collision);
       DEGRADED = safe, but followers must be held back > tolerance, so the timetable is not achievable;
       ACCEPT  = safe and followers are not slowed (within tolerance)."""
    mb = apply_moving_block(run, plan, cfg)
    other = [e for e in evaluate_safety(run, plan, cfg).events if e.kind.startswith("HEAD_ON") or e.kind == "HEADWAY"]
    events = sorted(mb.events + other, key=lambda e: e.t_hrs)
    worst = max(mb.intervention_delay_s.values(), default=0.0)
    verdict = "REJECT" if events else ("DEGRADED" if worst > delay_tolerance_s else "ACCEPT")
    return Assessment(verdict, events, mb.run, mb.intervention_delay_s)
