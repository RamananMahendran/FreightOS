"""Safety monitor and KPIs for a SimRun.

evaluate_safety() is an UNMITIGATED check: it assumes followers do not react (they follow their free
profile). Use moving_block.assess() for the realistic verdict where followers brake and wait.

CONSERVATIVE V2V / moving-block rule (user decision, Week 21):
  * the LEADER may stop at its full EMERGENCY deceleration (worst case for the follower);
  * the FOLLOWER may only rely on its guaranteed SERVICE deceleration, after comms latency
    plus its own brake build-up delay;
  * required gap (rear of leader to front of follower) at every instant:
        s_req = v_f (t_comm + t_delay_f) + v_f^2 / (2 b_f)  -  v_l^2 / (2 b_l)  +  margin
  Trains that have ARRIVED are assumed to clear immediately into the terminal yard.
All numeric defaults below are ASSUMPTIONS.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .cosim import SimRun
from .params import LocoParams
from .schemas import Plan

HRS = 3600.0


@dataclass(frozen=True)
class SafetyConfig:
    leader_decel: str = "emergency"        # "emergency" (conservative, default) | "service"
    follower_decel: str = "service"        # guaranteed rate the follower may rely on
    comm_latency_s: float = 1.0            # V2V message latency                          [ASSUMED]
    standstill_margin_m: float = 50.0      # buffer left when stopped                     [ASSUMED]
    train_length_m: float = 700.0          # rake length (~58 wagons)                     [ASSUMED]
    static_headway_tol_hrs: float = 1e-3
    v2v_headway_hrs: float = 0.05          # headway allowed for convoy members (3 min)
    default_headway_hrs: float = 0.25
    dt_s: float = 1.0                      # monitor time step


@dataclass
class SafetyEvent:
    kind: str                  # BRAKING_MARGIN | REAR_END_COLLISION | HEAD_ON_MARGIN | HEAD_ON_COLLISION | HEADWAY
    segment_id: str
    trains: Tuple[str, ...]
    t_hrs: float
    chainage_km: float
    gap_m: float
    required_m: float
    convoy: bool = False

    @property
    def location(self) -> str:
        return f"{self.segment_id}@km{self.chainage_km:.1f}"


@dataclass
class SafetyReport:
    events: List[SafetyEvent]

    @property
    def violation_count(self) -> int:
        return len(self.events)

    @property
    def safe(self) -> bool:
        return not self.events

    @property
    def hazard_location(self) -> Optional[str]:
        return min(self.events, key=lambda e: e.t_hrs).location if self.events else None


def _decel(loco: LocoParams, mode: str) -> float:
    return loco.emergency_decel_mps2 if mode == "emergency" else loco.service_decel_mps2


def required_gap_m(v_f, v_l, loco_f: LocoParams, loco_l: LocoParams, cfg: SafetyConfig):
    b_f, b_l = _decel(loco_f, cfg.follower_decel), _decel(loco_l, cfg.leader_decel)
    return (v_f * (cfg.comm_latency_s + loco_f.brake_delay_s) + v_f ** 2 / (2 * b_f)
            - v_l ** 2 / (2 * b_l) + cfg.standstill_margin_m)


def steady_state_min_headway_s(v_mps: float, loco_f: LocoParams, loco_l: LocoParams,
                               cfg: SafetyConfig = SafetyConfig()) -> float:
    """Smallest time headway that is safe for two identical-speed trains cruising at v."""
    need = required_gap_m(v_mps, v_mps, loco_f, loco_l, cfg) + cfg.train_length_m
    return need / v_mps


def _clusters(bad: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(bad)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts, ends))


def evaluate_safety(run: SimRun, plan: Plan, cfg: SafetyConfig = SafetyConfig()) -> SafetyReport:
    trains = {t.id: t for t in plan.trains}
    seg_hw = {s.id: s.safe_headway_hrs for s in plan.segments}
    convoy_of = {m: cv.convoy_id for cv in plan.convoys for m in [cv.leader_train, *cv.follower_trains]}
    events: List[SafetyEvent] = []

    by_seg: Dict[str, List[str]] = {}
    for tid in run.trains:
        by_seg.setdefault(trains[tid].segment_id, []).append(tid)

    for seg_id, ids in by_seg.items():
        ids.sort(key=lambda i: run.trains[i].depart_hrs)
        L_seg = run.trains[ids[0]].profile.length_m
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1:]:
                ra, rb = run.trains[a], run.trains[b]
                ta, tb = trains[a], trains[b]
                opposite = ta.origin == tb.destination and ta.destination == tb.origin
                dep_a, dep_b = ra.depart_hrs * HRS, rb.depart_hrs * HRS
                arr_a, arr_b = ra.arrive_hrs * HRS, rb.arrive_hrs * HRS
                t_lo, t_hi = max(dep_a, dep_b), min(arr_a, arr_b)
                same_convoy = a in convoy_of and convoy_of[a] == convoy_of.get(b)

                if not opposite:
                    # static / V2V departure headway (spec 5.1)
                    hw = cfg.v2v_headway_hrs if same_convoy else seg_hw.get(seg_id, cfg.default_headway_hrs)
                    gap_h = rb.depart_hrs - ra.depart_hrs
                    if gap_h < hw - cfg.static_headway_tol_hrs:
                        events.append(SafetyEvent("HEADWAY", seg_id, (a, b), rb.depart_hrs, 0.0,
                                                  gap_h * 60, hw * 60, same_convoy))
                if t_hi <= t_lo:
                    continue
                t = np.arange(t_lo, t_hi + 1e-9, cfg.dt_s)
                xa = ra.profile.positions(t - dep_a); va = ra.profile.speeds(t - dep_a)
                xb = rb.profile.positions(t - dep_b); vb = rb.profile.speeds(t - dep_b)
                loco_a, loco_b = run.locos[a], run.locos[b]

                if not opposite:
                    # leader = a (departed earlier). gap = rear of leader - front of follower
                    gap = xa - cfg.train_length_m - xb
                    req = required_gap_m(vb, va, loco_b, loco_a, cfg)
                    for i0, i1 in _clusters(gap < req):
                        seg_gap = gap[i0:i1 + 1]
                        coll = bool((seg_gap < 0).any())
                        k = i0 + (int(np.argmax(seg_gap < 0)) if coll else int(np.argmin((gap - req)[i0:i1 + 1])))
                        events.append(SafetyEvent("REAR_END_COLLISION" if coll else "BRAKING_MARGIN", seg_id,
                                                  (a, b), t[k] / HRS, xb[k] / 1000, float(gap[k]),
                                                  float(req[k]), same_convoy))
                else:
                    # head-on on a single line: fronts meet when x_a + x_b >= L
                    gap = L_seg - xa - xb
                    req = (va * (cfg.comm_latency_s + loco_a.brake_delay_s) + va ** 2 / (2 * loco_a.service_decel_mps2)
                           + vb * (cfg.comm_latency_s + loco_b.brake_delay_s) + vb ** 2 / (2 * loco_b.service_decel_mps2)
                           + cfg.standstill_margin_m)
                    for i0, i1 in _clusters(gap < req):
                        seg_gap = gap[i0:i1 + 1]
                        coll = bool((seg_gap <= 0).any())
                        k = i0 + (int(np.argmax(seg_gap <= 0)) if coll else int(np.argmin((gap - req)[i0:i1 + 1])))
                        events.append(SafetyEvent("HEAD_ON_COLLISION" if coll else "HEAD_ON_MARGIN", seg_id,
                                                  (a, b), t[k] / HRS, xa[k] / 1000, float(gap[k]), float(req[k])))
    events.sort(key=lambda e: e.t_hrs)
    return SafetyReport(events)


# ---------------- delay KPIs ----------------
def weighted_tardiness(run: SimRun, plan: Plan) -> float:
    """sum_j w_j * max(0, arrival - due) over allocated cargo (spec 5.2), hours."""
    cargos = {c.id: c for c in plan.cargos}
    tot = 0.0
    for c_id, t_id in plan.allocation.items():
        if t_id in run.trains and c_id in cargos:
            tot += cargos[c_id].priority * max(0.0, run.trains[t_id].arrive_hrs - cargos[c_id].due_date_offset_hrs)
    return tot


def delay_reduction_pct(baseline: SimRun, proposed: SimRun, plan: Plan, train_id: Optional[str] = None) -> float:
    """% reduction in priority-weighted tardiness (positive = proposed is better). If the baseline
    has no tardiness there is nothing to reduce -> 0.0. `train_id` restricts to that train's cargo."""
    if train_id is not None:
        plan = plan.model_copy(update={"allocation": {c: t for c, t in plan.allocation.items() if t == train_id}})
    b, p = weighted_tardiness(baseline, plan), weighted_tardiness(proposed, plan)
    if b <= 1e-9:
        return 0.0 if p <= 1e-9 else -100.0 * p
    return 100.0 * (b - p) / b
