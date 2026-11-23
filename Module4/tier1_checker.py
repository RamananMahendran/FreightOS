"""Tier 1: millisecond analytic triage of a Module 3 plan against spec Section 5.

Independent re-implementation: does NOT import Module 3. It recomputes the schedule
with the same semantics as Module 3's evaluate_schedule (one segment queue, static
headway) and then checks the result against the Section 5 constraints.

What Tier 1 can NOT decide (flagged in `needs_tier2`): whether a V2V 3-minute convoy
headway is physically safe (needs braking physics) and anything involving yard dwell
detail (Week 20).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple
from .params import YardParams
from .schemas import Plan, Severity, Tier1Report, Violation

E, W = Severity.ERROR, Severity.WARNING


@dataclass(frozen=True)
class Tier1Config:
    time_tol_hrs: float = 1e-3                 # ~3.6 s
    enforce_min_utilization: bool = False      # Module 3 treats mu_t as a soft penalty
    min_utilization_fraction: float = 0.60     # mu_t (spec 5.2)
    convoy_departure_tolerance_hrs: float = 0.5
    # defaults copied from Module 3 so both modules agree when a segment is missing
    default_traversing_time_hrs: float = 2.2
    default_headway_hrs: float = 0.25
    yard: YardParams = YardParams()            # dwell minimums (spec 5.1); assumed rates


def _opposite(a, b) -> bool:
    return a.origin == b.destination and a.destination == b.origin


def check_plan(plan: Plan, cfg: Tier1Config = Tier1Config()) -> Tier1Report:
    t0 = perf_counter()
    tol = cfg.time_tol_hrs
    trains = {t.id: t for t in plan.trains}
    cargos = {c.id: c for c in plan.cargos}
    segs = {s.id: s for s in plan.segments}
    v: List[Violation] = []

    def seg_params(seg_id: str) -> Tuple[float, float]:
        s = segs.get(seg_id)
        return (s.traversing_time_hrs, s.safe_headway_hrs) if s else (
            cfg.default_traversing_time_hrs, cfg.default_headway_hrs)

    for sid in {t.segment_id for t in plan.trains} - set(segs):
        v.append(Violation(code="SEGMENT_DEFAULTED", severity=W, subject=sid, location=sid,
                           message=f"Segment {sid} not defined; using Module 3 defaults."))

    # --- 1. sequence must be a permutation of the train ids ---
    seq = plan.train_sequence
    seq_ok = len(seq) == len(set(seq)) and set(seq) == set(trains)
    if not seq_ok:
        dup = sorted({x for x in seq if seq.count(x) > 1})
        missing = sorted(set(trains) - set(seq))
        unknown = sorted(set(seq) - set(trains))
        v.append(Violation(code="SEQUENCE_NOT_PERMUTATION", severity=E,
                           message=f"duplicates={dup} missing={missing} unknown={unknown}"))

    # --- 2. allocation: ids, route compatibility, single allocation ---
    load: Dict[str, float] = {tid: 0.0 for tid in trains}
    for c_id, t_id in plan.allocation.items():
        if c_id not in cargos:
            v.append(Violation(code="ALLOC_UNKNOWN_CARGO", severity=E, subject=c_id,
                               message=f"Allocation references unknown cargo {c_id}."))
            continue
        if t_id is None:
            continue
        if t_id not in trains:
            v.append(Violation(code="ALLOC_UNKNOWN_TRAIN", severity=E, subject=c_id,
                               message=f"Cargo {c_id} allocated to unknown train {t_id}."))
            continue
        c, t = cargos[c_id], trains[t_id]
        if (c.origin, c.destination) != (t.origin, t.destination):
            v.append(Violation(code="ROUTE_MISMATCH", severity=E, subject=c_id, location=t.segment_id,
                               message=f"Cargo {c_id} ({c.origin}->{c.destination}) on train {t_id} "
                                       f"({t.origin}->{t.destination})."))
        load[t_id] += c.weight
    unallocated = [c for c in cargos if plan.allocation.get(c) is None]

    # --- 3. capacity bounds  mu*lambda <= load <= lambda (spec 5.2) ---
    util: Dict[str, float] = {}
    for tid, t in trains.items():
        cap = t.max_capacity_tons
        util[tid] = load[tid] / cap if cap > 0 else 0.0
        if load[tid] > cap + 1e-9:
            v.append(Violation(code="CAPACITY_EXCEEDED", severity=E, subject=tid, location=t.segment_id,
                               message=f"Train {tid} load {load[tid]:.1f} t > capacity {cap:.1f} t.",
                               details={"load": load[tid], "capacity": cap}))
        if 0 < load[tid] and util[tid] < cfg.min_utilization_fraction - 1e-9:
            v.append(Violation(code="UTILIZATION_BELOW_FLOOR",
                               severity=E if cfg.enforce_min_utilization else W, subject=tid,
                               message=f"Train {tid} utilisation {util[tid]:.0%} < floor "
                                       f"{cfg.min_utilization_fraction:.0%}.",
                               details={"utilization": util[tid]}))
        if plan.remaining_capacity is not None and tid in plan.remaining_capacity:
            expected = cap - load[tid]
            if abs(plan.remaining_capacity[tid] - expected) > 1e-6:
                v.append(Violation(code="REMAINING_CAPACITY_MISMATCH", severity=E, subject=tid,
                                   message=f"Train {tid} remaining_capacity "
                                           f"{plan.remaining_capacity[tid]:.1f} != recomputed {expected:.1f}."))

    # --- 4/5. schedule (only meaningful when the sequence is valid) ---
    recomputed: Dict[str, float] = {}
    tardiness: Dict[str, float] = {}
    if seq_ok:
        occ: Dict[str, float] = {}
        dep_re: Dict[str, float] = {}
        for tid in seq:                       # same semantics as Module 3 evaluate_schedule
            t = trains[tid]
            trav, hw = seg_params(t.segment_id)
            dep = max(t.ready_time_hrs, occ.get(t.segment_id, 0.0))
            dep_re[tid] = dep
            recomputed[tid] = dep + trav
            occ[t.segment_id] = dep + hw

        if plan.claimed_arrivals:             # solver's claim must match the recomputation
            for tid in trains:
                if tid not in plan.claimed_arrivals:
                    v.append(Violation(code="ARRIVAL_MISSING", severity=E, subject=tid,
                                       message=f"No claimed arrival for train {tid}."))
                elif abs(plan.claimed_arrivals[tid] - recomputed[tid]) > tol:
                    v.append(Violation(code="ARRIVAL_MISMATCH", severity=E, subject=tid,
                                       location=trains[tid].segment_id,
                                       message=f"Train {tid} claimed arrival {plan.claimed_arrivals[tid]:.3f} h "
                                               f"vs recomputed {recomputed[tid]:.3f} h.",
                                       details={"claimed": plan.claimed_arrivals[tid],
                                                "recomputed": recomputed[tid]}))
        # physical checks use the claim if one was made (so a corrupted claim is caught)
        arr = {tid: (plan.claimed_arrivals or {}).get(tid, recomputed[tid]) for tid in trains}
        dep = {tid: arr[tid] - seg_params(trains[tid].segment_id)[0] for tid in trains}

        for tid, t in trains.items():         # departure >= ready time
            if dep[tid] < t.ready_time_hrs - tol:
                v.append(Violation(code="DEPARTS_BEFORE_READY", severity=E, subject=tid,
                                   location=t.segment_id,
                                   message=f"Train {tid} departs {dep[tid]:.3f} h before ready "
                                           f"{t.ready_time_hrs:.3f} h."))

        for tid, t in trains.items():         # dwell: d - a >= Ul + Lo + dw  (necessary condition)
            if t.yard_arrival_hrs is None:
                continue
            need = cfg.yard.min_dwell_hrs(load[tid])
            have = dep[tid] - t.yard_arrival_hrs
            if have < need - tol:
                v.append(Violation(code="DWELL_TOO_SHORT", severity=E, subject=tid, location=t.segment_id,
                                   message=f"Train {tid} dwells {have:.2f} h in yard; needs >= {need:.2f} h "
                                           f"(loading {load[tid]:.0f} t + crew).",
                                   details={"dwell_hrs": have, "required_hrs": need}))

        by_seg: Dict[str, List[str]] = {}
        for tid, t in trains.items():
            by_seg.setdefault(t.segment_id, []).append(tid)

        for seg_id, ids in by_seg.items():
            ids.sort(key=lambda i: dep[i])
            _, hw = seg_params(seg_id)
            # safe headway (same direction): compare with nearest earlier non-opposing train
            for k, j in enumerate(ids):
                for i in reversed(ids[:k]):
                    if _opposite(trains[i], trains[j]):
                        continue
                    gap = dep[j] - dep[i]
                    if gap < hw - tol:
                        v.append(Violation(
                            code="HEADWAY_VIOLATION", severity=E, subject=f"{i},{j}", location=seg_id,
                            message=f"Trains {i} and {j} depart {gap*60:.1f} min apart on {seg_id}; "
                                    f"minimum {hw*60:.1f} min.",
                            details={"gap_hrs": gap, "required_hrs": hw}))
                    break
            # single-line opposing-direction conflict: occupancy windows [dep, arr] must not overlap
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b]
                    if _opposite(trains[i], trains[j]) and dep[j] < arr[i] - tol:
                        v.append(Violation(
                            code="OPPOSING_CONFLICT", severity=E, subject=f"{i},{j}", location=seg_id,
                            message=f"Opposing trains {i} and {j} both occupy single line {seg_id} "
                                    f"between {dep[j]:.2f} h and {arr[i]:.2f} h.",
                            details={"overlap_hrs": arr[i] - dep[j]}))

        # tardiness (informational; spec 5.2)
        for c_id, t_id in plan.allocation.items():
            if t_id in trains and c_id in cargos:
                tardiness[c_id] = max(0.0, arr[t_id] - cargos[c_id].due_date_offset_hrs)

    # --- 6. convoys: structural checks; physical safety deferred to Tier 2 ---
    needs_tier2: List[str] = []
    for cv in plan.convoys:
        members = [cv.leader_train, *cv.follower_trains]
        bad = [m for m in members if m not in trains]
        if bad:
            v.append(Violation(code="CONVOY_UNKNOWN_TRAIN", severity=E, subject=cv.convoy_id,
                               message=f"Convoy {cv.convoy_id} references unknown trains {bad}."))
            continue
        lead = trains[cv.leader_train]
        for f in cv.follower_trains:
            ft = trains[f]
            if ft.origin != lead.origin:
                v.append(Violation(code="CONVOY_ORIGIN_MISMATCH", severity=E, subject=cv.convoy_id,
                                   message=f"Convoy {cv.convoy_id}: {f} origin {ft.origin} != leader {lead.origin}."))
            spread = abs(ft.ready_time_hrs - lead.ready_time_hrs)
            if spread > cfg.convoy_departure_tolerance_hrs + tol:
                v.append(Violation(code="CONVOY_DEPARTURE_SPREAD", severity=E, subject=cv.convoy_id,
                                   message=f"Convoy {cv.convoy_id}: ready times differ by {spread*60:.0f} min "
                                           f"(limit {cfg.convoy_departure_tolerance_hrs*60:.0f})."))
        v.append(Violation(code="CONVOY_NEEDS_PHYSICS", severity=W, subject=cv.convoy_id,
                           location=lead.segment_id,
                           message=f"Convoy {cv.convoy_id}: reduced V2V headway needs braking-distance "
                                   f"verification in Tier 2."))
        needs_tier2.append(cv.convoy_id)

    w_tard = sum(cargos[c].priority * td for c, td in tardiness.items())
    return Tier1Report(
        passed=not any(x.severity == E for x in v),
        violations=v,
        recomputed_arrivals=recomputed,
        train_load_tons=load,
        train_utilization=util,
        cargo_tardiness_hrs=tardiness,
        weighted_tardiness=w_tard,
        needs_tier2=needs_tier2,
        elapsed_ms=(perf_counter() - t0) * 1000,
    )


if __name__ == "__main__":  # python -m Module4.tier1_checker plan.json
    from .adapters import load_plan
    rep = check_plan(load_plan(sys.argv[1]))
    print(f"passed={rep.passed}  errors={len(rep.errors)}  warnings={len(rep.warnings)}  {rep.elapsed_ms:.2f} ms")
    for x in rep.violations:
        print(f"  [{x.severity.value}] {x.code}: {x.message}")
    sys.exit(0 if rep.passed else 1)
