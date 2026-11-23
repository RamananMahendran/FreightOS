import numpy as np, pytest
from hypothesis import given, settings, strategies as st
from Module4.cosim import simulate_plan
from Module4.kpi import (SafetyConfig, required_gap_m, steady_state_min_headway_s, evaluate_safety,
                         delay_reduction_pct, weighted_tardiness)
from Module4.moving_block import assess
from Module4.params import LOCOS
from Module4.physics import build_profile, with_breakdown
from Module4.scenarios import Breakdown, Delay, LineClosure, Scenario
from Module4.schemas import Plan
from Module4.tier1_checker import check_plan
from Module4.conftest import SEGMENTS

WAG9 = LOCOS["WAG_9"]

def convoy_plan(n=2, w=2400, due=99.0, convoy=True):
    tr = [{"id": f"T{i}", "origin": "A", "destination": "B", "ready_time_hrs": 0.0, "max_capacity_tons": 2600}
          for i in range(n)]
    cg = [{"id": f"C{i}", "origin": "A", "destination": "B", "weight": w, "priority": 0.5,
           "due_date_offset_hrs": due} for i in range(n)]
    cv = [{"convoy_id": "V1", "leader_train": "T0", "follower_trains": [f"T{i}" for i in range(1, n)]}] if convoy else []
    return Plan(trains=tr, cargos=cg, segments=SEGMENTS, train_sequence=[t["id"] for t in tr],
                allocation={f"C{i}": f"T{i}" for i in range(n)}, convoys=cv)

def verdict(plan, h_s, loco="WAG_9", **kw):
    r = simulate_plan(plan, loco=loco, v2v_headway_hrs=h_s / 3600, **kw)
    return assess(r, plan, SafetyConfig(v2v_headway_hrs=h_s / 3600))

# ---------------- physics: breakdown profile ----------------
def test_breakdown_profile_conserves_distance_and_stalls():
    L, v = 100_000, 80 / 3.6
    free = build_profile(L, v, 4e6, WAG9)
    p = with_breakdown(free, free.time_at_position(40_000), 1800.0, 4e6, WAG9, v)
    assert [x.kind for x in p.phases].count("stall") == 1
    assert p.total_time_s > free.total_time_s + 1800
    xs = p.positions(np.arange(0, p.total_time_s, 10.0))
    assert np.all(np.diff(xs) >= -1e-6) and p.position_at(p.total_time_s) == pytest.approx(L, abs=1.0)
    assert sum(1 for x in p.phases if x.kind == "stall" and x.v0 == 0) == 1

def test_speed_matches_finite_difference_of_position():
    p = build_profile(60_000, 80 / 3.6, 4e6, WAG9)
    t = np.arange(5.0, p.total_time_s - 5, 7.0)
    fd = (p.positions(t + 0.5) - p.positions(t - 0.5)) / 1.0
    assert np.allclose(p.speeds(t), fd, atol=0.05)

def test_time_at_position_inverts_position():
    p = build_profile(60_000, 80 / 3.6, 4e6, WAG9)
    for x in (1000, 20_000, 59_000):
        assert p.position_at(p.time_at_position(x)) == pytest.approx(x, abs=0.5)

# ---------------- conservative rule ----------------
def test_conservative_leader_rule_needs_more_gap():
    v = 80 / 3.6
    emer = required_gap_m(v, v, WAG9, WAG9, SafetyConfig(leader_decel="emergency"))
    serv = required_gap_m(v, v, WAG9, WAG9, SafetyConfig(leader_decel="service"))
    assert emer > serv
    # hand calculation, spec of the rule
    expect = v * (1.0 + 8.0) + v**2 / (2 * 0.35) - v**2 / (2 * 0.60) + 50.0
    assert emer == pytest.approx(expect)

def test_steady_state_min_headway_is_about_a_minute():
    h = steady_state_min_headway_s(80 / 3.6, WAG9, WAG9)
    assert 50 < h < 65

# ---------------- V2V verdicts ----------------
def test_three_minute_v2v_is_accepted_for_loaded_trains():
    a = verdict(convoy_plan(), 180)
    assert a.verdict == "ACCEPT" and a.violation_count == 0 and a.max_intervention_delay_s < 3

def test_thirty_second_headway_rejected_departure_clearance():
    a = verdict(convoy_plan(), 30)
    assert a.verdict == "REJECT" and {e.kind for e in a.events} == {"DEPARTURE_CLEARANCE"}
    assert a.hazard_location.startswith("CORRIDOR_MAIN@km")

def test_v2v_headway_applied_by_simpy_lock():
    p = convoy_plan()
    on = simulate_plan(p, loco="IDEAL", v2v_headway_hrs=0.05)
    off = simulate_plan(p, loco="IDEAL")
    assert on.trains["T1"].depart_hrs == pytest.approx(0.05, abs=1e-4)
    assert off.trains["T1"].depart_hrs == pytest.approx(0.25, abs=1e-4)

def test_static_fifteen_minute_headway_accepted():
    p = convoy_plan(convoy=False)
    assert assess(simulate_plan(p, loco="WAG_9"), p).verdict == "ACCEPT"

def test_far_leader_does_not_change_follower_motion():
    """Controller equivalence: when the leader is never binding, delay vs free profile ~ 0."""
    a = verdict(convoy_plan(), 900)
    assert abs(a.intervention_delay_s["T1"]) < 3.0

@settings(max_examples=30, deadline=None)
@given(h=st.integers(20, 600), extra=st.integers(0, 300))
def test_property_verdict_monotone_in_headway(h, extra):
    p = convoy_plan()
    if verdict(p, h).verdict == "ACCEPT":
        assert verdict(p, h + extra).verdict == "ACCEPT"

def test_mixed_locos_fast_follower_does_not_catch_slow_leader_at_3min():
    a = verdict(convoy_plan(), 180, loco={"T0": "WDG_4D", "T1": "WAG_12B"})
    assert a.verdict == "ACCEPT"

# ---------------- breakdown ----------------
def test_leader_breakdown_follower_stops_and_waits_without_collision():
    p = convoy_plan()
    sc = Scenario(breakdowns=(Breakdown("T0", 60.0, 1.0),))
    r = simulate_plan(p, loco="WAG_9", v2v_headway_hrs=0.05, scenario=sc)
    a = assess(r, p)
    assert a.violation_count == 0 and a.verdict == "DEGRADED"
    assert 0.8 * 3600 < a.intervention_delay_s["T1"] < 1.05 * 3600           # held ~ the stall duration
    assert a.mitigated.trains["T1"].arrive_hrs > a.mitigated.trains["T0"].arrive_hrs   # never passes

def test_unmitigated_check_shows_follower_would_hit_stalled_leader():
    p = convoy_plan()
    r = simulate_plan(p, loco="WAG_9", v2v_headway_hrs=0.05,
                      scenario=Scenario(breakdowns=(Breakdown("T0", 60.0, 1.0),)))
    assert "REAR_END_COLLISION" in {e.kind for e in evaluate_safety(r, p).events}

# ---------------- head-on (single line) ----------------
def opposing(t1_ready):
    tr = [{"id": "T0", "origin": "A", "destination": "B", "ready_time_hrs": 0.0, "max_capacity_tons": 2600},
          {"id": "T1", "origin": "B", "destination": "A", "ready_time_hrs": t1_ready, "max_capacity_tons": 2600}]
    return Plan(trains=tr, cargos=[], segments=SEGMENTS, train_sequence=["T0", "T1"], allocation={})

def test_head_on_detected_and_reported_at_first_contact():
    p = opposing(0.5)
    a = assess(simulate_plan(p, loco="WAG_9"), p)
    ev = [e for e in a.events if e.kind == "HEAD_ON_COLLISION"]
    assert a.verdict == "REJECT" and ev and abs(ev[0].gap_m) < 50          # first contact, not 100+ km overlap
    assert 0 < ev[0].chainage_km < 180

def test_opposing_trains_in_separate_time_windows_accepted():
    p = opposing(3.0)                      # T0 arrives ~2.0 h; T1 departs 3.0 h
    assert assess(simulate_plan(p, loco="WAG_9"), p).verdict == "ACCEPT"

@pytest.mark.parametrize("t1_ready", [0.0, 0.5, 1.9, 2.5, 3.0, 5.0])
def test_tier1_and_sim_agree_on_opposing_conflicts(t1_ready):
    p = opposing(t1_ready)
    tier1 = "OPPOSING_CONFLICT" in {v.code for v in check_plan(p).violations}
    sim = any(e.kind.startswith("HEAD_ON") for e in assess(simulate_plan(p, loco="IDEAL"), p).events)
    assert tier1 == sim

# ---------------- scenarios & KPIs ----------------
def test_delay_scenario_shifts_train():
    p = convoy_plan(convoy=False)
    base = simulate_plan(p, loco="IDEAL")
    d = simulate_plan(p, loco="IDEAL", scenario=Scenario(delays=(Delay("T1", 1.0),)))
    assert d.trains["T1"].depart_hrs == pytest.approx(base.trains["T1"].depart_hrs + 0.75, abs=1e-3)  # 1.0 vs 0.25
    assert d.trains["T0"].arrive_hrs == pytest.approx(base.trains["T0"].arrive_hrs)

def test_line_closure_holds_departures_until_reopen():
    p = convoy_plan(convoy=False)
    r = simulate_plan(p, loco="IDEAL", scenario=Scenario(closures=(LineClosure("CORRIDOR_MAIN", 0.0, 3.0),)))
    assert r.trains["T0"].depart_hrs == pytest.approx(3.0, abs=1e-4)
    assert r.trains["T1"].depart_hrs == pytest.approx(3.25, abs=1e-4)

def test_delay_reduction_kpi():
    p = convoy_plan(convoy=False, due=2.3)
    base = simulate_plan(p, loco="IDEAL", scenario=Scenario(delays=(Delay("T0", 1.0),)))   # T0 late
    prop = simulate_plan(p, loco="IDEAL")
    assert weighted_tardiness(base, p) > weighted_tardiness(prop, p)
    assert 0 < delay_reduction_pct(base, prop, p) <= 100
    assert delay_reduction_pct(prop, prop, p) == 0.0                         # nothing to reduce
    assert delay_reduction_pct(prop, base, p) < 0                            # worse plan -> negative

def test_assessment_deterministic():
    p = convoy_plan()
    a1, a2 = verdict(p, 90), verdict(p, 90)
    assert (a1.verdict, a1.intervention_delay_s) == (a2.verdict, a2.intervention_delay_s)

# ---------------- real Module 3 plan ----------------
pytest.importorskip("Module3.virtual_coupling", reason="set MODULE3_PATH")
from Module3.optimizer import FreightHeuristicSolverImproved
from Module3.virtual_coupling import VirtualCouplingPlanner
from Module4.adapters import plan_from_solver_result
from Module4.test_calibration_module3 import instance

def test_real_module3_convoys_resolved_by_tier2(capsys):
    trains, cargos, segs = instance(12)
    for t in trains:                       # same direction so convoys form on one corridor
        t["origin"], t["destination"] = "A", "B"
    for c in cargos:
        c["origin"], c["destination"] = "A", "B"
    res = FreightHeuristicSolverImproved(trains, cargos, segs, seed=7).solve()
    td = {t["id"]: t for t in trains}
    cv = VirtualCouplingPlanner().plan_convoys(res["best_solution"]["train_sequence"], td, res["train_arrivals"])
    plan = plan_from_solver_result(res, trains, cargos, segs, cv)
    t1 = check_plan(plan)
    run = simulate_plan(plan, loco="WAG_9", v2v_headway_hrs=0.05)
    a = assess(run, plan)
    print(f"\nconvoys={len(cv)} tier1_needs_tier2={t1.needs_tier2} -> verdict={a.verdict} "
          f"events={[e.kind for e in a.events]} max follower delay={a.max_intervention_delay_s:.0f}s")
    assert set(t1.needs_tier2) == {c['convoy_id'] for c in cv}
    assert a.verdict in {"ACCEPT", "DEGRADED", "REJECT"}
