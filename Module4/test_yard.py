import pytest
from hypothesis import given, settings, strategies as st
from Module4.cosim import simulate_plan
from Module4.params import YardParams
from Module4.schemas import Plan
from Module4.tier1_checker import check_plan
from Module4.conftest import make_plan, SEGMENTS

Y = YardParams()   # 2 slots, 800 t/h, 30 min crew (assumed)

def three_trains(slots_prio=None):
    trains = [{"id": f"T{i}", "origin": "A", "destination": "B", "ready_time_hrs": 0.0,
               "max_capacity_tons": 2600, "priority": pr} for i, pr in enumerate(slots_prio or [0.5, 0.5, 0.5])]
    cargos = [{"id": f"C{i}", "origin": "A", "destination": "B", "weight": 1600, "priority": 0.5,
               "due_date_offset_hrs": 99} for i in range(3)]
    return Plan(trains=trains, cargos=cargos, segments=SEGMENTS,
                train_sequence=[t["id"] for t in trains], allocation={f"C{i}": f"T{i}" for i in range(3)})

def test_instant_yard_equals_no_yard(plan):
    a = simulate_plan(plan, loco="IDEAL")
    b = simulate_plan(plan, loco="IDEAL", yard=YardParams.instant())
    assert a.arrivals_hrs == pytest.approx(b.arrivals_hrs, abs=1e-6)

def test_dwell_formula_spec_5_1(plan):
    run = simulate_plan(plan, loco="IDEAL", yard=Y)
    t1 = run.trains["T1"]                      # 2000 t @ 800 t/h = 2.5 h, + 0.5 h crew
    assert t1.load_end_hrs - t1.load_start_hrs == pytest.approx(2.5)
    assert t1.dwell_hrs == pytest.approx(3.0) and t1.depart_hrs == pytest.approx(3.0)
    t2 = run.trains["T2"]                      # parallel slot; ready 3.1 but headway holds to 3.25
    assert t2.ready_hrs == pytest.approx(3.1) and t2.depart_hrs == pytest.approx(3.25)
    assert t2.handling_delay_hrs == pytest.approx(3.25 - 0.1)

def test_crew_concurrent_mode(plan):
    run = simulate_plan(plan, loco="IDEAL", yard=YardParams(crew_concurrent=True))
    assert run.trains["T1"].dwell_hrs == pytest.approx(2.5)        # max(2.5, 0.5)

def test_slot_contention_serialises_loading():
    run = simulate_plan(three_trains(), loco="IDEAL", yard=YardParams(slots=1))   # 2 h load each
    w = sorted((r.load_start_hrs, r.load_end_hrs) for r in run.trains.values())
    assert all(b[0] >= a[1] - 1e-9 for a, b in zip(w, w[1:]))
    assert max(r.yard_wait_hrs for r in run.trains.values()) == pytest.approx(4.0)

def test_concurrent_loadings_never_exceed_slots():
    run = simulate_plan(three_trains(), loco="IDEAL", yard=YardParams(slots=2))
    ev = sorted([(r.load_start_hrs, 1) for r in run.trains.values()] +
                [(r.load_end_hrs, -1) for r in run.trains.values()], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d; peak = max(peak, cur)
    assert peak == 2

def test_priority_train_loads_first():
    run = simulate_plan(three_trains([0.2, 0.2, 0.9]), loco="IDEAL", yard=YardParams(slots=1))
    # T0 grabs the slot first (arrives first in queue order at t=0 tie); T2 must beat T1 for the next slot
    assert run.trains["T2"].load_start_hrs < run.trains["T1"].load_start_hrs

def test_yard_busy_hours_and_utilization(plan):
    run = simulate_plan(plan, loco="IDEAL", yard=Y)
    assert run.yard_busy_hrs["A"] == pytest.approx(5.0)            # 2 x 2.5 h
    assert run.trains["T1"].utilization == pytest.approx(2000 / 2600)

def test_deterministic(plan):
    a, b = simulate_plan(plan, yard=Y, seed=4), simulate_plan(plan, yard=Y, seed=4)
    assert a.arrivals_hrs == b.arrivals_hrs

# ---- every simulated train satisfies the Section 5.1 dwell minimum ----
@settings(max_examples=50, deadline=None)
@given(loads=st.lists(st.floats(0, 2600), min_size=1, max_size=8), slots=st.integers(1, 3),
       concurrent=st.booleans())
def test_property_dwell_minimum_holds(loads, slots, concurrent):
    trains = [{"id": f"T{i}", "origin": "A", "destination": "B", "ready_time_hrs": 0.1 * i,
               "max_capacity_tons": 2600} for i in range(len(loads))]
    cargos = [{"id": f"C{i}", "origin": "A", "destination": "B", "weight": w, "priority": 0.5,
               "due_date_offset_hrs": 99} for i, w in enumerate(loads)]
    p = Plan(trains=trains, cargos=cargos, segments=SEGMENTS, train_sequence=[t["id"] for t in trains],
             allocation={f"C{i}": f"T{i}" for i in range(len(loads))})
    yp = YardParams(slots=slots, crew_concurrent=concurrent)
    run = simulate_plan(p, loco="IDEAL", yard=yp)
    for i, w in enumerate(loads):
        r = run.trains[f"T{i}"]
        assert r.depart_hrs - r.yard_arrival_hrs >= yp.min_dwell_hrs(w) - 1e-9

# ---- Tier 1 dwell check ----
def test_tier1_dwell_too_short_detected():
    p = make_plan(claimed_arrivals=None)
    p.trains[0].yard_arrival_hrs = 0.0            # needs 3.0 h dwell but Module 3 departs at 0.0
    r = check_plan(p)
    assert "DWELL_TOO_SHORT" in {x.code for x in r.violations} and not r.passed

def test_tier1_dwell_ok_when_ready_late_enough():
    p = make_plan(claimed_arrivals=None)
    p.trains[0].yard_arrival_hrs = 0.0; p.trains[0].ready_time_hrs = 3.0
    p.trains[1].yard_arrival_hrs = 0.0; p.trains[1].ready_time_hrs = 3.0
    assert "DWELL_TOO_SHORT" not in {x.code for x in check_plan(p).violations}

def test_tier1_and_sim_agree_on_minimum_dwell(plan):
    """Sim dwell == Tier 1 necessary minimum when uncontended."""
    run = simulate_plan(plan, loco="IDEAL", yard=Y)
    assert run.trains["T1"].dwell_hrs == pytest.approx(Y.min_dwell_hrs(2000))

# ---- real Module 3 solver ----
pytest.importorskip("Module3.optimizer", reason="set MODULE3_PATH")
from Module3.optimizer import FreightHeuristicSolverImproved
from Module4.adapters import plan_from_solver_result
from Module4.test_calibration_module3 import instance

def test_real_plan_handling_delay_report():
    trains, cargos, segs = instance(10)
    res = FreightHeuristicSolverImproved(trains, cargos, segs, seed=7).solve()
    plan = plan_from_solver_result(res, trains, cargos, segs)
    assert simulate_plan(plan, loco="IDEAL", yard=YardParams.instant()).arrivals_hrs == \
        pytest.approx(res["train_arrivals"], abs=1 / 3600)
    run = simulate_plan(plan, loco="WAG_9", yard=Y)
    d = [r.handling_delay_hrs for r in run.trains.values()]
    late = sum(run.arrivals_hrs[t] > res["train_arrivals"][t] + 1e-9 for t in run.trains)
    print(f"\nreal plan, 10 trains: handling delay min {min(d):.2f} h / max {max(d):.2f} h; "
          f"{late}/10 arrive later than Module 3 claims; yard busy {run.yard_busy_hrs}")
    for r in run.trains.values():
        assert r.depart_hrs - r.yard_arrival_hrs >= Y.min_dwell_hrs(r.load_tons) - 1e-9
