import pytest
from hypothesis import given, settings, strategies as st
from Module4.schemas import Plan
from Module4.tier1_checker import check_plan, Tier1Config
from Module4.conftest import make_plan, SEGMENTS

def codes(rep): return {x.code for x in rep.violations}

def test_valid_plan_passes(plan):
    r = check_plan(plan)
    assert r.passed, r.violations
    assert r.recomputed_arrivals == pytest.approx({"T1": 2.0, "T2": 2.25})
    assert r.weighted_tardiness == pytest.approx(0.0)   # arrivals 2.0/2.25 <= due 3.0

def test_tardiness_computed():
    p = make_plan(); p.cargos[1].due_date_offset_hrs = 2.0
    r = check_plan(p)
    assert r.cargo_tardiness_hrs["C2"] == pytest.approx(0.25)
    assert r.weighted_tardiness == pytest.approx(0.5 * 0.25)

def test_corrupted_arrival_detected(plan):
    plan.claimed_arrivals["T2"] = 2.05            # departs 0.05 h after T1 -> unsafe & inconsistent
    r = check_plan(plan)
    assert not r.passed and {"ARRIVAL_MISMATCH", "HEADWAY_VIOLATION"} <= codes(r)

def test_departs_before_ready(plan):
    plan.claimed_arrivals["T2"] = 2.0 + 0.25 - 0.3
    assert "DEPARTS_BEFORE_READY" in codes(check_plan(plan))

def test_capacity_exceeded(plan):
    plan.cargos[0].weight = 3000
    r = check_plan(plan)
    assert "CAPACITY_EXCEEDED" in codes(r) and "REMAINING_CAPACITY_MISMATCH" in codes(r)

def test_route_mismatch(plan):
    plan.cargos[0].destination = "Z"
    assert "ROUTE_MISMATCH" in codes(check_plan(plan))

def test_unknown_ids(plan):
    plan.allocation["C1"] = "T99"; plan.allocation["CX"] = "T1"
    assert {"ALLOC_UNKNOWN_TRAIN", "ALLOC_UNKNOWN_CARGO"} <= codes(check_plan(plan))

@pytest.mark.parametrize("seq", [["T1"], ["T1", "T1"], ["T1", "T2", "T3"]])
def test_bad_sequence(plan, seq):
    plan.train_sequence = seq
    assert "SEQUENCE_NOT_PERMUTATION" in codes(check_plan(plan))

def test_utilization_floor_warning_vs_error():
    p = make_plan(allocation={"C1": "T1", "C2": None}, remaining_capacity={"T1": 600, "T2": 2800})
    p.cargos[0].weight = 1000   # 1000/2600 = 38%
    p.remaining_capacity["T1"] = 1600
    r = check_plan(p)
    assert r.passed and "UTILIZATION_BELOW_FLOOR" in codes(r)
    r2 = check_plan(p, Tier1Config(enforce_min_utilization=True))
    assert not r2.passed

def test_opposing_conflict_single_line():
    p = make_plan()
    p.trains[1].origin, p.trains[1].destination = "B", "A"
    p.cargos[1].origin, p.cargos[1].destination = "B", "A"
    p.claimed_arrivals = None
    r = check_plan(p)       # T2 departs 0.25 h while T1 occupies the line until 2.0 h
    assert "OPPOSING_CONFLICT" in codes(r)
    assert r.violations[[x.code for x in r.violations].index("OPPOSING_CONFLICT")].location == "CORRIDOR_MAIN"

def test_convoy_flagged_for_tier2(plan):
    p = make_plan(convoys=[{"convoy_id": "V1", "leader_train": "T1", "follower_trains": ["T2"]}])
    r = check_plan(p)
    assert r.passed and r.needs_tier2 == ["V1"]

def test_convoy_structural_errors(plan):
    p = make_plan(convoys=[{"convoy_id": "V1", "leader_train": "T1", "follower_trains": ["T2"]}])
    p.trains[1].ready_time_hrs = 1.0
    assert "CONVOY_DEPARTURE_SPREAD" in codes(check_plan(p))

def test_missing_segment_uses_module3_defaults():
    p = make_plan(segments=[], claimed_arrivals=None)
    r = check_plan(p)
    assert "SEGMENT_DEFAULTED" in codes(r)
    assert r.recomputed_arrivals["T1"] == pytest.approx(2.2)

# ---- property: any Module-3-semantics schedule of same-direction trains is headway-safe ----
@settings(max_examples=100, deadline=None)
@given(ready=st.lists(st.floats(0, 12), min_size=1, max_size=15), data=st.data())
def test_property_recomputed_schedule_is_safe(ready, data):
    n = len(ready)
    trains = [{"id": f"T{i}", "origin": "A", "destination": "B", "ready_time_hrs": r,
               "max_capacity_tons": 2600} for i, r in enumerate(ready)]
    seq = data.draw(st.permutations([t["id"] for t in trains]))
    p = Plan(trains=trains, cargos=[], segments=SEGMENTS, train_sequence=list(seq), allocation={})
    assert check_plan(p).passed

# ---- property: shrinking any gap below the headway is always caught ----
@settings(max_examples=100, deadline=None)
@given(gap=st.floats(0.0, 0.24))
def test_property_unsafe_gap_detected(gap):
    p = make_plan()
    p.claimed_arrivals = {"T1": 2.0, "T2": 2.0 + gap}
    assert "HEADWAY_VIOLATION" in codes(check_plan(p))
