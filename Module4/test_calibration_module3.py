"""Calibration: Tier 1 must agree with the real Module 3 solver output."""
import pytest
pytest.importorskip("Module3.optimizer", reason="set MODULE3_PATH to the dir containing Module3/")
from Module3.optimizer import FreightHeuristicSolverImproved
from Module4.adapters import plan_from_solver_result
from Module4.tier1_checker import check_plan

def instance(n_trains=10, seed=1):
    import random; r = random.Random(seed)
    routes = [("A", "B"), ("A", "C"), ("D", "B")]
    trains, cargos = [], []
    for i in range(n_trains):
        o, d = routes[i % 3]
        trains.append({"id": f"T{i}", "origin": o, "destination": d, "segment_id": "CORRIDOR_MAIN",
                       "ready_time_hrs": round(r.uniform(0, 4), 2), "max_capacity_tons": 2600, "priority": 0.5})
    for j in range(int(n_trains * 1.5)):
        o, d = routes[j % 3]
        cargos.append({"id": f"C{j}", "origin": o, "destination": d, "weight": r.choice([800, 1200, 1600]),
                       "priority": r.choice([0.5, 0.8]), "due_date_offset_hrs": r.uniform(3, 12)})
    segs = [{"id": "CORRIDOR_MAIN", "traversing_time_hrs": 2.2, "safe_headway_hrs": 0.25}]
    return trains, cargos, segs

@pytest.mark.parametrize("n", [5, 10, 30, 50])
def test_module3_plan_matches_tier1(n):
    trains, cargos, segs = instance(n)
    res = FreightHeuristicSolverImproved(trains, cargos, segs, seed=7).solve()
    rep = check_plan(plan_from_solver_result(res, trains, cargos, segs))
    # arrivals, capacity, allocation and headway must all agree with Module 3
    for code in ("ARRIVAL_MISMATCH", "HEADWAY_VIOLATION", "CAPACITY_EXCEEDED",
                 "ROUTE_MISMATCH", "REMAINING_CAPACITY_MISMATCH", "SEQUENCE_NOT_PERMUTATION"):
        assert code not in {x.code for x in rep.violations}, rep.violations
    assert rep.recomputed_arrivals == pytest.approx(res["train_arrivals"])
    assert rep.elapsed_ms < 50
