"""Week-19 exit test: simulated arrivals must reproduce Module 3's train_arrivals."""
import mesa, pytest
from Module4.cosim import simulate_plan
from Module4.train_agent import TrainAgent, Status
from Module4.conftest import make_plan

def test_single_train_matches_module3_closed_form():
    p = make_plan(trains=[make_plan().trains[0].model_dump()], cargos=[], allocation={},
                  train_sequence=["T1"], remaining_capacity=None, claimed_arrivals=None)
    run = simulate_plan(p, loco="IDEAL")
    assert run.trains["T1"].depart_hrs == pytest.approx(0.0)
    assert run.trains["T1"].arrive_hrs == pytest.approx(2.0, abs=1e-3)     # ready 0 + traversing 2.0 h

def test_two_trains_headway_enforced_by_simpy_lock(plan):
    run = simulate_plan(plan, loco="IDEAL")
    assert run.trains["T2"].depart_hrs == pytest.approx(0.25, abs=1e-4)    # max(ready 0.1, 0+0.25)
    assert run.arrivals_hrs["T2"] == pytest.approx(plan.claimed_arrivals["T2"], abs=1e-3)

def test_agents_are_registered_mesa_agents(plan):
    import Module4.cosim as c
    seen = {}
    orig = c.TrainAgent.__init__
    def spy(self, model, *a, **k):
        orig(self, model, *a, **k); seen.setdefault("model", model)
    c.TrainAgent.__init__ = spy
    try: simulate_plan(plan, loco="IDEAL", seed=3)
    finally: c.TrainAgent.__init__ = orig
    assert len(seen["model"].agents) == 2 and all(isinstance(a, TrainAgent) for a in seen["model"].agents)

def test_deterministic(plan):
    a, b = simulate_plan(plan, seed=1), simulate_plan(plan, seed=1)
    assert a.arrivals_hrs == b.arrivals_hrs

def test_realistic_physics_is_slower_than_module3(plan):
    ideal, real = simulate_plan(plan, loco="IDEAL"), simulate_plan(plan, loco="WAG_9")
    for tid in ("T1", "T2"):
        d = real.arrivals_hrs[tid] - ideal.arrivals_hrs[tid]
        assert 0 < d < 0.25, d          # accel/brake costs minutes, not hours

# ---- real Module 3 solver ----
pytest.importorskip("Module3.optimizer", reason="set MODULE3_PATH")
from Module3.optimizer import FreightHeuristicSolverImproved
from Module4.adapters import plan_from_solver_result
from Module4.test_calibration_module3 import instance

@pytest.mark.parametrize("n", [1, 5, 10, 30, 50])
def test_ideal_sim_reproduces_module3_arrivals(n):
    trains, cargos, segs = instance(n) if n > 1 else instance(2)
    res = FreightHeuristicSolverImproved(trains, cargos, segs, seed=7).solve()
    plan = plan_from_solver_result(res, trains, cargos, segs)
    run = simulate_plan(plan, loco="IDEAL")
    for tid, m3 in res["train_arrivals"].items():
        assert run.arrivals_hrs[tid] == pytest.approx(m3, abs=1.0 / 3600), (tid, m3)   # within 1 s

def test_report_physics_delta_vs_module3(capsys):
    trains, cargos, segs = instance(10)
    res = FreightHeuristicSolverImproved(trains, cargos, segs, seed=7).solve()
    plan = plan_from_solver_result(res, trains, cargos, segs)
    real = simulate_plan(plan, loco="WAG_9")
    deltas = [(real.arrivals_hrs[t] - m3) * 60 for t, m3 in res["train_arrivals"].items()]
    print(f"\nWAG-9 physics delta vs Module 3: min {min(deltas):.1f}  max {max(deltas):.1f} min")
    assert all(d > 0 for d in deltas)
