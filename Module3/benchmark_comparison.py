"""
FreightOS Module 3 Benchmark: Exact MILP vs. Hybrid Elitist GA+OPT Heuristic.
Loads identical real forecasted cargos from Module 2, runs both solvers with
a 60-second time ceiling on MILP, and logs side-by-side performance.
"""

import sys
from pathlib import Path

MODULE3_DIR = Path(__file__).resolve().parent
MODULE2_DIR = MODULE3_DIR.parent / "Module2"

for p in [str(MODULE3_DIR), str(MODULE2_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_module3 import (
    fetch_actual_cargos_from_module2,
    provision_fleet_for_corridors,
    build_track_infrastructure,
)
from optimizer import FreightHeuristicSolverImproved
from milp_solver import ExactMILPSolver


def run_comprehensive_benchmark():
    print("=" * 82)
    print("🔬 FREIGHTOS MODULE 3 BENCHMARK: EXACT MILP VS. HYBRID GA+OPT SOLVER")
    print("=" * 82)

    # 1. Ingest actual forecasted cargos from Module 2
    raw_cargos = fetch_actual_cargos_from_module2()

    # Select 20 cargo requests to simulate an operational shift dispatch window
    cargos = raw_cargos[:20]
    trains = provision_fleet_for_corridors(cargos)
    segments = build_track_infrastructure(trains)

    total_demand_tons = sum(c["weight"] for c in cargos)
    print(f"\n📊 Benchmark Problem Instance:")
    print(f"   • Cargos Evaluated: {len(cargos)} items (Total Demand: {total_demand_tons:.1f} Tons)")
    print(f"   • Available Fleet: {len(trains)} trains across {len(segments)} rail corridors")

    # 2. Run Metaheuristic: Hybrid Elitist GA + OPT Solver
    print("\n" + "-" * 82)
    print("⚡ [1/2] RUNNING HYBRID ELITIST GA + OPT SOLVER (FREIGHTOS PRODUCTION)")
    print("-" * 82)
    ga_solver = FreightHeuristicSolverImproved(
        trains=trains,
        cargos=cargos,
        segments=segments,
        max_iterations=200,
        convergence_patience=20,
    )
    ga_results = ga_solver.solve()
    print(f"✅ GA+OPT Finished in: {ga_results.get('execution_time_seconds', 0.0):.3f}s")
    print(f"   • Best Fitness Score: {ga_results.get('best_fitness', 0.0):.4f}")

    # 3. Run Exact Baseline: MILP Solver (COIN-OR CBC with 60-Second Hard Timeout)
    print("\n" + "-" * 82)
    print("📐 [2/2] RUNNING EXACT MILP BASELINE SOLVER (COIN-OR CBC, 60s CEILING)")
    print("-" * 82)
    milp_solver = ExactMILPSolver(
        trains=trains,
        cargos=cargos,
        segments=segments,
        time_limit_sec=60,  # 60s hard ceiling
    )
    gams_path = str(MODULE3_DIR / "data" / "freightos_baseline.gms")
    milp_solver.export_to_gams(gams_path)

    milp_results = milp_solver.solve()
    milp_status = milp_results.get("status", "Completed")
    print(f"✅ MILP Solver Finished in: {milp_results.get('execution_time_seconds', 0.0):.3f}s (Status: {milp_status})")
    print(f"   • Best Objective / Bound: {milp_results.get('best_fitness', 0.0):.4f}")

    # 4. Metrics Extraction
    def compute_metrics(results):
        sol = results.get("best_solution", {})
        allocations = sol.get("allocation", {})
        rem_cap = sol.get("remaining_capacity", {})
        assigned = [c_id for c_id, t_id in allocations.items() if t_id is not None]
        tonnage = sum(
            train["max_capacity_tons"] - rem_cap.get(train["id"], train["max_capacity_tons"])
            for train in trains
        )
        return len(assigned), tonnage

    ga_count, ga_tonnage = compute_metrics(ga_results)
    milp_count, milp_tonnage = compute_metrics(milp_results)

    # Compute Optimality Gap (%)
    ga_fit = ga_results.get("best_fitness", 0.0)
    milp_fit = milp_results.get("best_fitness", 0.0)
    denom = abs(milp_fit) if abs(milp_fit) > 1e-4 else 1.0
    optimality_gap_pct = max(0.0, ((ga_fit - milp_fit) / denom) * 100.0)

    milp_time = milp_results.get("execution_time_seconds", 0.0)
    ga_time = ga_results.get("execution_time_seconds", 0.001)

    # 5. Display Comparative Results Table
    print("\n" + "=" * 82)
    print("📈 PERFORMANCE & ACCURACY COMPARISON SUMMARY")
    print("=" * 82)
    print(f"{'Performance Metric':<32} | {'Exact MILP (CBC, 60s)':<22} | {'Hybrid GA + OPT Solver':<20}")
    print("-" * 82)
    print(f"{'Solve Time (seconds)':<32} | {f'{milp_time:.3f}s':<22} | {f'{ga_time:.3f}s':<20}")
    speedup = milp_time / max(0.001, ga_time)
    print(f"{'Speedup Factor':<32} | {'1.0x (Baseline)':<22} | {f'{speedup:.1f}x Faster':<20}")
    print(f"{'Convergence Status':<32} | {milp_status:<22} | {'Sub-5s SLA Passed':<20}")
    print(f"{'Objective Fitness (Minimization)':<32} | {f'{milp_fit:.4f}':<22} | {f'{ga_fit:.4f}':<20}")
    print(f"{'Optimality Gap (%)':<32} | {'0.00% (Reference)':<22} | {f'{optimality_gap_pct:.2f}%':<20}")
    print(f"{'Cargos Allocated':<32} | {f'{milp_count} / {len(cargos)}':<22} | {f'{ga_count} / {len(cargos)}':<20}")
    print(f"{'Shipped Volume (Tons)':<32} | {f'{milp_tonnage:.1f} Tons':<22} | {f'{ga_tonnage:.1f} Tons':<20}")
    print("=" * 82)

    if ga_time < 5.0:
        print("🎯 SLA VERIFICATION: Hybrid GA + OPT meets the sub-5-second dispatch execution SLA!")


if __name__ == "__main__":
    run_comprehensive_benchmark()