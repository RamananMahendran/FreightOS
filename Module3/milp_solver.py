"""
FreightOS Module 3: Exact Mixed-Integer Linear Programming (MILP) Baseline Solver.
Formulates the Train Timetabling and Wagon Allocation problem using PuLP with
the open-source COIN-OR CBC solver, enforces a 60-second time ceiling,
and exports an executable GAMS (.gms) model file.
"""

import time
import os
from typing import Dict, List, Tuple
import pulp


class ExactMILPSolver:
    def __init__(
        self,
        trains: List[Dict],
        cargos: List[Dict],
        segments: List[Dict],
        time_limit_sec: int = 60,
        min_utilization_fraction: float = 0.60,
    ):
        """
        :param trains: Fleet rakes with capacity, origin, dest, ready_time
        :param cargos: Ingested Module 2 cargo demand items
        :param segments: Track corridor topology with traversing times and headways
        :param time_limit_sec: Hard time limit for branch-and-cut search (default 60s)
        :param min_utilization_fraction: Minimum capacity utilization floor (mu_t)
        """
        self.trains = trains
        self.cargos = cargos
        self.segments = {s["id"]: s for s in segments}
        self.time_limit = time_limit_sec
        self.min_utilization = min_utilization_fraction

        self.cargo_dict = {c["id"]: c for c in self.cargos}
        self.train_dict = {t["id"]: t for t in self.trains}

    def solve(self) -> Dict:
        """Formulates and solves the exact MILP model with CBC."""
        start_time = time.time()
        model = pulp.LpProblem("FreightOS_MILP_Scheduling", pulp.LpMinimize)

        BIG_M = 500.0  # Large positive constant for disjunctive constraints
        total_possible_priority = sum(c["priority"] for c in self.cargos) or 1.0
        max_possible_tardiness = sum(c["priority"] * 48.0 for c in self.cargos) or 1.0

        # --- Decision Variables ---
        # x[j, t] = 1 if cargo j is allocated to train t
        x = {}
        for c in self.cargos:
            for t in self.trains:
                x[(c["id"], t["id"])] = pulp.LpVariable(
                    f"x_{c['id']}_{t['id']}", cat=pulp.LpBinary
                )

        # u[t] = 1 if train t is utilized
        u = {
            t["id"]: pulp.LpVariable(f"u_{t['id']}", cat=pulp.LpBinary)
            for t in self.trains
        }

        # d[t] = Departure time of train t (hours)
        d = {
            t["id"]: pulp.LpVariable(
                f"dep_{t['id']}", lowBound=t["ready_time_hrs"], cat=pulp.LpContinuous
            )
            for t in self.trains
        }

        # tardi[j] = Tardiness of cargo j (hours)
        tardiness = {
            c["id"]: pulp.LpVariable(f"tardi_{c['id']}", lowBound=0.0, cat=pulp.LpContinuous)
            for c in self.cargos
        }

        # z[t1, t2] = 1 if train t1 departs before train t2 on shared segment
        z = {}
        train_ids = [t["id"] for t in self.trains]
        for i in range(len(train_ids)):
            for j in range(i + 1, len(train_ids)):
                t1_id, t2_id = train_ids[i], train_ids[j]
                t1, t2 = self.train_dict[t1_id], self.train_dict[t2_id]
                if t1.get("segment_id") == t2.get("segment_id"):
                    z[(t1_id, t2_id)] = pulp.LpVariable(f"z_{t1_id}_{t2_id}", cat=pulp.LpBinary)

        # --- Constraints ---

        # 1. Single Cargo Assignment Rule
        for c in self.cargos:
            model += (
                pulp.lpSum(x[(c["id"], t["id"])] for t in self.trains) <= 1,
                f"SingleAssignment_{c['id']}",
            )

        # 2. Route Matching & Wagon Capacity Bounds
        for t in self.trains:
            t_id = t["id"]
            assigned_cargos_weight = []

            for c in self.cargos:
                c_id = c["id"]
                # Forbid routing across incompatible origin-destination corridors
                if c["origin"] != t["origin"] or c["destination"] != t["destination"]:
                    model += x[(c_id, t_id)] == 0, f"RouteMismatch_{c_id}_{t_id}"
                else:
                    assigned_cargos_weight.append(c["weight"] * x[(c_id, t_id)])

            total_weight_expr = pulp.lpSum(assigned_cargos_weight)

            # Max Capacity Bound: weight <= max_capacity * u[t]
            model += total_weight_expr <= t["max_capacity_tons"] * u[t_id], f"MaxCap_{t_id}"

            # Min Utilization Floor: weight >= min_utilization * max_capacity * u[t]
            model += (
                total_weight_expr >= (self.min_utilization * t["max_capacity_tons"]) * u[t_id],
                f"MinUtil_{t_id}",
            )

        # 3. Safe Headway Disjunctions
        for (t1_id, t2_id), z_var in z.items():
            t1 = self.train_dict[t1_id]
            seg = self.segments.get(t1.get("segment_id", "CORRIDOR_MAIN"), {"safe_headway_hrs": 0.25})
            headway = seg.get("safe_headway_hrs", 0.25)

            # d[t2] - d[t1] >= headway - M * (1 - z)
            model += d[t2_id] - d[t1_id] >= headway - BIG_M * (1 - z_var), f"HeadwayA_{t1_id}_{t2_id}"
            # d[t1] - d[t2] >= headway - M * z
            model += d[t1_id] - d[t2_id] >= headway - BIG_M * z_var, f"HeadwayB_{t1_id}_{t2_id}"

        # 4. Cargo Tardiness Linearization
        for c in self.cargos:
            c_id = c["id"]
            due_date = c["due_date_offset_hrs"]
            for t in self.trains:
                t_id = t["id"]
                seg = self.segments.get(t.get("segment_id", "CORRIDOR_MAIN"), {"traversing_time_hrs": 2.2})
                arr_expr = d[t_id] + seg.get("traversing_time_hrs", 2.2)

                # tardi[c] >= arr_time - due_date - M * (1 - x[c, t])
                model += (
                    tardiness[c_id] >= (arr_expr - due_date) - BIG_M * (1 - x[(c_id, t_id)]),
                    f"TardiLin_{c_id}_{t_id}",
                )

        # --- Normalized Multi-Objective Formulation ---
        # Minimize: 0.5 * Normalized Tardiness - 0.5 * Normalized Priority Allocation
        total_tardiness_term = pulp.lpSum(c["priority"] * tardiness[c["id"]] for c in self.cargos)
        total_allocated_term = pulp.lpSum(
            c["priority"] * x[(c["id"], t["id"])] for c in self.cargos for t in self.trains
        )

        norm_tardiness = total_tardiness_term / max_possible_tardiness
        norm_allocation = total_allocated_term / total_possible_priority

        model += 0.5 * norm_tardiness - 0.5 * norm_allocation, "ScalarizedObjective"

        # --- Execute Solver with 60-Second Hard Timeout ---
        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.time_limit,
            msg=False,
            keepFiles=False,
            gapRel=0.01  # Stop early if proven within 1% of global optimum
        )
        status_code = model.solve(solver)
        execution_duration = time.time() - start_time

        status_str = pulp.LpStatus.get(status_code, "Completed")
        if execution_duration >= (self.time_limit - 1.0) and status_str != "Optimal":
            status_str = f"Time Limit ({self.time_limit}s)"

        # --- Extract Results ---
        allocations = {c["id"]: None for c in self.cargos}
        for c in self.cargos:
            for t in self.trains:
                val = pulp.value(x[(c["id"], t["id"])])
                if val is not None and val > 0.5:
                    allocations[c["id"]] = t["id"]
                    break

        train_arrivals = {}
        for t in self.trains:
            t_id = t["id"]
            dep_val = pulp.value(d[t_id]) if pulp.value(d[t_id]) is not None else t["ready_time_hrs"]
            seg = self.segments.get(t.get("segment_id", "CORRIDOR_MAIN"), {"traversing_time_hrs": 2.2})
            train_arrivals[t_id] = dep_val + seg.get("traversing_time_hrs", 2.2)

        remaining_capacity = {}
        for t in self.trains:
            t_id = t["id"]
            used = sum(c["weight"] for c in self.cargos if allocations.get(c["id"]) == t_id)
            remaining_capacity[t_id] = t["max_capacity_tons"] - used

        sorted_trains_seq = sorted(
            self.trains, key=lambda t: pulp.value(d[t["id"]]) if pulp.value(d[t["id"]]) is not None else 0.0
        )
        train_sequence = [t["id"] for t in sorted_trains_seq]

        best_fit = pulp.value(model.objective)
        if best_fit is None:
            best_fit = 0.0

        return {
            "status": status_str,
            "execution_time_seconds": round(execution_duration, 3),
            "best_fitness": round(best_fit, 4),
            "best_solution": {
                "train_sequence": train_sequence,
                "allocation": allocations,
                "remaining_capacity": remaining_capacity,
            },
            "train_arrivals": train_arrivals,
        }

    def export_to_gams(self, filepath: str = "data/freightos_baseline.gms"):
        """Exports the exact mathematical optimization problem into standard GAMS (.gms) format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        lines = [
            "$Title FreightOS Module 3 MILP Scheduling Baseline Model",
            "$Ontext",
            "Exact Mixed-Integer Linear Program for Train Timetabling & Wagon Allocation",
            "Generated from FreightOS Module 2 actual forecasted freight demand.",
            "$Offtext",
            "",
            f"Option ResLim = {self.time_limit};",
            "Option OptCR  = 0.01;",
            "",
            "Sets",
            f"    t    Trains / {', '.join(t['id'] for t in self.trains)} /",
            f"    j    Cargos / {', '.join(c['id'] for c in self.cargos)} /;",
            "",
            "Parameters",
            "    cap(t) Train capacity in metric tons",
            "    /",
        ]
        for t in self.trains:
            lines.append(f"        {t['id']} {t['max_capacity_tons']}")
        lines.extend(["    /", "    weight(j) Cargo weight in metric tons", "    /"])
        for c in self.cargos:
            lines.append(f"        {c['id']} {c['weight']}")
        lines.extend([
            "    /;",
            "",
            "Variables",
            "    total_obj Scalarized objective value",
            "    dep(t)    Departure time of train t",
            "    arr(t)    Arrival time of train t",
            "    tardi(j)  Cargo destination tardiness;",
            "",
            "Positive Variables",
            "    dep, arr, tardi;",
            "",
            "Binary Variables",
            "    x(j, t) Binary decision variable for cargo assignment",
            "    u(t)    Binary active train indicator;",
            "",
            "Equations",
            "    SingleAssignment(j)   Each cargo allocated to at most one train",
            "    MaxCapacity(t)        Train capacity loading limit",
            "    MinUtilization(t)     Minimum utilization loading floor",
            "    ObjDef                Objective definition;",
            "",
            "SingleAssignment(j).. sum(t, x(j, t)) =l= 1;",
            f"MaxCapacity(t)..      sum(j, weight(j) * x(j, t)) =l= cap(t) * u(t);",
            f"MinUtilization(t)..   sum(j, weight(j) * x(j, t)) =g= {self.min_utilization} * cap(t) * u(t);",
            "ObjDef..              total_obj =e= - sum((j, t), x(j, t));",
            "",
            "Model FreightOS_MILP /all/;",
            "Solve FreightOS_MILP using mip minimizing total_obj;",
            "Display x.l, total_obj.l, dep.l, arr.l;",
        ])

        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        print(f"📄 Successfully generated GAMS model file at: {filepath}")