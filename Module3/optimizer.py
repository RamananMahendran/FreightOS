"""
FreightOS Module 3: Improved Hybrid Elitist GA + OPT Solver

Changes vs. the original optimizer.py, all targeted at the specific issues
found in review (see accompanying analysis):

1. Fitness caching  - every solution dict carries its own '_fitness' value,
   computed once and invalidated only when the solution actually changes.
   The original recomputed fitness_function() 2-3x per individual per
   generation (once in sorted(), again for current_fitness, again inside
   opt_local_search's internal loop) purely to re-derive a value it already
   had moments earlier.
2. True elitism - the top-k solutions are cloned unchanged into the next
   generation instead of being subjected to _apply_sequence_corrections()
   like everyone else. The original had no elitism: best_solution was
   tracked out-of-band, but the *population* could regress generation to
   generation because even elite members got mutated.
3. Seeded RNG - solve() accepts an optional seed so baseline and improved
   runs (or repeated improved runs) are directly comparable instead of being
   confounded by randomness.
4. Edge-case safety - max_iterations=0 and an all-infeasible population no
   longer crash on `iteration` being unbound or best_solution being None.
5. Widened local search neighborhood (optional) - in addition to adjacent
   swaps, samples a handful of non-adjacent swaps so the search isn't
   restricted to a 1-neighborhood of transpositions.
"""

import copy
import random
import time
from typing import Dict, List, Tuple, Optional


class FreightHeuristicSolverImproved:
    def __init__(
        self,
        trains: List[Dict],
        cargos: List[Dict],
        segments: List[Dict],
        max_iterations: int = 250,
        convergence_patience: int = 25,
        population_size: int = 40,
        min_utilization_fraction: float = 0.60,
        elite_carry_count: int = 3,
        local_search_extra_samples: int = 4,
        seed: Optional[int] = None,
    ):
        self.trains = trains
        self.cargos = cargos
        self.segments = {s["id"]: s for s in segments}
        self.max_iterations = max_iterations
        self.convergence_patience = convergence_patience
        self.population_size = population_size
        self.min_utilization = min_utilization_fraction
        self.elite_carry_count = elite_carry_count
        self.local_search_extra_samples = local_search_extra_samples
        self.rng = random.Random(seed)

        self.cargo_dict = {c["id"]: c for c in self.cargos}
        self.train_dict = {t["id"]: t for t in self.trains}
        self.total_possible_priority = sum(c["priority"] for c in self.cargos) or 1.0
        self.max_possible_tardiness = sum(c["priority"] * 48.0 for c in self.cargos) or 1.0

    # ---------- construction ----------

    def generate_initial_solution(self) -> Dict:
        sorted_trains = sorted(
            self.trains, key=lambda t: (t["priority"] * t["max_capacity_tons"]), reverse=True
        )
        remaining_capacity = {t["id"]: t["max_capacity_tons"] for t in self.trains}
        allocation = {c["id"]: None for c in self.cargos}

        sorted_cargos = sorted(
            self.cargos, key=lambda c: (c["priority"], -c["due_date_offset_hrs"]), reverse=True
        )
        for cargo in sorted_cargos:
            for train in sorted_trains:
                t_id = train["id"]
                if cargo["origin"] == train["origin"] and cargo["destination"] == train["destination"]:
                    if remaining_capacity[t_id] >= cargo["weight"]:
                        allocation[cargo["id"]] = t_id
                        remaining_capacity[t_id] -= cargo["weight"]
                        break

        train_seq = [t["id"] for t in sorted_trains]
        self.rng.shuffle(train_seq)

        sol = {
            "train_sequence": train_seq,
            "allocation": allocation,
            "remaining_capacity": remaining_capacity,
            "_fitness": None,
        }
        return sol

    # ---------- evaluation ----------

    def evaluate_schedule(self, solution: Dict) -> Dict[str, float]:
        train_arrivals = {}
        segment_occupancy_until = {s_id: 0.0 for s_id in self.segments}

        for t_id in solution["train_sequence"]:
            train = self.train_dict[t_id]
            current_time = float(train["ready_time_hrs"])
            assigned_segment_id = train.get("segment_id", "CORRIDOR_MAIN")
            seg = self.segments.get(
                assigned_segment_id, {"traversing_time_hrs": 2.5, "safe_headway_hrs": 0.25}
            )
            departure_time = max(current_time, segment_occupancy_until[assigned_segment_id])
            arrival_time = departure_time + seg["traversing_time_hrs"]
            segment_occupancy_until[assigned_segment_id] = departure_time + seg["safe_headway_hrs"]
            train_arrivals[t_id] = arrival_time

        return train_arrivals

    def fitness_function(self, solution: Dict, use_cache: bool = True) -> float:
        if use_cache and solution.get("_fitness") is not None:
            return solution["_fitness"]

        train_arrivals = self.evaluate_schedule(solution)
        total_tardiness_penalty = 0.0
        allocated_priority = 0.0

        for c_id, t_id in solution["allocation"].items():
            if t_id is not None:
                cargo = self.cargo_dict[c_id]
                allocated_priority += cargo["priority"]
                arr_time = train_arrivals.get(t_id, 0.0)
                tardiness = max(0.0, arr_time - cargo["due_date_offset_hrs"])
                total_tardiness_penalty += cargo["priority"] * tardiness

        utilization_penalty = 0.0
        for t_id, train in self.train_dict.items():
            used_cap = train["max_capacity_tons"] - solution["remaining_capacity"][t_id]
            if used_cap > 0:
                utilization_ratio = used_cap / train["max_capacity_tons"]
                if utilization_ratio < self.min_utilization:
                    utilization_penalty += (self.min_utilization - utilization_ratio) * 0.5

        norm_tardiness = total_tardiness_penalty / self.max_possible_tardiness
        norm_allocation = allocated_priority / self.total_possible_priority

        w_tardiness, w_allocation = 0.5, 0.5
        fitness = (w_tardiness * norm_tardiness) - (w_allocation * norm_allocation) + utilization_penalty

        if use_cache:
            solution["_fitness"] = fitness
        return fitness

    # ---------- local search ----------

    def opt_local_search(self, solution: Dict) -> Dict:
        best_sol = solution
        best_fit = self.fitness_function(best_sol)
        seq = best_sol["train_sequence"]
        n = len(seq)

        # Adjacent swaps (as original) ...
        candidate_positions = [(i, i + 1) for i in range(n - 1)]
        # ...plus a handful of non-adjacent swaps to widen the neighborhood.
        for _ in range(min(self.local_search_extra_samples, max(0, n - 2))):
            i, j = self.rng.sample(range(n), 2)
            candidate_positions.append((i, j))

        for i, j in candidate_positions:
            new_seq = list(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            candidate_sol = {
                "train_sequence": new_seq,
                "allocation": best_sol["allocation"],
                "remaining_capacity": best_sol["remaining_capacity"],
                "_fitness": None,
            }
            candidate_fit = self.fitness_function(candidate_sol)
            if candidate_fit < best_fit:
                best_sol = candidate_sol
                best_fit = candidate_fit
                seq = best_sol["train_sequence"]

        return best_sol

    def _apply_sequence_corrections(self, current: List[str], reference: List[str]) -> List[str]:
        corrected = list(current)
        num_corrections = self.rng.randint(1, max(1, int(0.25 * len(current))))
        for _ in range(num_corrections):
            pos = self.rng.randint(0, len(current) - 1)
            target_val = reference[pos]
            target_idx = corrected.index(target_val)
            corrected[pos], corrected[target_idx] = corrected[target_idx], corrected[pos]
        return corrected

    # ---------- main loop ----------

    def solve(self) -> Dict:
        start_time = time.time()
        population = [self.generate_initial_solution() for _ in range(self.population_size)]
        for sol in population:
            self.fitness_function(sol)  # prime cache once

        best_solution = None
        best_fitness = float("inf")
        stagnant_iterations = 0
        iteration = -1

        for iteration in range(self.max_iterations):
            population.sort(key=lambda s: s["_fitness"])
            current_best = population[0]
            current_fitness = current_best["_fitness"]

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = copy.deepcopy(current_best)
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
                if stagnant_iterations >= self.convergence_patience:
                    break

            elite_count = max(self.elite_carry_count, int(0.20 * self.population_size))
            reference_pool = population[:elite_count]
            # True elitism: clone elites forward untouched.
            next_population = [copy.deepcopy(s) for s in population[: self.elite_carry_count]]

            while len(next_population) < self.population_size:
                parent = self.rng.choice(population)
                ref = self.rng.choice(reference_pool)
                child = {
                    "train_sequence": self._apply_sequence_corrections(
                        parent["train_sequence"], ref["train_sequence"]
                    ),
                    "allocation": parent["allocation"],
                    "remaining_capacity": parent["remaining_capacity"],
                    "_fitness": None,
                }
                if self.rng.random() < 0.35:
                    child = self.opt_local_search(child)
                else:
                    self.fitness_function(child)
                next_population.append(child)

            population = next_population

        if best_solution is None:
            # max_iterations == 0, or nothing ever improved on inf: fall back
            # to the best member of whatever population exists.
            population.sort(key=lambda s: s.get("_fitness", self.fitness_function(s)))
            best_solution = population[0]
            best_fitness = best_solution["_fitness"]

        execution_duration = time.time() - start_time
        final_arrivals = self.evaluate_schedule(best_solution)

        return {
            "best_solution": best_solution,
            "best_fitness": round(best_fitness, 4),
            "execution_time_seconds": round(execution_duration, 3),
            "iterations_completed": iteration + 1,
            "train_arrivals": final_arrivals,
        }