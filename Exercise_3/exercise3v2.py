#!/usr/bin/env python3
"""
exercise3.py

Population-based single-objective and multi-objective evolutionary algorithms
for monotone submodular optimisation under a uniform constraint.

This script:
 - Implements PopulationEA (single-objective) and PopulationGSEMO (multi-objective)
 - Runs experiments across instances, populations, diversity/selection modes
 - Writes CSV summary results (as before)
 - ALSO writes IOH-style .dat and .json files for each run, using this layout:
     results_ioh/<instance>/<algorithm>/<population>/<run>/data_f0.dat
     results_ioh/<instance>/<algorithm>/<population>/<run>/data_f0.json

Logging policy:
 - Logs every evaluation (best-so-far) — ideal for fixed-budget progress plots in IOHanalyzer.

Usage:
    python exercise3.py --run-all
    python exercise3.py --alg both --pop 20 --runs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

# -----------------------------
# Utilities: bitstring helpers
# -----------------------------
def random_bitstring(n: int, p: float = 0.5) -> List[int]:
    return [1 if random.random() < p else 0 for _ in range(n)]

def hamming_distance(a: List[int], b: List[int]) -> int:
    return sum(x != y for x, y in zip(a, b))

def bitstring_to_set(x: List[int]) -> set:
    return {i for i, xi in enumerate(x) if xi}

def set_to_bitstring(s: set, n: int) -> List[int]:
    b = [0] * n
    for i in s:
        if 0 <= i < n:
            b[i] = 1
    return b

def copy_bitstring(x: List[int]) -> List[int]:
    return list(x)

def mutate_plus(x: List[int], p_flip: Optional[float] = None) -> List[int]:
    """Standard-bit-mutation-plus: repeat until offspring != parent."""
    n = len(x)
    if p_flip is None:
        p_flip = 1.0 / max(1, n)
    while True:
        y = x.copy()
        for i in range(n):
            if random.random() < p_flip:
                y[i] = 1 - y[i]
        if y != x:
            return y

# -----------------------------
# IOH-style logger for runs
# -----------------------------
class IOHLogger:
    """
    Simple IOH-style logger that writes:
      - data_f0.dat  (evaluation_number best_so_far_value)
      - data_f0.json (metadata)
    for a single run.

    Directory layout:
      base_dir/<instance>/<algorithm>/<population>/<run>/
            data_f0.dat
            data_f0.json
    """
    def __init__(self, base_dir: str, instance: int, algorithm: str, population: int, run_id: int,
                 n: int, budget: int, seed: Optional[int], selection: Optional[str], extra_meta: Optional[Dict] = None):
        self.base_dir = base_dir
        self.instance = instance
        self.algorithm = algorithm
        self.population = population
        self.run_id = run_id
        self.n = n
        self.budget = budget
        self.seed = seed
        self.selection = selection
        self.extra_meta = extra_meta or {}

        # create dir
        self.run_dir = os.path.join(self.base_dir, str(self.instance), self.algorithm, f"pop_{self.population}", f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        # file paths
        self.dat_path = os.path.join(self.run_dir, "data_f0.dat")
        self.json_path = os.path.join(self.run_dir, "data_f0.json")

        # open dat file and write header
        self._dat_file = open(self.dat_path, "w")
        # IOH data files often start without a formal header; write a simple informative header as comment
        # then columns: evaluations best
        self._dat_file.write("# evaluation best\n")
        self._dat_file.flush()

        # state tracking
        self.best_so_far: Optional[float] = None
        self.lines_written = 0
        self.start_time = time.time()

    def log_evaluation(self, evaluation: int, value: float) -> None:
        """
        Record an evaluation. We keep best_so_far and write a line with evaluation and best_so_far.
        We log every evaluation (EVAL logging).
        """
        if self.best_so_far is None or value > self.best_so_far:
            self.best_so_far = float(value)
        # write line: evaluation best_so_far
        self._dat_file.write(f"{evaluation} {self.best_so_far}\n")
        self.lines_written += 1
        # flush occasionally
        if self.lines_written % 256 == 0:
            self._dat_file.flush()

    def close(self, final_metadata_custom: Optional[Dict[str, Any]] = None) -> None:
        # finalize files
        self._dat_file.flush()
        self._dat_file.close()

        # create JSON metadata (FULL)
        meta = {
            "algorithm_name": self.algorithm,
            "instance": self.instance,
            "population": self.population,
            "run_id": self.run_id,
            "dimension": self.n,
            "budget": self.budget,
            "seed": self.seed,
            "selection": self.selection,
            "lines_written": self.lines_written,
            "best_so_far": self.best_so_far,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }
        # merge extras
        meta.update(self.extra_meta or {})
        if final_metadata_custom:
            meta.update(final_metadata_custom)

        # write json
        with open(self.json_path, "w") as jf:
            json.dump(meta, jf, indent=2)

# -----------------------------
# Problem wrapper (IOH or fallback)
# -----------------------------
class SubmodularIOHProblem:
    """
    Wrapper around IOH problem or a fallback synthetic 'coverage'-like problem.

    The wrapper supports an optional evaluation callback: functions registered via
    register_callback(callback) will be called with arguments (eval_count, value)
    each time evaluate(x) is called. This allows logging every evaluation.
    """
    def __init__(self, instance_id: Optional[int] = None, use_ioh: bool = False):
        self.instance_id = instance_id
        self.use_ioh = use_ioh
        self.eval_count = 0
        self._callbacks: List[Callable[[int, float], None]] = []

        # Try to use ioh if requested; otherwise create synthetic problem
        if self.use_ioh:
            try:
                import ioh  # type: ignore
                # attempt to get problem (safe-guard)
                try:
                    self._prob = ioh.get_problem(instance_id, problem_class=ioh.ProblemClass.GRAPH)
                    # extract number of variables
                    if hasattr(self._prob, "number_of_variables"):
                        self.n = int(self._prob.number_of_variables)
                    elif hasattr(self._prob, "meta_data") and hasattr(self._prob.meta_data, "n_variables"):
                        self.n = int(self._prob.meta_data.n_variables)
                    elif hasattr(self._prob, "n"):
                        self.n = int(self._prob.n)
                    else:
                        self.n = 100
                except Exception:
                    # fallback to synthetic if IOH fetching fails
                    self.use_ioh = False
            except Exception:
                self.use_ioh = False

        if not self.use_ioh:
            # synthetic fallback
            if instance_id is None:
                self.n = 50
            else:
                self.n = max(40, (instance_id % 100) + 40)
            # seed deterministic for instance
            random.seed(instance_id or 0)
            universe_size = max(60, self.n * 3)
            self.universe = list(range(universe_size))
            self.incidence = [
                set(random.sample(self.universe, random.randint(1, max(1, universe_size // 10))))
                for _ in range(self.n)
            ]
        # cost function: uniform cost = sum bits
        self.cost_fn = lambda x: int(sum(x))

    def register_callback(self, cb: Callable[[int, float], None]) -> None:
        self._callbacks.append(cb)

    def evaluate(self, x: List[int]) -> float:
        """Evaluate x and call callbacks with (eval_count, value)."""
        self.eval_count += 1
        if self.use_ioh and hasattr(self, "_prob") and self._prob is not None:
            try:
                val = float(self._prob(x))
            except Exception:
                # fallback to synthetic compute if IOH call fails
                val = self._coverage_value(x)
        else:
            val = self._coverage_value(x)

        # call callbacks
        for cb in self._callbacks:
            try:
                cb(self.eval_count, float(val))
            except Exception:
                # ensure robustness of experiment loop
                pass

        return float(val)

    def _coverage_value(self, x: List[int]) -> float:
        covered = set()
        for i, xi in enumerate(x):
            if xi:
                covered |= self.incidence[i]
        return float(len(covered))

    def reset_evalcount(self) -> None:
        self.eval_count = 0

# -----------------------------
# PopulationEA (single-objective)
# -----------------------------
class PopulationEA:
    def __init__(self, problem: SubmodularIOHProblem, budget: int = 10000, mu: int = 20,
                 diversity: str = "random_replacement", repair: str = "greedy_remove",
                 seed: Optional[int] = None, ioh_logger: Optional[IOHLogger] = None):
        self.problem = problem
        self.budget = budget
        self.mu = mu
        self.diversity = diversity
        self.repair = repair
        self.seed = seed
        self.ioh_logger = ioh_logger
        if seed is not None:
            random.seed(seed)
        self.n = problem.n
        self.pop: List[Tuple[List[int], float, int, bool]] = []
        self.eval_history: List[Tuple[int, float]] = []
        self.sharing_sigma = max(1, self.n // 10)
        self.sharing_alpha = 1.0

        # Hook logging callback to problem if logger passed
        if self.ioh_logger is not None:
            # define callback that updates best and logs best-so-far:
            def cb(eval_count: int, value: float):
                # update current population state best if necessary:
                # we will maintain best_so_far in the logger itself (it compares value)
                self.ioh_logger.log_evaluation(eval_count, value)
            self.problem.register_callback(cb)

    def initial_population(self) -> None:
        self.pop = []
        for _ in range(self.mu):
            x = random_bitstring(self.n, p=0.1)
            x, fitness, cost, feasible = self.evaluate_with_repair(x)
            self.pop.append((x, fitness, cost, feasible))

    def evaluate_with_repair(self, x: List[int]) -> Tuple[List[int], float, int, bool]:
        cost = self.problem.cost_fn(x)
        B = getattr(self.problem, "budget", None)
        feasible = True if (B is None) else (cost <= B)
        fitness = self.problem.evaluate(x)
        if B is None:
            return x, fitness, cost, True
        if cost <= B:
            return x, fitness, cost, True

        if self.repair == "simple_trim":
            y = x.copy()
            ones = [i for i, xi in enumerate(y) if xi]
            random.shuffle(ones)
            while sum(y) > B and ones:
                idx = ones.pop()
                y[idx] = 0
            cost = sum(y)
            fitness = self.problem.evaluate(y)
            return y, fitness, cost, True
        elif self.repair == "greedy_remove":
            y = x.copy()
            while sum(y) > B:
                base = self.problem.evaluate(y)
                marginals = []
                for i in range(self.n):
                    if y[i]:
                        y2 = y.copy()
                        y2[i] = 0
                        val = self.problem.evaluate(y2)
                        marginals.append((base - val, i))
                if not marginals:
                    break
                marginals.sort(key=lambda t: (t[0], t[1]))
                _, idx = marginals[0]
                y[idx] = 0
            fitness = self.problem.evaluate(y)
            cost = sum(y)
            return y, fitness, cost, True
        else:
            return x, -1e9, cost, False

    def fitness_sharing_adjusted(self) -> List[float]:
        sigma = self.sharing_sigma
        alpha = self.sharing_alpha
        adjusted = []
        for i, (xi, fi, ci, feas) in enumerate(self.pop):
            denom = 0.0
            for j, (xj, fj, cj, feasj) in enumerate(self.pop):
                d = hamming_distance(xi, xj)
                if d < sigma:
                    denom += (1 - (d / sigma) ** alpha)
            if denom <= 0.0:
                denom = 1.0
            adjusted.append(fi / denom)
        return adjusted

    def select_parents(self) -> List[List[int]]:
        parents: List[List[int]] = []
        if self.diversity == "random_replacement":
            for _ in range(self.mu):
                parents.append(random.choice(self.pop)[0])
        elif self.diversity == "tournament":
            k = min(3, max(2, self.mu // 4))
            for _ in range(self.mu):
                candidates = random.sample(self.pop, k)
                candidates.sort(key=lambda t: t[1], reverse=True)
                parents.append(candidates[0][0])
        elif self.diversity == "fitness_sharing":
            adjusted = self.fitness_sharing_adjusted()
            min_adj = min(adjusted)
            if min_adj <= 0:
                adjusted = [a - min_adj + 1e-6 for a in adjusted]
            total = sum(adjusted)
            if total <= 0:
                for _ in range(self.mu):
                    parents.append(random.choice(self.pop)[0])
            else:
                probs = [a / total for a in adjusted]
                for _ in range(self.mu):
                    if np is not None:
                        idx = int(np.random.choice(range(len(self.pop)), p=probs))
                    else:
                        r = random.random()
                        cum = 0.0
                        idx = 0
                        for i, p in enumerate(probs):
                            cum += p
                            if r <= cum:
                                idx = i
                                break
                    parents.append(self.pop[idx][0])
        else:
            for _ in range(self.mu):
                parents.append(random.choice(self.pop)[0])
        return parents

    def replacement(self, offspring: List[Tuple[List[int], float, int, bool]]) -> None:
        combined = self.pop + offspring
        combined.sort(key=lambda t: (1 if t[3] else 0, t[1]), reverse=True)
        self.pop = combined[: self.mu]

    def best_feasible(self) -> Optional[Tuple[List[int], float, int, bool]]:
        feas = [ind for ind in self.pop if ind[3]]
        if not feas:
            return None
        return max(feas, key=lambda t: t[1])

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        if not hasattr(self.problem, "budget"):
            raise ValueError("Problem must have attribute 'budget' (uniform constraint B).")

        self.initial_population()
        best = self.best_feasible()
        best_val = best[1] if best is not None else -math.inf
        self.eval_history = [(self.problem.eval_count, best_val)]

        while self.problem.eval_count < self.budget:
            parents = self.select_parents()
            offspring = []
            for p in parents:
                child = mutate_plus(p)
                child, fitness, cost, feas = self.evaluate_with_repair(child)
                offspring.append((child, fitness, cost, feas))
                if self.problem.eval_count >= self.budget:
                    break
            self.replacement(offspring)
            best = self.best_feasible()
            best_val = best[1] if best is not None else -math.inf
            self.eval_history.append((self.problem.eval_count, best_val))
            if verbose and (self.problem.eval_count % (max(1, self.budget // 10)) == 0):
                print(f"evals {self.problem.eval_count}, best_feasible {best_val}")

        best = self.best_feasible()
        return {
            "best": best,
            "eval_history": self.eval_history,
            "final_pop": self.pop,
            "n_evals": self.problem.eval_count,
        }

# -----------------------------
# PopulationGSEMO (multi-objective population-based)
# -----------------------------
def dominates(a_obj: Tuple[float, ...], b_obj: Tuple[float, ...]) -> bool:
    assert len(a_obj) == len(b_obj)
    ge = all(a >= b for a, b in zip(a_obj, b_obj))
    gt = any(a > b for a, b in zip(a_obj, b_obj))
    return ge and gt

def crowding_distance(archive: List[Tuple[List[int], Tuple[float, ...], dict]]) -> List[float]:
    m = len(archive)
    if m == 0:
        return []
    k = len(archive[0][1])
    distances = [0.0] * m
    for obj_i in range(k):
        sorted_idx = sorted(range(m), key=lambda i: archive[i][1][obj_i])
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")
        obj_vals = [archive[i][1][obj_i] for i in sorted_idx]
        minv = obj_vals[0]
        maxv = obj_vals[-1]
        if maxv == minv:
            continue
        for idx in range(1, m - 1):
            i = sorted_idx[idx]
            distances[i] += (archive[sorted_idx[idx + 1]][1][obj_i] - archive[sorted_idx[idx - 1]][1][obj_i]) / (maxv - minv)
    return distances

class PopulationGSEMO:
    def __init__(self, problem: SubmodularIOHProblem, budget: int = 10000, mu: int = 20,
                 selection: str = "uniform", seed: Optional[int] = None, ioh_logger: Optional[IOHLogger] = None):
        self.problem = problem
        self.budget = budget
        self.mu = mu
        self.selection = selection
        self.seed = seed
        self.ioh_logger = ioh_logger
        if seed is not None:
            random.seed(seed)
        self.n = problem.n
        self.archive: List[Tuple[List[int], Tuple[float, float], dict]] = []
        self.eval_history: List[Tuple[int, float]] = []
        self.problem.cost_fn = getattr(self.problem, "cost_fn", lambda x: sum(x))

        # Hook logger
        if self.ioh_logger is not None:
            def cb(eval_count: int, value: float):
                # For GSEMO we log the primary objective (value) as the single-objective f0 trace
                self.ioh_logger.log_evaluation(eval_count, value)
            self.problem.register_callback(cb)

    def initial_archive(self) -> None:
        x0 = [0] * self.n
        f0 = self.problem.evaluate(x0)
        c0 = self.problem.cost_fn(x0)
        self.archive = [(x0, (f0, -float(c0)), {"f": f0, "c": c0})]

    def select_parent(self, t: int) -> List[int]:
        if len(self.archive) == 0:
            return random_bitstring(self.n, p=0.1)
        if self.selection == "uniform":
            return random.choice(self.archive)[0]
        elif self.selection == "crowding":
            distances = crowding_distance(self.archive)
            d = [(1e6 if math.isinf(v) else v) for v in distances]
            total = sum(d)
            if total <= 0:
                return random.choice(self.archive)[0]
            probs = [v / total for v in d]
            if np is not None:
                idx = int(np.random.choice(range(len(self.archive)), p=probs))
            else:
                r = random.random()
                cum = 0.0
                idx = 0
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        idx = i
                        break
            return self.archive[idx][0]
        else:
            return random.choice(self.archive)[0]

    def update_archive(self, cand: Tuple[List[int], Tuple[float, float], dict]) -> bool:
        x, obj, meta = cand
        new_archive: List[Tuple[List[int], Tuple[float, float], dict]] = []
        dominated_flag = False
        for a in self.archive:
            if dominates(obj, a[1]):
                continue
            if dominates(a[1], obj):
                dominated_flag = True
                break
            new_archive.append(a)
        if dominated_flag:
            self.archive = new_archive
            return False
        new_archive.append((x, obj, meta))
        if len(new_archive) > self.mu:
            distances = crowding_distance(new_archive)
            idxs = list(range(len(new_archive)))
            idxs.sort(key=lambda i: (distances[i], sum(new_archive[i][1])), reverse=True)
            kept_idxs = set(idxs[: self.mu])
            new_archive = [new_archive[i] for i in range(len(new_archive)) if i in kept_idxs]
        self.archive = new_archive
        return True

    def best_feasible(self) -> Optional[Tuple[List[int], Tuple[float, float], dict]]:
        B = getattr(self.problem, "budget", None)
        feas = []
        for x, obj, meta in self.archive:
            c = meta["c"]
            if (B is None) or (c is None) or (c <= B):
                feas.append((x, obj, meta))
        if not feas:
            return None
        return max(feas, key=lambda t: t[1][0])

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        if not hasattr(self.problem, "budget"):
            raise ValueError("Problem must have attribute 'budget' (uniform constraint B).")
        self.initial_archive()
        self.eval_history = []
        self.eval_history.append((self.problem.eval_count, self.best_feasible()[1][0] if self.best_feasible() else -math.inf))
        while self.problem.eval_count < self.budget:
            parent = self.select_parent(self.problem.eval_count)
            child = mutate_plus(parent)
            f = self.problem.evaluate(child)
            c = self.problem.cost_fn(child)
            obj = (f, -float(c))
            meta = {"f": f, "c": c}
            self.update_archive((child, obj, meta))
            self.eval_history.append((self.problem.eval_count, self.best_feasible()[1][0] if self.best_feasible() else -math.inf))
            if verbose and (self.problem.eval_count % (max(1, self.budget // 10)) == 0):
                print(f"evals {self.problem.eval_count}, best feasible f {self.best_feasible()[1][0] if self.best_feasible() else -math.inf}")
        return {"archive": self.archive, "eval_history": self.eval_history, "n_evals": self.problem.eval_count}

# -----------------------------
# Experiment orchestration with IOH logging (Option D)
# -----------------------------
def _write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if pd is not None:
        try:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            return
        except Exception:
            pass
    # fallback
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            safe_row = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (list, tuple, dict)):
                    try:
                        safe_row[k] = json.dumps(v)
                    except Exception:
                        safe_row[k] = str(v)
                else:
                    safe_row[k] = v
            writer.writerow(safe_row)

def run_experiment_on_instance(instance_id: int,
                               alg: str = "both",
                               pop_sizes: List[int] = [10, 20, 50],
                               diversity_methods: List[str] = ["random_replacement", "fitness_sharing", "tournament"],
                               runs: int = 30,
                               budget: int = 10000,
                               results_dir: str = "results",
                               ioh_dir: str = "results_ioh") -> List[Dict[str, Any]]:
    """
    Run experiments for one instance. Write:
      - CSV summary to results_dir/instance_<id>_{alg}_results.csv
      - IOH trace files to results_ioh/<instance>/<algorithm>/pop_<pop>/run_<run>/data_f0.{dat,json}
    """

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ioh_dir, exist_ok=True)
    single_rows: List[Dict[str, Any]] = []
    multi_rows: List[Dict[str, Any]] = []

    for pop in pop_sizes:
        for diversity in diversity_methods:
            for run_idx in range(runs):
                # Create fresh problem
                problem = SubmodularIOHProblem(instance_id=instance_id, use_ioh=False)
                # budget uniform constraint
                problem.budget = min(problem.n, max(1, int(problem.n // 10)))
                problem.reset_evalcount()

                seed = 1000 + run_idx + pop * 13 + (hash(diversity) % 1000)
                # SINGLE
                if alg in ("single", "both"):
                    # create IOH logger for this run
                    ioh_logger_single = IOHLogger(base_dir=ioh_dir, instance=instance_id,
                                                  algorithm="single", population=pop,
                                                  run_id=run_idx, n=problem.n,
                                                  budget=budget, seed=seed, selection=diversity,
                                                  extra_meta={"variant": "populationEA"})
                    # reset eval count and register logger
                    problem.reset_evalcount()
                    random.seed(seed)
                    ea = PopulationEA(problem, budget=budget, mu=pop, diversity=diversity,
                                      repair="greedy_remove", seed=seed, ioh_logger=ioh_logger_single)
                    start = time.time()
                    res = ea.run(verbose=False)
                    elapsed = time.time() - start
                    best = res.get("best", None)
                    best_val = best[1] if best else None
                    # close logger and write metadata
                    ioh_logger_single.close(final_metadata_custom={
                        "elapsed_seconds": elapsed,
                        "algorithm_type": "PopulationEA",
                        "run_summary_best": best_val
                    })
                    single_rows.append({
                        "instance": instance_id,
                        "alg": "single",
                        "pop": pop,
                        "diversity": diversity,
                        "run": run_idx,
                        "best": best_val,
                        "n_evals": res.get("n_evals", problem.eval_count),
                        "time": elapsed
                    })

                # MULTI
                if alg in ("multi", "both"):
                    problem.reset_evalcount()
                    random.seed(seed)
                    ioh_logger_multi = IOHLogger(base_dir=ioh_dir, instance=instance_id,
                                                 algorithm="multi", population=pop,
                                                 run_id=run_idx, n=problem.n,
                                                 budget=budget, seed=seed, selection="uniform",
                                                 extra_meta={"variant": "PopulationGSEMO"})
                    ga = PopulationGSEMO(problem, budget=budget, mu=pop, selection="uniform",
                                         seed=seed, ioh_logger=ioh_logger_multi)
                    start = time.time()
                    res = ga.run(verbose=False)
                    elapsed = time.time() - start
                    best = ga.best_feasible()
                    best_val = best[1][0] if best else None
                    ioh_logger_multi.close(final_metadata_custom={
                        "elapsed_seconds": elapsed,
                        "algorithm_type": "PopulationGSEMO",
                        "run_summary_best": best_val
                    })
                    multi_rows.append({
                        "instance": instance_id,
                        "alg": "multi",
                        "pop": pop,
                        "selection": "uniform",
                        "run": run_idx,
                        "best": best_val,
                        "n_evals": res.get("n_evals", problem.eval_count),
                        "time": elapsed
                    })

                # flush partial CSVs periodically
                if (run_idx + 1) % 10 == 0:
                    # write CSV partials
                    if single_rows:
                        csv_path = os.path.join(results_dir, f"instance_{instance_id}_single_results.csv")
                        _write_csv_rows(csv_path, list(single_rows[0].keys()), single_rows)
                    if multi_rows:
                        csv_path = os.path.join(results_dir, f"instance_{instance_id}_multi_results.csv")
                        _write_csv_rows(csv_path, list(multi_rows[0].keys()), multi_rows)
                    print(f"[inst {instance_id}] pop {pop} diversity {diversity} run {run_idx} flushed to disk.")

    # final save
    if single_rows:
        csv_path = os.path.join(results_dir, f"instance_{instance_id}_single_results.csv")
        _write_csv_rows(csv_path, list(single_rows[0].keys()), single_rows)
        print("Saved single CSV:", csv_path)
    if multi_rows:
        csv_path = os.path.join(results_dir, f"instance_{instance_id}_multi_results.csv")
        _write_csv_rows(csv_path, list(multi_rows[0].keys()), multi_rows)
        print("Saved multi CSV:", csv_path)

    return single_rows + multi_rows

# -----------------------------
# CLI and top-level run
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run default full experiment set")
    parser.add_argument("--alg", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (use 30 for full experiment)")
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ioh-dir", type=str, default="results_ioh")
    args = parser.parse_args()

    insts = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]
    if args.instance is not None:
        insts = [args.instance]

    if args.run_all:
        runs = 30
        pop_sizes = [10, 20, 50]
        diversity_methods = ["random_replacement", "fitness_sharing", "tournament"]
        for inst in insts:
            print("Running instance", inst)
            run_experiment_on_instance(inst, alg=args.alg, pop_sizes=pop_sizes,
                                       diversity_methods=diversity_methods, runs=runs,
                                       budget=args.budget, results_dir=args.results_dir, ioh_dir=args.ioh_dir)
    else:
        inst = insts[0] if args.instance is None else args.instance
        pop_sizes = [args.pop]
        diversity_methods = ["random_replacement"]
        run_experiment_on_instance(inst, alg=args.alg, pop_sizes=pop_sizes,
                                   diversity_methods=diversity_methods, runs=args.runs,
                                   budget=args.budget, results_dir=args.results_dir, ioh_dir=args.ioh_dir)

if __name__ == "__main__":
    main()
