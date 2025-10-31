# Exercise_2.py
# Single-file Exercise 2 runner:
# - Implements GSEMO (multi-objective, Week 8)
# - Runs all required instances with IOH logging
# - Saves ALL outputs under Exercise_2/results/
# - No plotting (use IOHanalyzer / your website)
#
# Usage:
#   pip install ioh numpy
#   python Exercise_2.py --runs 30 --budget 10000

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import ioh

PROBLEM_IDS = {
    "coverage":  [2100, 2101, 2102, 2103],  # MaxCoverage
    "influence": [2200, 2201, 2202, 2203],  # MaxInfluence
    "pwt":       [2300, 2301, 2302],        # PackWhileTravel
}

RUNS_DEFAULT = 30
BUDGET_DEFAULT = 10_000
SEED_BASE = 42

# Save all outputs inside Exercise_2/results/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(THIS_DIR, "results")


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """
    Pareto dominance for 'maximize both' objectives.
    a dominates b iff a_i >= b_i for all i and a_j > b_j for some j.
    """
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))


def gsemo(problem, budget: int, rng: np.random.Generator, k: int = None):
    """
    GSEMO for (mono)tone submodular optimisation consistent with Week 8 lecture.

    - If k is not None (MaxCoverage/MaxInfluence): objectives = (z(x), -|S|)
      with z(x) = f(x) if |S| <= k else -1 (infeasible sentinel).
    - If k is None (PWT): objectives = (f(x), -|S|).

    Returns:
      best_x: np.ndarray (binary)
      best_f: (f1, f2) tuple
      pareto: list of (x, (f1,f2)) nondominated points discovered
    """
    n = problem.meta_data.n_variables

    if k is None:
        # trade-off (no uniform bound)
        def obj(x: np.ndarray) -> Tuple[float, float]:
            fx = float(problem(x))
            return (fx, float(-np.sum(x)))
    else:
        # uniform k-constraint
        def obj(x: np.ndarray) -> Tuple[float, float]:
            fx = float(problem(x))
            size = int(np.sum(x))
            z = fx if size <= k else -1.0
            return (z, float(-size))

    # Initialise with empty set + one random solution
    empty = np.zeros(n, dtype=int)
    P: List[Tuple[np.ndarray, Tuple[float, float]]] = [(empty, obj(empty))]

    x0 = rng.integers(0, 2, size=n, dtype=int)
    P.append((x0, obj(x0)))
    evals = 2

    while evals < budget:
        # Uniform parent selection
        pidx = rng.integers(len(P))
        parent = P[pidx][0]

        # 1/n bit mutation, ensure at least one flip
        flips = rng.random(n) < (1.0 / n)
        if not flips.any():
            flips[rng.integers(n)] = True

        child = parent.copy()
        child[flips] ^= 1  # flip bits

        fchild = obj(child)
        evals += 1

        # Pareto update of population P
        keep, dominated_flag = [], False
        for (z, fz) in P:
            if _dominates(fchild, fz):
                # drop dominated point
                continue
            if _dominates(fz, fchild):
                dominated_flag = True
            keep.append((z, fz))
        if not dominated_flag:
            keep.append((child, fchild))
        P = keep

    # Select best by primary objective (first component)
    best = max(P, key=lambda t: t[1][0])
    return best[0], best[1], P


# --------------------------- IOH HELPERS --------------------------------------

def get_problem(problem_id: int):
    """Load problem instance from IOHProfiler GRAPH class."""
    return ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)


def make_logger(algo_name: str, problem_id: int, run_id: int):
    """
    Create IOH Analyzer logger writing under Exercise_2/results/.
    NOTE: do not pass 'suite_name' (not in current API).
    """
    outdir = os.path.join(RESULTS_ROOT, algo_name, str(problem_id), f"run_{run_id}")
    os.makedirs(outdir, exist_ok=True)
    return ioh.logger.Analyzer(
        folder_name=outdir,
        algorithm_name=algo_name,
        algorithm_info="S2_2025_submodular",
        store_positions=False
    )


def attach_logger(problem, logger):
    problem.attach_logger(logger)


def seeded_rng(seed: int):
    return np.random.default_rng(seed)


def dump_pareto(pareto, outpath: str):
    """Save Pareto set (for run 0) as JSON for later plotting/analysis."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data = []
    for x, f in pareto:
        data.append({
            "f1": float(f[0]),
            "f2": float(f[1]),
            "size": int(np.sum(x)),
            "x": x.tolist()
        })
    with open(outpath, "w") as fh:
        json.dump(data, fh, indent=2)


def detect_k(problem) -> int:
    """Infer uniform cardinality k from metadata; fallback to 10."""
    meta = getattr(problem, "meta_data", None)
    info = getattr(meta, "info", {}) or {}
    for key in ("k", "budget", "constraint_k", "cardinality"):
        if key in info:
            try:
                return int(info[key])
            except Exception:
                pass
    return 10


# ------------------------------ RUNNERS ---------------------------------------

def run_group(problem_ids: List[int], runs: int, budget: int, use_k: bool):
    """
    Run GSEMO on a list of problem IDs.
    - use_k=True  -> Coverage & Influence (uniform constraint)
    - use_k=False -> PWT (no k)
    """
    for pid in problem_ids:
        for r in range(runs):
            rng = seeded_rng(SEED_BASE + 2000 * r + pid)
            problem = get_problem(pid)

            algo_tag = "GSEMO" if use_k else "GSEMO_PWT"
            logger = make_logger(algo_tag, pid, r)
            attach_logger(problem, logger)

            try:
                k = detect_k(problem) if use_k else None
                x_best, f_best, pareto = gsemo(problem, budget, rng, k=k)

                # Save Pareto front JSON for first run only (needed for trade-off plots on website)
                if r == 0:
                    out = os.path.join(RESULTS_ROOT, algo_tag, str(pid), f"run_{r}", "pareto.json")
                    dump_pareto(pareto, out)

            finally:
                # Always detach to flush logs
                problem.detach_logger()


def main(runs: int, budget: int):
    # 1) MaxCoverage (uniform k)   2100–2103
    run_group(PROBLEM_IDS["coverage"], runs, budget, use_k=True)
    # 2) MaxInfluence (uniform k)  2200–2203
    run_group(PROBLEM_IDS["influence"], runs, budget, use_k=True)
    # 3) PackWhileTravel (no k)    2300–2302
    run_group(PROBLEM_IDS["pwt"], runs, budget, use_k=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise 2 — GSEMO with IOH logging (results only).")
    parser.add_argument("--runs", type=int, default=RUNS_DEFAULT, help="Number of independent runs per instance (default: 30)")
    parser.add_argument("--budget", type=int, default=BUDGET_DEFAULT, help="Fitness evaluation budget per run (default: 10000)")
    args = parser.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    main(args.runs, args.budget)

    print("\n✅ Exercise 2 complete.")
    print(f"   All logs & pareto.json saved under:\n   {RESULTS_ROOT}\n")
    print("   Upload the 'results' folder to your website / IOHanalyzer for fixed-budget & trade-off plots.\n")
