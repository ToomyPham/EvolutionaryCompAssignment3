# Exercise_2.py
# Single-file Exercise 2:
# - Implements GSEMO (multi-objective, Week 8)
# - Runs all required instances with IOH logging
# - Saves EVERYTHING under Exercise_2/results/
# - Optional plotting of trade-off (first-run) scatter from saved pareto.json
#
# Usage:
#   pip install ioh numpy matplotlib
#   # Run experiments only:
#   python Exercise_2.py --runs 30 --budget 10000
#   # Run + also make trade-off plots for all instances:
#   python Exercise_2.py --runs 30 --budget 10000 --plot
#   # Only plot (assuming results/ already exists from a previous run):
#   python Exercise_2.py --plot-only

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import ioh  # pip install ioh


# ------------------------------ CONFIG ----------------------------------------

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
PLOTS_DIR = os.path.join(RESULTS_ROOT, "plots", "tradeoffs")  # where PNGs go


# ------------------------------ GSEMO -----------------------------------------

def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Pareto dominance for 'maximize both' objectives."""
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))


def gsemo(problem, budget: int, rng: np.random.Generator, k: int = None):
    """
    GSEMO for (mono)tone submodular optimisation (Week 8 formulation).

    - If k is not None (MaxCoverage/MaxInfluence): objectives = (z(x), -|S|)
      with z(x) = f(x) if |S| <= k else -1 (infeasible sentinel).
    - If k is None (PWT): objectives = (f(x), -|S|).

    Returns:
      best_x: np.ndarray (binary solution)
      best_f: (f1, f2)
      pareto: list of (x, (f1,f2)) nondominated points discovered
    """
    n = problem.meta_data.n_variables

    if k is None:
        def obj(x: np.ndarray) -> Tuple[float, float]:
            fx = float(problem(x))
            return (fx, float(-np.sum(x)))
    else:
        def obj(x: np.ndarray) -> Tuple[float, float]:
            fx = float(problem(x))
            size = int(np.sum(x))
            z = fx if size <= k else -1.0
            return (z, float(-size))

    # Initialise population with empty set + random point
    empty = np.zeros(n, dtype=int)
    P: List[Tuple[np.ndarray, Tuple[float, float]]] = [(empty, obj(empty))]

    x0 = rng.integers(0, 2, size=n, dtype=int)
    P.append((x0, obj(x0)))
    evals = 2

    while evals < budget:
        # Uniform parent selection
        pidx = rng.integers(len(P))
        parent = P[pidx][0]

        # 1/n-bit mutation, ensure at least one flip
        flips = rng.random(n) < (1.0 / n)
        if not flips.any():
            flips[rng.integers(n)] = True

        child = parent.copy()
        child[flips] ^= 1

        fchild = obj(child)
        evals += 1

        # Pareto update of P
        keep, dominated_flag = [], False
        for (z, fz) in P:
            if _dominates(fchild, fz):  # child dominates z
                continue
            if _dominates(fz, fchild):  # z dominates child
                dominated_flag = True
            keep.append((z, fz))
        if not dominated_flag:
            keep.append((child, fchild))
        P = keep

    # Best by first objective component
    best = max(P, key=lambda t: t[1][0])
    return best[0], best[1], P


# --------------------------- IOH HELPERS --------------------------------------

def get_problem(problem_id: int):
    """Load GRAPH-class problem instance from IOH."""
    return ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)


def make_logger(algo_name: str, problem_id: int, run_id: int):
    """
    Create IOH Analyzer logger writing under Exercise_2/results/.
    (Do not pass 'suite_name' – not in current API.)
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
                _, _, pareto = gsemo(problem, budget, rng, k=k)

                # Save Pareto front JSON for first run only (used for trade-off plots)
                if r == 0:
                    out = os.path.join(RESULTS_ROOT, algo_tag, str(pid), f"run_{r}", "pareto.json")
                    dump_pareto(pareto, out)

            finally:
                # Always detach to flush logs
                problem.detach_logger()


def run_experiments(runs: int, budget: int):
    # 1) MaxCoverage (uniform k)   2100–2103
    run_group(PROBLEM_IDS["coverage"], runs, budget, use_k=True)
    # 2) MaxInfluence (uniform k)  2200–2203
    run_group(PROBLEM_IDS["influence"], runs, budget, use_k=True)
    # 3) PackWhileTravel (no k)    2300–2302
    run_group(PROBLEM_IDS["pwt"], runs, budget, use_k=False)


# ------------------------------- PLOTTING -------------------------------------

def plot_tradeoff_for(algo_tag: str, problem_id: int, run_index: int = 0):
    """
    Create a trade-off (|S| vs objective value) scatter from pareto.json
    and save under Exercise_2/results/plots/tradeoffs/.
    """
    try:
        import matplotlib.pyplot as plt  # local import so script works without it if not plotting
    except Exception as e:
        print("⚠️ matplotlib is not installed. Install with: pip install matplotlib")
        return

    src = os.path.join(RESULTS_ROOT, algo_tag, str(problem_id), f"run_{run_index}", "pareto.json")
    if not os.path.exists(src):
        print(f"❌ Missing pareto.json: {src} (run experiments first)")
        return

    with open(src, "r") as fh:
        pts = json.load(fh)

    xs = [-p["f2"] for p in pts]  # |S|
    ys = [ p["f1"] for p in pts]  # objective value

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_png = os.path.join(PLOTS_DIR, f"{algo_tag}_{problem_id}_run{run_index}.png")

    plt.figure()
    plt.scatter(xs, ys, s=25)
    plt.xlabel("|S|")
    plt.ylabel("Objective value")
    plt.title(f"Trade-off (first run) – {algo_tag} – {problem_id}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"✅ Saved trade-off plot: {out_png}")


def plot_all_tradeoffs():
    """
    Generate trade-off plots (first run) for all required instances, if pareto.json exists.
    """
    for pid in PROBLEM_IDS["coverage"]:
        plot_tradeoff_for("GSEMO", pid, 0)
    for pid in PROBLEM_IDS["influence"]:
        plot_tradeoff_for("GSEMO", pid, 0)
    for pid in PROBLEM_IDS["pwt"]:
        plot_tradeoff_for("GSEMO_PWT", pid, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise 2 — GSEMO with IOH logging (results + optional plots).")
    parser.add_argument("--runs", type=int, default=RUNS_DEFAULT, help="Runs per instance (default: 30)")
    parser.add_argument("--budget", type=int, default=BUDGET_DEFAULT, help="Evaluations per run (default: 10000)")
    parser.add_argument("--plot", action="store_true", help="After running, also create trade-off plots for all instances")
    parser.add_argument("--plot-only", action="store_true", help="Only generate trade-off plots from existing results/")
    args = parser.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if not args.plot_only:
        run_experiments(args.runs, args.budget)
        print(f"\n✅ Experiments complete. All logs & pareto.json saved under:\n   {RESULTS_ROOT}\n")

    if args.plot or args.plot_only:
        plot_all_tradeoffs()
        print(f"\n📊 Plots (if generated) saved under:\n   {PLOTS_DIR}\n")

    print("Done.")
