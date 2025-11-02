import argparse
import json
import os
from typing import List, Tuple
import numpy as np
import ioh

# problem ids for each submodular problem type
PROBLEM_IDS = {
    "coverage":  [2100, 2101, 2102, 2103],
    "influence": [2200, 2201, 2202, 2203],
    "pwt":       [2300, 2301, 2302],
}

# Default parameters for runs & budget
RUNS_DEFAULT = 30
BUDGET_DEFAULT = 10_000
SEED_BASE = 42

# Directories for results and plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(THIS_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_ROOT, "plots", "tradeoffs")

# dominance check for 2D objectives 
def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))

# this is the GSEMO implementation from lectures
# with minor adjustments for our bi-objective setting
def gsemo(problem, budget: int, rng: np.random.Generator, k: int = None):
    n = problem.meta_data.n_variables

    # bi-objective (f(x), -|x|)
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

    empty = np.zeros(n, dtype = int)
    P: List[Tuple[np.ndarray, Tuple[float, float]]] = [(empty, obj(empty))]

    x0 = rng.integers(0, 2, size = n, dtype = int)
    P.append((x0, obj(x0)))
    evals = 2

    # this is the main loop of GSEMO
    while evals < budget:
        pidx = rng.integers(len(P))
        parent = P[pidx][0]

        flips = rng.random(n) < (1.0 / n)
        if not flips.any(): # this ensures at least one bit flip
            flips[rng.integers(n)] = True

        child = parent.copy()
        child[flips] ^= 1

        fchild = obj(child)
        evals += 1

        keep, dominated_flag = [], False
        for (z, fz) in P: # these lines of code maintain the Pareto set
            if _dominates(fchild, fz):# this checks if fchild dominates fz
                continue
            if _dominates(fz, fchild):# this checks if fz dominates fchild 
                dominated_flag = True
            keep.append((z, fz))
        if not dominated_flag: # only add child if its not dominated
            keep.append((child, fchild))
        P = keep

    best = max(P, key = lambda t: t[1][0])
    return best[0], best[1], P

# helper function to load a problem instance
def get_problem(problem_id: int):
    return ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)

# these lines of code set up logging with IOH Analyser
def make_logger(algo_name: str, problem_id: int, run_id: int):
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

# helper function to dump Pareto front to JSON
def dump_pareto(pareto, outpath: str):
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

# these lines of code are the helper function to detect uniform cardinality k from problem metadata
def detect_k(problem) -> int:
    meta = getattr(problem, "meta_data", None)
    info = getattr(meta, "info", {}) or {}
    for key in ("k", "budget", "constraint_k", "cardinality"):
        if key in info:
            try:
                return int(info[key])
            except Exception:
                pass
    return 10

# these lines of code run groups of experiments
def run_group(problem_ids: List[int], runs: int, budget: int, use_k: bool):
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

                if r == 0: # if r == 0, it saves the Pareto front to JSON
                    out = os.path.join(RESULTS_ROOT, algo_tag, str(pid), f"run_{r}", "pareto.json")
                    dump_pareto(pareto, out)

            finally:
                problem.detach_logger()

# these lines of code run all experiments for Exercise 2 
def run_experiments(runs: int, budget: int):
    run_group(PROBLEM_IDS["coverage"], runs, budget, use_k=True)
    run_group(PROBLEM_IDS["influence"], runs, budget, use_k=True)
    run_group(PROBLEM_IDS["pwt"], runs, budget, use_k=False)

# These plots the trade-off scatter from saved pareto.json files and saves as PNG
def plot_tradeoff_for(algo_tag: str, problem_id: int, run_index: int = 0):

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is not installed. Install with: pip install matplotlib")
        return

    src = os.path.join(RESULTS_ROOT, algo_tag, str(problem_id), f"run_{run_index}", "pareto.json")
    if not os.path.exists(src):
        print(f"Missing pareto.json: {src} (run experiments first)")
        return

    with open(src, "r") as fh:
        pts = json.load(fh)

    xs = [-p["f2"] for p in pts] 
    ys = [ p["f1"] for p in pts] 

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_png = os.path.join(PLOTS_DIR, f"{algo_tag}_{problem_id}_run{run_index}.png")

    plt.figure()
    plt.scatter(xs, ys, s=25)
    plt.xlabel("|S|")
    plt.ylabel("Objective value")
    plt.title(f"Trade-off – {algo_tag} – {problem_id}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# these lines of code plot trade-offs for all instances
def plot_all_tradeoffs():

    for pid in PROBLEM_IDS["coverage"]:
        plot_tradeoff_for("GSEMO", pid, 0)
    for pid in PROBLEM_IDS["influence"]:
        plot_tradeoff_for("GSEMO", pid, 0)
    for pid in PROBLEM_IDS["pwt"]:
        plot_tradeoff_for("GSEMO_PWT", pid, 0)

# main function to parse arguments and run experiments/plots 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise 2 — GSEMO with IOH logging (results + optional plots).")
    parser.add_argument("--runs", type=int, default=RUNS_DEFAULT, help="Runs per instance (default: 30)")
    parser.add_argument("--budget", type=int, default=BUDGET_DEFAULT, help="Evaluations per run (default: 10000)")
    parser.add_argument("--plot", action="store_true", help="After running, also create trade-off plots for all instances")
    parser.add_argument("--plot-only", action="store_true", help="Only generate trade-off plots from existing results/")
    args = parser.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if not args.plot_only: # if not plot_only, then run experiments
        run_experiments(args.runs, args.budget)

    if args.plot or args.plot_only: # if plot or plot_only, then generate plots
        plot_all_tradeoffs()

    print("Done.")
