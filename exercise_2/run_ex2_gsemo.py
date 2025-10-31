# exercise_2/run_ex2_gsemo.py
import argparse, json, os
import numpy as np
import ioh  # pip install ioh

from gsemo import gsemo  # same-folder import

# ----------------- CONFIG -----------------
PROBLEM_IDS = {
    "coverage":  [2100, 2101, 2102, 2103],
    "influence": [2200, 2201, 2202, 2203],
    "pwt":       [2300, 2301, 2302],
}
RUNS = 30
BUDGET_DEFAULT = 10_000
SEED_BASE = 42

# All output will be saved here ⬇️
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "ioh_data")

# ----------------- IOH HELPERS -----------------
def get_problem(problem_id: int):
    """Load an IOH submodular optimisation problem."""
    return ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)

def make_logger(algo_name: str, problem_id: int, run_id: int):
    """
    Create an IOH Analyzer logger that writes to exercise_2/ioh_data/.
    (No 'suite_name' argument — this matches IOH's latest API.)
    """
    outdir = os.path.join(OUTPUT_ROOT, algo_name, str(problem_id), f"run_{run_id}")
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

# ----------------- EXERCISE 2 HELPERS -----------------
def _dump_pareto(pareto, outpath: str):
    """Save Pareto front as JSON for trade-off plotting."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data = []
    for x, f in pareto:
        data.append({
            "f1": float(f[0]),      # first objective (coverage/influence/PWT value)
            "f2": float(f[1]),      # second objective (-|S|)
            "size": int(np.sum(x)), # |S|
            "x": x.tolist()
        })
    with open(outpath, "w") as fh:
        json.dump(data, fh, indent=2)

def _detect_k(problem) -> int:
    """Try to infer k (uniform constraint) from IOH metadata; fallback to 10."""
    meta = getattr(problem, "meta_data", None)
    info = getattr(meta, "info", {}) or {}
    for key in ("k", "budget", "constraint_k", "cardinality"):
        if key in info:
            try:
                return int(info[key])
            except Exception:
                pass
    return 10

def _run_group(ids, runs: int, budget: int, use_k: bool):
    """Run GSEMO across all problem IDs within a group (coverage/influence/pwt)."""
    for pid in ids:
        for r in range(runs):
            rng = seeded_rng(SEED_BASE + 2000 * r + pid)
            problem = get_problem(pid)
            algo_tag = "GSEMO" if use_k else "GSEMO_PWT"
            logger = make_logger(algo_tag, pid, r)
            attach_logger(problem, logger)
            try:
                k = _detect_k(problem) if use_k else None
                x_best, f_best, P = gsemo(problem, budget, rng, k=k)
                if r == 0:  # save Pareto front for first run
                    out = os.path.join(
                        OUTPUT_ROOT, algo_tag, str(pid), f"run_{r}", "pareto.json"
                    )
                    _dump_pareto(P, out)
            finally:
                problem.detach_logger()

def main(runs: int, budget: int):
    """Run Exercise 2 experiments."""
    # 1️⃣ MaxCoverage and MaxInfluence (use uniform constraint k)
    _run_group(PROBLEM_IDS["coverage"], runs, budget, use_k=True)
    _run_group(PROBLEM_IDS["influence"], runs, budget, use_k=True)
    # 2️⃣ PackWhileTravel (no constraint — trade-off mode)
    _run_group(PROBLEM_IDS["pwt"], runs, budget, use_k=False)

# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    args = ap.parse_args()

    main(args.runs, args.budget)
    print(f"\n✅ All results saved under: {OUTPUT_ROOT}\n")
