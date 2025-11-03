import os
import numpy as np
import matplotlib.pyplot as plt

def load_dat_file(filepath):
    """Load evaluations and raw_y values from an IOHprofiler .dat file."""
    runs = []
    current_evals, current_fitness = [], []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("evaluations") or line.strip() == "":
                if current_evals and current_fitness:
                    runs.append((np.array(current_evals), np.array(current_fitness)))
                    current_evals, current_fitness = [], []
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                current_evals.append(float(parts[0]))
                current_fitness.append(float(parts[1]))
    if current_evals and current_fitness:
        runs.append((np.array(current_evals), np.array(current_fitness)))
    return runs


def compute_mean_std(runs, num_points=1000):
    """Interpolate best-so-far runs to common x-axis and compute mean/std."""
    if not runs:
        return None, None, None

    # Use full range across all runs
    min_eval = min(run[0][0] for run in runs)
    max_eval = max(run[0][-1] for run in runs)
    x_common = np.linspace(min_eval, max_eval, num_points)

    interp_runs = []
    for evals, fitness in runs:
        # ensure best-so-far (monotone non-decreasing) fitness
        fitness = np.maximum.accumulate(fitness)
        interp = np.interp(x_common, evals, fitness, left=fitness[0], right=fitness[-1])
        interp_runs.append(interp)

    interp_runs = np.array(interp_runs)
    mean_fitness = np.mean(interp_runs, axis=0)
    std_fitness = np.std(interp_runs, axis=0)
    return x_common, mean_fitness, std_fitness


def plot_results(outdir="results", problems=[2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]):
    methods = ["(1+1)EA/MaxCoverage", "(1+1)EA/MaxInfluence", "GA/MaxCoverage", "GA/MaxInfluence", "RLS/MaxCoverage", "RLS/MaxInfluence"]
    log_scale_problems = []

    for method in methods:
        method_path = os.path.join(outdir, method)
        if not os.path.exists(method_path):
            print(f"Missing folder: {method_path}")
            continue

        print(f"\n📊 Plotting results for {method}")

        for pid in problems:
            plt.figure(figsize=(8, 6))

            # find folder
            folder = None
            for name in os.listdir(method_path):
                if name.startswith(f"data_f{pid}_"):
                    folder = os.path.join(method_path, name)
                    break
            if folder is None:
                print(f"No data folder for F{pid} in {method_path}")
                plt.close()
                continue

            # find .dat file
            dat_file = None
            for name in os.listdir(folder):
                if name.endswith(".dat"):
                    dat_file = os.path.join(folder, name)
                    break
            if dat_file is None:
                print(f"No .dat file for F{pid} in {folder}")
                plt.close()
                continue

            runs = load_dat_file(dat_file)
            if not runs:
                print(f"Empty data in {dat_file}")
                plt.close()
                continue

            # Plot individual runs faintly
            for evals, fitness in runs:
                fitness = np.maximum.accumulate(fitness)
                plt.step(evals, fitness, where='post', linewidth=1, alpha=0.4, color="gray")

            # Compute mean ± std and plot as step function
            x_common, mean_fitness, std_fitness = compute_mean_std(runs)
            if x_common is not None:
                plt.step(x_common, mean_fitness, where='post', color='blue', linewidth=2, label="Mean fitness")
                plt.fill_between(x_common,
                                 mean_fitness - std_fitness,
                                 mean_fitness + std_fitness,
                                 step='post', color='blue', alpha=0.2, label="±1 Std. Dev.")

            plt.title(f"{method} — Convergence on F{pid}")
            plt.xlabel("Evaluations")
            plt.ylabel("Best Fitness")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

            # Apply log scale if needed
            if pid in log_scale_problems:
                plt.xscale('log')

            plt.tight_layout()
            outpath = os.path.join(outdir, f"{method}_F{pid}.pdf")
            plt.savefig(outpath, dpi=200)
            plt.close()
            print(f"✅ Saved plot for {method} F{pid} -> {outpath}")



if __name__ == "__main__":
    plot_results()
