import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
RESULTS_DIR = "results"
OUTPUT_DIR = "final/doc/ex3_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

# Load all CSVs
def load_all_results():
    all_data = []
    for root, _, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(root, f))
                    df["source_file"] = f
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {f}: {e}")
    if not all_data:
        raise RuntimeError("❌ No CSV files found in results/.")
    return pd.concat(all_data, ignore_index=True)

data = load_all_results()
data.columns = [c.strip().lower() for c in data.columns]

# Rename columns to consistent names
colmap = {
    "instance": "instance",
    "alg": "algorithm",
    "pop": "population",
    "best": "fitness",
    "run": "run",
}
data = data.rename(columns={k: v for k, v in colmap.items() if k in data.columns})

# Drop rows without fitness
data = data.dropna(subset=["fitness"])

# Compute mean ± std per instance and algorithm
grouped = (
    data.groupby(["instance", "algorithm", "population"])["fitness"]
    .agg(["mean", "std"])
    .reset_index()
)

# === PLOT EACH INSTANCE ===
for instance in sorted(grouped["instance"].unique()):
    df_inst = grouped[grouped["instance"] == instance]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_inst,
        x="algorithm",
        y="mean",
        hue="population",
        ci=None,
        capsize=0.1,
        palette="tab10",
    )

    # Add error bars manually (std)
    for i, row in enumerate(df_inst.itertuples()):
        plt.errorbar(
            i,
            row.mean,
            yerr=row.std,
            fmt="none",
            c="black",
            capsize=5,
        )

    plt.title(f"Fixed-Budget Performance – Instance {instance}")
    plt.ylabel("Mean Best Fitness ± SD (30 runs)")
    plt.xlabel("Algorithm")
    plt.legend(title="Population Size")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"fixed_budget_{instance}.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved {out_path}")

print("\nAll fixed-budget plots saved in:", OUTPUT_DIR)
