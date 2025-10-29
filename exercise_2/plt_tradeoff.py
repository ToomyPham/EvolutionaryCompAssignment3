import argparse, json, os
import matplotlib.pyplot as plt

def plot_from_pareto(json_path, out_png):
    with open(json_path, "r") as fh:
        pts = json.load(fh)
    # We stored objectives as (f1, f2). For both modes we maximized f2 where f2 = -|S|.
    xs = [ -p["f2"] for p in pts ]  # |S|
    ys = [  p["f1"] for p in pts ]  # objective value
    plt.figure()
    plt.scatter(xs, ys, s=25)
    plt.xlabel("|S|")
    plt.ylabel("Objective value")
    plt.title("Trade-off (first run)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["GSEMO","GSEMO_PWT"], required=True)
    ap.add_argument("--problem_id", type=int, required=True)
    ap.add_argument("--run_index", type=int, default=0)
    args = ap.parse_args()
    src = f"ioh_data/{args.algo}/{args.problem_id}/run_{args.run_index}/pareto.json"
    out = f"final/doc/ex2/tradeoffs/{args.algo}_{args.problem_id}_run{args.run_index}.png"
    plot_from_pareto(src, out)
    print(f"Saved {out}")
