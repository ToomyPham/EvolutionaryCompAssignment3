import os, re, glob, math, argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# this is the list of candidate column names for evaluations and best-so-far values
EVAL_CANDS = ["evaluation","evaluations","eval","n_evaluations","evals","x","evaluation_number","#evals"]
BEST_CANDS = ["f_best","best","y_best","best_y","best_f","current_best","y","f0","f1","value","fitness","objective","best_so_far"]

# utility functions for reading and processing .dat files
def tf(s):
    try: return float(s)
    except: return None

# split a line into parts based on common delimiters 
def split_line(line):
    for sep in [",",";","\t"," "]:
        parts = [p for p in line.strip().split(sep) if p!=""]
        if len(parts)>1: return parts
    return [line.strip()]

# read a .dat file, skips blanks lines and comments
def read_table(path):
    lines = []
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f: # reads line by line
            ln = ln.strip()
            if not ln or ln.lstrip().startswith(("#","%")): 
                continue
            lines.append(ln)
    if not lines: # this make sure we have at least one data line
        raise ValueError("no data rows found")
    first = split_line(lines[0])
    has_header = any(tf(tok) is None for tok in first)
    if has_header: # first line is header line
        header = [tok.strip() for tok in first]
        rows = [split_line(ln) for ln in lines[1:]]
        rows = [r for r in rows if len(r)==len(header)]
    else: # else no header line, generate default column names
        k = len(split_line(lines[0]))
        header = [f"c{i}" for i in range(k)]
        rows = [split_line(ln) for ln in lines if len(split_line(ln))==k]
    return header, rows

# this detects which columns to use for evaluations and best-so-far values
# based on candidate names or forced names
def detect_cols(header, force_eval=None, force_best=None):
    he = [h.strip() for h in header]
    eval_col = force_eval if (force_eval and force_eval in he) else None
    best_col = force_best if (force_best and force_best in he) else None
    if not eval_col: # tries to find a suitable eval column
        for c in EVAL_CANDS:
            if c in he: eval_col=c; break
        if eval_col is None and "c0" in he: eval_col="c0"
    if not best_col: # tries to find a suitable best-so-far column
        for c in BEST_CANDS:
            if c in he: best_col=c; break
        if best_col is None and len(he)>1: best_col=he[-1]
    return eval_col,best_col

# extract (x,y) data from the table based on specified columns
def extract_xy(header,rows,eval_col,best_col):
    idx_e = header.index(eval_col)
    idx_b = header.index(best_col)
    xs, ys = [], []
    for r in rows: # process each row to extract (x,y)
        ex, bx = tf(r[idx_e]), tf(r[idx_b])
        if ex is not None and bx is not None:
            xs.append(ex); ys.append(bx)
    if not xs: raise ValueError("no data") # no valid data found
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return xs, ys

# aggregates multiple runs into fixed budget series
def aggregate(runs, budget):
    xs_all = set()
    # collect all unique x values across runs
    for xs, _ in runs: xs_all.update(int(x) for x in xs if 0 <= x <= budget)
    if not xs_all: return [], [], []
    cps = sorted(xs_all)
    if len(cps) > 2000: # downsample to at most 2000 points for efficiency
        step = math.ceil(len(cps) / 2000)
        cps = cps[::step]
    mu, sd = [], []
    for cp in cps: # this computes mean and std at each checkpoint
        vals = []
        for xs, ys in runs:
            idx = None
            for i, x in enumerate(xs):
                if x <= cp: idx = i
                else: break
            if idx is not None: vals.append(ys[idx])
        if vals: # this then computes mean and std
            m = sum(vals) / len(vals)
            v = sum((v - m) ** 2 for v in vals) / len(vals)
            s = v ** 0.5
        else: # no data for this checkpoint
            m = s = float("nan")
        mu.append(m); sd.append(s)
    return cps, mu, sd

# plot fixed-budget results for a given instance and series data
def plot_fixed_budget_for(inst, series_by_algo):
    if not series_by_algo: return # nothing to plot 
    outdir = os.path.join("Exercise_2", "results", "plots", "fixed_budget_plots")
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f"fixed_budget_plot_{inst}.pdf")
    with PdfPages(out_pdf) as pdf: # creates a PDF file for the results of that data set
        plt.figure(figsize=(8, 5))
        for algo, (xs, mu, sd) in series_by_algo.items():
            if not xs: continue
            plt.plot(xs, mu, label=f"{algo} (±1σ)")
            upper = [m + s for m, s in zip(mu, sd)]
            lower = [m - s for m, s in zip(mu, sd)]
            plt.fill_between(xs, lower, upper, alpha=0.2)
        plt.xlabel("Evaluations")
        plt.ylabel("Best objective (mean ± std)")
        plt.title(f"Fixed-Budget — Instance {inst}")
        plt.legend(fontsize=9)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print("Saved PDF:", out_pdf)

# this function tries to guess instance id and algorithm from file path
def guess_instance_algo(path):
    m = re.search(r"(\d{4})", path)
    inst = m.group(1) if m else None
    algo = None
    for p in path.split(os.sep): # check each path component for algo name
        p = p.lower()
        if p in ["rls", "oneplusoneea", "ga", "gsemo", "gsemo_pwt"]: algo = p; break
    return inst, algo

# this function scans the project directory for .dat files and groups data
def scan(project_root, eval_col=None, best_col=None, verbose=False):
    files = glob.glob(os.path.join(project_root, "**", "*.dat"), recursive=True)
    files = [f for f in files if ("run_" in f or "IOHprofiler_" in os.path.basename(f))]
    grouped = {}
    for f in sorted(files): # process each .dat file found
        inst, algo = guess_instance_algo(f)
        if not (inst and algo): continue
        try:
            header, rows = read_table(f)
            ec, bc = detect_cols(header, eval_col, best_col)
            xs, ys = extract_xy(header, rows, ec, bc)
            grouped.setdefault((inst, algo), []).append((xs, ys))
            if verbose: print(f"[ok] {f} -> inst={inst}, algo={algo}, cols=({ec},{bc}), pts={len(xs)}")
        except Exception as e:
            if verbose: print(f"[skip] {f} ({e})")
    return grouped

# this is the main function to generate fixed-budget plots
def generate_fixed_budget_plots(budget):
    grouped = scan(".", None, None, False)
    if not grouped: # no data found
        print("No data found.")
        return
    insts = sorted(set(i for i, _ in grouped))
    algos = sorted(set(a for _, a in grouped))
    for inst in insts: # process each instance
        series = {}
        for algo in algos:
            runs = grouped.get((inst, algo), [])
            if not runs: continue
            xs, mu, sd = aggregate(runs, budget)
            series[algo.upper()] = (xs, mu, sd)
        plot_fixed_budget_for(inst, series)

# main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise 2 — Fixed Budget Plot Generator")
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--plot", action="store_true", help="Generate fixed-budget plots")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing .dat files")
    args = parser.parse_args()
    if args.plot or args.plot_only:
        generate_fixed_budget_plots(args.budget)
    print("Done.")
