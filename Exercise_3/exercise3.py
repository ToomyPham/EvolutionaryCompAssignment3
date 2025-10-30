"""
exercise3_ea.py

Population-based single-objective and multi-objective evolutionary algorithms
for monotone submodular optimization under a uniform constraint.

Designed for COMP SCI 3316/7316 Assignment 3 Exercise 3.
Implements:
 - PopulationEA (single-objective)
 - PopulationGSEMO (multi-objective population-based variant)
 - Experiment driver that hooks into IOHprofiler problems if available

Usage:
    python exercise3_ea.py --run-all
    python exercise3_ea.py --alg single --pop 10
"""

import argparse
import math
import os
import random
import time
import json
from collections import defaultdict, namedtuple
from copy import deepcopy

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    np = None
    pd = None
    plt = None

# Try to import IOHprofiler (ioh). If not found, use a fallback stub problem provider.
try:
    import ioh
    FROM_IOH = True
except Exception:
    ioh = None
    FROM_IOH = False

# -----------------------------
# Utilities: bitstring helpers
# -----------------------------
def random_bitstring(n, p=0.5):
    return [1 if random.random() < p else 0 for _ in range(n)]

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def bitstring_to_set(x):
    return {i for i, xi in enumerate(x) if xi}

def set_to_bitstring(s, n):
    b = [0]*n
    for i in s:
        b[i] = 1
    return b

def copy_bitstring(x):
    return list(x)

# Standard-bit-mutation-plus: repeat until offspring != parent
def mutate_plus(x, p_flip= None):
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
# Problem wrapper (IOH or fallback)
# -----------------------------
class SubmodularIOHProblem:
    """
    Wrapper around IOH problem or a fallback synthetic problem when IOH isn't installed.

    Expected IOH usage in the assignment: ioh.get_problem(instance_id, problem_class=ioh.ProblemClass.GRAPH)
    Instances:
      MaxCoverage: ids 2100..2103
      MaxInfluence: ids 2200..2203

    This wrapper expects the IOH problem to expose:
      - number_of_variables (n)
      - evaluate(x) -> objective value (float)
      - (we implement cost function c(x) ourselves as uniform: sum(bits))
    """

    def __init__(self, instance_id=None, budget=None, use_ioh=FROM_IOH):
        self.instance_id = instance_id
        self.budget = budget
        self.use_ioh = use_ioh and (ioh is not None)
        self.eval_count = 0
        if self.use_ioh:
            # try to fetch problem via IOH
            # using problem_class=ioh.ProblemClass.GRAPH as described in assignment prompt
            # Some environments might need different import; assume standard IOH.
            try:
                self._prob = ioh.get_problem(instance_id, problem_class=ioh.ProblemClass.GRAPH)
                # Try different possible locations for number of variables
                if hasattr(self._prob, "number_of_variables"):
                    self.n = self._prob.number_of_variables
                elif hasattr(self._prob, "meta_data") and hasattr(self._prob.meta_data, "n_variables"):
                    self.n = self._prob.meta_data.n_variables
                else:
                    # last resort: assume 100 and print warning
                    print("Warning: could not determine number of variables, defaulting to 100")
                    self.n = 100
            except Exception as e:
                print(f"Error while fetching problem: {e}")
            except Exception as e:
                print("Warning: IOH import happened but couldn't get problem:", e)
                self.use_ioh = False
            if not self.use_ioh:
                # fallback synthetic submodular-like problem (coverage-like)
                # We'll create n and a ground-sets incidence so coverage is submodular.
                self.n = 50 if instance_id is None else max(40, instance_id % 100 + 40)
                random.seed(instance_id or 0)
            # Create universe of elements and for each variable choose a subset
            universe_size = max(60, self.n * 3)
            self.universe = list(range(universe_size))
            # For each variable, pick a random subset of universe
            self.incidence = [set(random.sample(self.universe, random.randint(1, max(1, universe_size//10))))
                              for _ in range(self.n)]
            # A quirky "influence" variant available by instance_id group
            self.is_influence = False
        # uniform cost
        self.cost_fn = lambda x: sum(x)

    def evaluate(self, x):
        """Return objective value f(x). Tracks eval_count."""
        self.eval_count += 1
        if self.use_ioh:
            # IOH expects a list of 0/1 or floats. We'll pass binary list.
            return self._prob(x)
        # fallback: coverage-like: number of covered universe elements
        covered = set()
        for i, xi in enumerate(x):
            if xi:
                covered |= self.incidence[i]
        return float(len(covered))

    def reset_evalcount(self):
        self.eval_count = 0

# -----------------------------
# Population-based Single-Objective EA
# -----------------------------
class PopulationEA:
    def __init__(self, problem, budget=10000, mu=20, diversity='random_replacement',
                 repair='greedy_remove', seed=None):
        self.problem = problem
        self.budget = budget
        self.mu = mu
        self.diversity = diversity
        self.repair = repair
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.n = problem.n
        self.pop = []  # list of (bitstring, fitness, cost, feasible)
        self.eval_history = []  # (eval_count, best_feasible)
        # helper for fitness sharing
        self.sharing_sigma = max(1, self.n // 10)
        self.sharing_alpha = 1.0

    def initial_population(self):
        self.pop = []
        for _ in range(self.mu):
            x = random_bitstring(self.n, p=0.1)  # start sparse for uniform constraint
            x, fitness, cost, feasible = self.evaluate_with_repair(x)
            self.pop.append((x, fitness, cost, feasible))

    def evaluate_with_repair(self, x):
        """Evaluate x; if infeasible cost>B then apply repair strategy if configured."""
        cost = self.problem.cost_fn(x)
        feasible = (cost <= self.problem.budget) if hasattr(self.problem, 'budget') else (cost <= 0 or True)
        fitness = self.problem.evaluate(x)
        # the assignment expects uniform constraint; ensure we have a budget B
        B = getattr(self.problem, 'budget', None)
        if B is None:
            # fallback: accept all
            return x, fitness, cost, True
        if cost <= B:
            return x, fitness, cost, True
        # infeasible -> repair or penalty
        if self.repair == 'simple_trim':
            # remove random ones until cost <= B
            y = x.copy()
            ones = [i for i, xi in enumerate(y) if xi]
            random.shuffle(ones)
            while sum(y) > B and ones:
                y[ones.pop()] = 0
            cost = sum(y)
            fitness = self.problem.evaluate(y)
            return y, fitness, cost, True
        elif self.repair == 'greedy_remove':
            # remove elements with smallest marginal contribution until feasible
            y = x.copy()
            # compute marginal contributions (approx) by f(x)-f(x\{i})
            # do this iteratively: remove smallest contribution each time
            while sum(y) > B:
                marginals = []
                base = self.problem.evaluate(y)
                for i in range(self.n):
                    if y[i]:
                        y2 = y.copy()
                        y2[i] = 0
                        val = self.problem.evaluate(y2)
                        marginals.append((base - val, i))
                # remove the item with smallest marginal (least harm to objective)
                if not marginals:
                    break
                marginals.sort()  # smallest loss first
                _, idx = marginals[0]
                y[idx] = 0
            fitness = self.problem.evaluate(y)
            cost = sum(y)
            return y, fitness, cost, True
        else:
            # penalty approach: keep x but mark as infeasible and set fitness = -inf
            return x, -1e9, cost, False

    def fitness_sharing_adjusted(self):
        """Adjust fitness in-place according to sharing scheme (returns list of adjusted fitness)."""
        sigma = self.sharing_sigma
        alpha = self.sharing_alpha
        base_fitness = [ind[1] for ind in self.pop]
        adjusted = []
        for i, (xi, fi, ci, feas) in enumerate(self.pop):
            denom = 0.0
            for j, (xj, fj, cj, feasj) in enumerate(self.pop):
                d = hamming_distance(xi, xj)
                if d < sigma:
                    denom += (1 - (d / sigma) ** alpha)
            if denom <= 0: denom = 1.0
            adjusted.append(fi / denom)
        return adjusted

    def select_parents(self):
        """Return list of parents for generating mu offspring."""
        parents = []
        if self.diversity == 'random_replacement':
            for _ in range(self.mu):
                parents.append(random.choice(self.pop)[0])
        elif self.diversity == 'tournament':
            k = min(3, max(2, self.mu // 4))
            for _ in range(self.mu):
                candidates = random.sample(self.pop, k)
                candidates.sort(key=lambda t: t[1], reverse=True)  # by fitness
                parents.append(candidates[0][0])
        elif self.diversity == 'fitness_sharing':
            adjusted = self.fitness_sharing_adjusted()
            # sample proportional to adjusted fitness (shift if negative)
            min_adj = min(adjusted)
            if min_adj <= 0:
                adjusted = [a - min_adj + 1e-6 for a in adjusted]
            total = sum(adjusted)
            probs = [a / total for a in adjusted]
            for _ in range(self.mu):
                idx = np.random.choice(range(len(self.pop)), p=probs)
                parents.append(self.pop[idx][0])
        else:
            # fallback to random
            for _ in range(self.mu):
                parents.append(random.choice(self.pop)[0])
        return parents

    def replacement(self, offspring):
        """Combine current population and offspring then select mu survivors according to replacement policy."""
        combined = self.pop + offspring  # each entry: (x,f,c,feas)
        # sort by fitness descending (only feasible prioritized)
        combined.sort(key=lambda t: (1 if t[3] else 0, t[1]), reverse=True)
        # different strategies: random_replacement already handled via parent selection; here we just truncate
        self.pop = combined[:self.mu]

    def best_feasible(self):
        feas = [ind for ind in self.pop if ind[3]]
        if not feas:
            return None
        return max(feas, key=lambda t: t[1])

    def run(self, verbose=False):
        """Run the EA until budget exhausted. Returns record dict."""
        # ensure problem has budget B (uniform constraint). We'll expect problem.budget set externally.
        if not hasattr(self.problem, 'budget'):
            raise ValueError("Problem must have attribute 'budget' (uniform constraint B).")

        self.initial_population()
        evals = self.problem.eval_count
        if verbose:
            print(f"Initial population evaluated; evals: {self.problem.eval_count}")
        # record initial best
        best = self.best_feasible()
        best_val = best[1] if best is not None else -math.inf
        self.eval_history.append((self.problem.eval_count, best_val))
        # generation loop: produce mu offspring each iter or until budget
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
            if verbose and (self.problem.eval_count % (self.budget//10 + 1) == 0):
                print(f"evals {self.problem.eval_count}, best_feasible {best_val}")
        # return summary
        best = self.best_feasible()
        return {
            'best': best,
            'eval_history': self.eval_history,
            'final_pop': self.pop,
            'n_evals': self.problem.eval_count
        }

# -----------------------------
# Population-based Multi-objective EA (GSEMO-ish)
# -----------------------------
def dominates(a_obj, b_obj):
    """Return True if a dominates b (maximize both objectives).
       a_obj, b_obj are tuples of objective values to be maximized.
    """
    assert len(a_obj) == len(b_obj)
    ge = all(a >= b for a, b in zip(a_obj, b_obj))
    gt = any(a > b for a, b in zip(a_obj, b_obj))
    return ge and gt

def nondominated_front(archive):
    """Return nondominated subset from archive (list of (x, obj_tuple, extra...))."""
    nond = []
    for i, ai in enumerate(archive):
        ai_obj = ai[1]
        is_dom = False
        for j, aj in enumerate(archive):
            if i == j: continue
            if dominates(aj[1], ai_obj):
                is_dom = True
                break
        if not is_dom:
            nond.append(ai)
    return nond

def crowding_distance(archive):
    """Compute crowding distance on archive of tuples (x, obj_tuple). Returns list of distances in same order."""
    m = len(archive)
    if m == 0:
        return []
    k = len(archive[0][1])
    distances = [0.0]*m
    for obj_i in range(k):
        sorted_idx = sorted(range(m), key=lambda i: archive[i][1][obj_i])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        obj_vals = [archive[i][1][obj_i] for i in sorted_idx]
        minv = obj_vals[0]
        maxv = obj_vals[-1]
        if maxv == minv:
            continue
        for idx in range(1, m-1):
            i = sorted_idx[idx]
            distances[i] += (archive[sorted_idx[idx+1]][1][obj_i] - archive[sorted_idx[idx-1]][1][obj_i])/(maxv-minv)
    return distances

class PopulationGSEMO:
    def __init__(self, problem, budget=10000, mu=20, selection='uniform', seed=None):
        """
        population-based multi-objective EA inspired by GSEMO.
        We will optimize objectives: (f(x), -c(x)) to turn minimization of cost into maximization domain.
        """
        self.problem = problem
        self.budget = budget
        self.mu = mu
        self.selection = selection
        self.seed = seed
        if seed is not None: random.seed(seed)
        self.n = problem.n
        self.archive = []  # list of (x, (f, -c), fitness_info)
        self.eval_history = []  # (evals, best_feasible_f)
        self.problem.cost_fn = getattr(self.problem, 'cost_fn', lambda x: sum(x))

    def initial_archive(self):
        # As in GSEMO start with 0^n
        x0 = [0]*self.n
        f0 = self.problem.evaluate(x0)
        c0 = self.problem.cost_fn(x0)
        self.archive = [(x0, (f0, -c0), {'f':f0, 'c':c0})]

    def select_parent(self, t):
        """Selection from archive using selection method."""
        if self.selection == 'uniform':
            return random.choice(self.archive)[0]
        elif self.selection == 'crowding':
            # compute crowding distances and sample proportional
            distances = crowding_distance(self.archive)
            # convert inf to large finite
            d = [ (1e6 if math.isinf(v) else v) for v in distances]
            total = sum(d)
            if total <= 0:
                return random.choice(self.archive)[0]
            probs = [v/total for v in d]
            idx = np.random.choice(range(len(self.archive)), p=probs)
            return self.archive[idx][0]
        else:
            return random.choice(self.archive)[0]

    def update_archive(self, cand):
        """Try to insert candidate (x,f,-c). Keep archive trimmed to nondominated front with at most mu by crowding."""
        x, obj, meta = cand
        # remove those dominated by cand
        new_archive = []
        dominated_flag = False
        for a in self.archive:
            if dominates(obj, a[1]):
                # candidate dominates a -> drop a
                continue
            if dominates(a[1], obj):
                # archive member dominates candidate -> candidate rejected
                dominated_flag = True
                break
            new_archive.append(a)
        if dominated_flag:
            self.archive = new_archive
            return False
        # candidate is nondominated -> add
        new_archive.append((x, obj, meta))
        # optionally trim by crowding distance if > mu
        if len(new_archive) > self.mu:
            # compute crowding distances and keep best mu
            distances = crowding_distance(new_archive)
            # pair them and sort by distance descending
            idxs = list(range(len(new_archive)))
            idxs.sort(key=lambda i: (distances[i],) , reverse=True)
            kept_idxs = set(idxs[:self.mu])
            new_archive = [new_archive[i] for i in range(len(new_archive)) if i in kept_idxs]
        self.archive = new_archive
        return True

    def best_feasible(self):
        """Return entry with max f among those with cost <= B."""
        B = getattr(self.problem, 'budget', None)
        feas = []
        for x, obj, meta in self.archive:
            c = meta['c']
            if (B is None) or (c <= B):
                feas.append((x, obj, meta))
        if not feas:
            return None
        return max(feas, key=lambda t: t[1][0])  # maximize f

    def run(self, verbose=False):
        if not hasattr(self.problem, 'budget'):
            raise ValueError("Problem must have attribute 'budget' (uniform constraint B).")
        self.initial_archive()
        self.eval_history.append((self.problem.eval_count, self.best_feasible()[1][0] if self.best_feasible() else -math.inf))
        while self.problem.eval_count < self.budget:
            parent = self.select_parent(self.problem.eval_count)
            child = mutate_plus(parent)
            f = self.problem.evaluate(child)
            c = self.problem.cost_fn(child)
            # objective vector: maximize f, maximize -c (so lower cost preferred)
            obj = (f, -c)
            meta = {'f': f, 'c': c}
            self.update_archive((child, obj, meta))
            self.eval_history.append((self.problem.eval_count, self.best_feasible()[1][0] if self.best_feasible() else -math.inf))
            if verbose and (self.problem.eval_count % (self.budget//10 + 1) == 0):
                print(f"evals {self.problem.eval_count}, best feasible f {self.best_feasible()[1][0] if self.best_feasible() else -math.inf}")
        return {
            'archive': self.archive,
            'eval_history': self.eval_history,
            'n_evals': self.problem.eval_count
        }

# -----------------------------
# Experiment orchestration
# -----------------------------
def run_experiment_on_instance(instance_id, alg='single', pop_sizes=[10,20,50],
                               diversity_methods=['random_replacement','fitness_sharing','tournament'],
                               runs=30, budget=10000, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    instance_results = []
    # Build problem wrapper
    for pop in pop_sizes:
        for diversity in diversity_methods:
            for run_idx in range(runs):
                # create problem
                problem = SubmodularIOHProblem(instance_id=instance_id, budget=None, use_ioh=FROM_IOH)
                # set the budget attribute for uniform constraint
                problem.budget = getattr(problem, 'budget', None) or max(1, int(math.sqrt(problem.n)))  # default heuristic if none
                # For assignment we must set budget such that uniform constraint used; user should modify as needed
                problem.budget = min(problem.n, int(max(1, problem.n//10)))  # default B = n/10
                problem.reset_evalcount()

                seed = 1000 + run_idx
                if alg == 'single':
                    ea = PopulationEA(problem, budget=budget, mu=pop, diversity=diversity, repair='greedy_remove', seed=seed)
                    start = time.time()
                    res = ea.run(verbose=False)
                    elapsed = time.time() - start
                    best = res['best']
                    best_val = best[1] if best else None
                    instance_results.append({
                        'instance': instance_id, 'alg':'single', 'pop':pop, 'diversity':diversity,
                        'run':run_idx, 'best': best_val, 'n_evals':res['n_evals'], 'time':elapsed
                    })
                elif alg == 'multi':
                    problem.reset_evalcount()
                    ga = PopulationGSEMO(problem, budget=budget, mu=pop, selection='uniform', seed=seed)
                    start = time.time()
                    res = ga.run(verbose=False)
                    elapsed = time.time() - start
                    best = ga.best_feasible()
                    best_val = best[1][0] if best else None
                    instance_results.append({
                        'instance': instance_id, 'alg':'multi', 'pop':pop, 'selection':'uniform',
                        'run':run_idx, 'best': best_val, 'n_evals':res['n_evals'], 'time':elapsed
                    })
                else:
                    raise ValueError("Unknown alg")
                # quick flush
                if run_idx % 10 == 0:
                    print(f"instance {instance_id}, alg {alg}, pop {pop}, run {run_idx}: best {instance_results[-1]['best']}")
    # Save to CSV
    df = pd.DataFrame(instance_results) if pd is not None else None
    csv_path = os.path.join(output_dir, f"instance_{instance_id}_{alg}_results.csv")
    if df is not None:
        df.to_csv(csv_path, index=False)
    print("Saved results to", csv_path)
    return instance_results

# -----------------------------
# CLI and top-level experiment
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true', help="Run default full experiment (small subset for demo)")
    parser.add_argument('--alg', choices=['single','multi'], default='single')
    parser.add_argument('--pop', type=int, default=20)
    parser.add_argument('--runs', type=int, default=3, help="Number of runs (use 30 for full experiments)")
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--instance', type=int, default=None)
    args = parser.parse_args()

    # instances as required by assignment: 2100..2103 and 2200..2203
    insts = [2100,2101,2102,2103,2200,2201,2202,2203]
    if args.instance is not None:
        insts = [args.instance]

    if args.run_all:
        runs = 30
        pop_sizes = [10,20,50]
        diversity_methods = ['random_replacement','fitness_sharing','tournament']
        for inst in insts:
            print("Running instance", inst)
            run_experiment_on_instance(inst, alg=args.alg, pop_sizes=pop_sizes,
                                       diversity_methods=diversity_methods, runs=runs, budget=args.budget)
    else:
        # quick demo on single instance with provided pop and runs
        inst = insts[0] if args.instance is None else args.instance
        if args.alg == 'single':
            run_experiment_on_instance(inst, alg='single', pop_sizes=[args.pop], diversity_methods=['random_replacement'], runs=args.runs, budget=args.budget)
        else:
            run_experiment_on_instance(inst, alg='multi', pop_sizes=[args.pop], diversity_methods=['random_replacement'], runs=args.runs, budget=args.budget)

if __name__ == '__main__':
    # Always run full experiment for all required instances
    runs = 30
    pop_sizes = [10, 20, 50]
    diversity_methods = ['random_replacement', 'fitness_sharing', 'tournament']
    insts = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]

    for inst in insts:
        print(f"\n=== Running instance {inst} (30 runs × 3 pop sizes × 3 diversity modes) ===")
        run_experiment_on_instance(
            instance_id=inst,
            alg='single',
            pop_sizes=pop_sizes,
            diversity_methods=diversity_methods,
            runs=runs,
            budget=10000
        )
    print("\nAll experiments completed. Results are saved in the 'results/' directory.")
