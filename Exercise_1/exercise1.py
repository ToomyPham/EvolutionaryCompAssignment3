from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from ioh import get_problem, ProblemClass, logger

def graph_family(pid: int) -> str:
    if 2100 <= pid <= 2103: return "MaxCoverage"
    if 2200 <= pid <= 2203: return "MaxInfluence"
    if 2300 <= pid <= 2302: return "PackWhileTravel"
    return "Other"

def make_logger(root: str, folder: str, algo_name: str, algo_info: str = ""):
    return logger.Analyzer(
        root=root,
        folder_name=folder,         # <-- nested path lives here
        algorithm_name=algo_name,
        algorithm_info=algo_info,   # optional; shows in IOHanalyzer
        store_positions=False,
    )

def robust_attach(problem, L) -> None:
    try:
        L.attach_problem(problem)   # newer API
    except Exception:
        problem.attach_logger(L)    # older API

def robust_detach(problem, L) -> None:
    try: problem.detach_logger()
    except Exception: pass
    try: L.close()
    except Exception: pass

PROBLEM_IDS = [
    2100, 2101, 2102, 2103,
    2200, 2201, 2202, 2203,
    2300, 2301, 2302,
]

def budget_reached(problem, budget: int) -> bool:
    return problem.state.evaluations >= budget

def random_bitstring(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.int8)

def rls_bitflip(bitstring: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = bitstring.copy()
    i = rng.integers(0, y.size)
    y[i] ^= 1
    return y

def one_plus_one_bitflip(bitstring: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    mask = rng.random(bitstring.size) < p
    if not mask.any():
        idx = rng.integers(0, bitstring.size)
        mask[idx] = True
    y = bitstring.copy()
    y[mask] ^= 1
    return y

def tournament_select(pop, fits, rng: np.random.Generator, k: int = 2) -> np.ndarray:
    i, j = rng.integers(0, len(pop), size=2)
    return pop[i] if fits[i] >= fits[j] else pop[j]

def rls(problem, budget: int, rng: np.random.Generator) -> None:
    n = problem.meta_data.n_variables
    x = random_bitstring(n, rng)
    fx = problem(x)
    while not budget_reached(problem, budget):
        y = rls_bitflip(x, rng)
        fy = problem(y)
        if fy >= fx:
            x, fx = y, fy

def one_plus_one_ea(problem, budget: int, rng: np.random.Generator) -> None:
    n = problem.meta_data.n_variables
    p = 1.0/n
    x = random_bitstring(n, rng)
    fx = problem(x)
    while not budget_reached(problem, budget):
        y = one_plus_one_bitflip(x, rng, p)
        fy = problem(y)
        if fy >= fx:
            x, fx = y, fy

def ga(problem, budget: int, rng: np.random.Generator) -> None:
    n = problem.meta_data.n_variables
    pop_size = 50
    cx_rate = 0.9
    mut_p = 1.0/n
    elitism = 1

    population = rng.integers(0, 2, size=(pop_size, n), dtype=np.int8)
    fitness = np.array([problem(ind) for ind in population], dtype=float)

    while not budget_reached(problem, budget):
        parents = np.stack([tournament_select(population, fitness, rng, k=2) for _ in range(pop_size)], axis = 0)
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[min(i + 1, pop_size - 1)]
            if rng.random() < cx_rate:
                mask = rng.integers(0, 2, size=n, dtype=np.int8)
                c1 = (p1 & mask) | (p2 & (1 - mask))
                c2 = (p2 & mask) | (p1 & (1 - mask))
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(c1); offspring.append(c2)
        offspring = np.stack(offspring[:pop_size], axis = 0).astype(np.int8)

        mut_mask = rng.random(size=offspring.shape) < mut_p
        offspring[mut_mask] ^= 1

        new_fitness = np.array([problem(ind) for ind in offspring], dtype = float)

        if elitism > 0:
            elite_idx = np.argsort(fitness)[-elitism:]
            worst_idx = np.argsort(new_fitness)[:elitism]
            offspring[worst_idx] = population[elite_idx]
            new_fitness[worst_idx] = fitness[elite_idx]
        
        population, fitness = offspring, new_fitness

@dataclass
class AlgoSpec:
    name: str
    fn: Callable



def run_all(root: str, runs: int, budget: int, seed: int | None):
    rng_master = np.random.default_rng(seed)

    algos: List[AlgoSpec] = [
        AlgoSpec("RLS", rls),
        AlgoSpec("(1+1)EA", one_plus_one_ea),
        AlgoSpec("GA", ga),
    ]

    for pid in PROBLEM_IDS:
        # create the problem ONCE per pid (reuse across runs via reset)
        problem = get_problem(pid, problem_class=ProblemClass.GRAPH)
        fam = graph_family(pid)

        for algo in algos:
            # create ONE logger per (pid, algo)
            folder = f"{algo.name}/{fam}/F{pid}"
            algo_info = f"{algo.name} on {fam} F{pid}"
            L = make_logger(root, folder, algo.name, algo_info)

            try:
                L.set_experiment_attributes({
                    "algorithm": str(algo.name),
                    "family": fam,
                    "instance": str(pid),
                })
            except Exception:
                pass

            robust_attach(problem, L)

            # multiple runs into the SAME IOH dataset
            for r in range(1, runs + 1):
                run_seed = int(rng_master.integers(0, 2**63 - 1))
                rng = np.random.default_rng(run_seed)

                algo.fn(problem, budget=budget, rng=rng)
                problem.reset()

            robust_detach(problem, L)

        del problem


def main(root="results/submodular", runs=30, budget=10000, seed=42):
    run_all(root=root,runs=runs,budget=budget,seed=seed)

if __name__ == "__main__":
    main()
