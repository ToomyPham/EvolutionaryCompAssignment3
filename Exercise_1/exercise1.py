from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from ioh import get_problem, ProblemClass, logger

PROBLEM_IDS = [
    2100, 2101, 2102, 2103,
    2200, 2201, 2202, 2203,
    2300, 2301, 2302,
]

def budget_reached(problem, budget: int) -> bool:
    return problem.state.evaluation >= budget

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
    n = problem.meta_data.m_variables
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

def attach_logger(problem, root: str, algo_name: str, instance_id: int, run_idx: int):

    log = logger.Analyzer(
        root = root,
        folder_name = f"{algo_name}",
        algorithm_name = algo_name,
        store_position = False,
    )
    log.set_experiment_attributes({
        "instance": instance_id,
        "run": run_idx,
        "algorithm": algo_name,
    })
    problem.attach_logger(log)
    return log

def run_all(root: str, runs: int, budget: int, seed: int | None):
    rng_master = np.random.default_rng(seed)

    algos: List[AlgoSpec] = [
        AlgoSpec("RLS", rls),
        AlgoSpec("(1+1)EA", one_plus_one_ea),
        AlgoSpec("GA", ga),
    ]

    for pid in PROBLEM_IDS:
        for algo in algos:
            for r in range(1, runs + 1):
                run_seed = int(rng_master.integers(0, 2**63 - 1))
                rng = np.random.default_rng(run_seed)

                problem = get_problem(pid, problem_class = ProblemClass.GRAPH)

                log = attach_logger(problem, root = root, algo_name = algo.name, instance_id = pid, run_idx = r)
                
                algo.fn(problem, budget = budget, rng = rng)

                problem.detach_logger()
                del log
                del problem

def main(root="results/submodular", runs=30, budget=10000, seed=42):
    run_all(root=root,runs=runs,budget=budget,seed=seed)

if __name__ == "__main__":
    main()
