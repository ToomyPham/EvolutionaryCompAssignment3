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

