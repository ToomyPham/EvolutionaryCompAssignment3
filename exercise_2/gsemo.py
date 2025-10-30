import numpy as np

def _dominates(a, b):
    # Pareto dominance for "maximize both" objectives
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))

def gsemo(problem, budget, rng, k=None):
    """
    GSEMO for monotone submodular optimisation.
    - If k is not None (uniform constraint): maximise ( z(x), -|x| ) where
      z(x) = f(x) if |x|<=k else -1  (lecture formulation with infeasible sentinel)
    - If k is None (no given uniform constraint, e.g., PWT): maximise ( f(x), -|x| )
    Returns: (best_x, best_obj_tuple, pareto_set) where pareto_set = list[(x, (f1,f2))]
    """
    n = problem.meta_data.n_variables

    if k is None:
        # trade-off mode (no explicit k)
        def obj(x):
            fx = float(problem(x))
            return (fx, float(-np.sum(x)))
    else:
        # uniform-constraint mode
        def obj(x):
            fx = float(problem(x))
            size = int(np.sum(x))
            z = fx if size <= k else -1.0
            return (z, float(-size))  # negative size so we still "maximize both"

    # Initialise with empty solution + one random point
    empty = np.zeros(n, dtype=int)
    P = [(empty, obj(empty))]
    x0 = rng.integers(0, 2, size=n, dtype=int)
    P.append((x0, obj(x0)))

    evals = 2
    while evals < budget:
        # parent selection
        pidx = rng.integers(len(P))
        parent = P[pidx][0]

        # standard 1/n-bit mutation (ensure at least one flip)
        flips = rng.random(n) < (1.0 / n)
        if not flips.any():
            flips[rng.integers(n)] = True
        child = parent.copy()
        child[flips] ^= 1

        fchild = obj(child)
        evals += 1

        # Pareto update
        keep, dominated_flag = [], False
        for (z, fz) in P:
            if _dominates(fchild, fz):
                continue          # drop dominated point
            if _dominates(fz, fchild):
                dominated_flag = True
            keep.append((z, fz))
        if not dominated_flag:
            keep.append((child, fchild))
        P = keep

    # Best by primary (first) objective
    best = max(P, key=lambda t: t[1][0])
    return best[0], best[1], P
