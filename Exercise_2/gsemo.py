import numpy as np

def _dominates(a, b):
    """Pareto dominance for 'maximize both' objectives."""
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))

def gsemo(problem, budget, rng, k=None):
    """
    GSEMO for monotone submodular optimisation.
    - With uniform constraint k: objectives = (z(x), -|S|), z(x)=f(x) if |S|<=k else -1
    - Without k (PWT): objectives = (f(x), -|S|)
    """
    n = problem.meta_data.n_variables

    if k is None:
        def obj(x):
            fx = float(problem(x))
            return (fx, float(-np.sum(x)))
    else:
        def obj(x):
            fx = float(problem(x))
            size = int(np.sum(x))
            z = fx if size <= k else -1.0
            return (z, float(-size))

    # Initialise with empty and one random solution
    empty = np.zeros(n, dtype=int)
    P = [(empty, obj(empty))]
    x0 = rng.integers(0, 2, size=n, dtype=int)
    P.append((x0, obj(x0)))
    evals = 2

    while evals < budget:
        pidx = rng.integers(len(P))
        parent = P[pidx][0]

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
                continue
            if _dominates(fz, fchild):
                dominated_flag = True
            keep.append((z, fz))
        if not dominated_flag:
            keep.append((child, fchild))
        P = keep

    best = max(P, key=lambda t: t[1][0])
    return best[0], best[1], P
