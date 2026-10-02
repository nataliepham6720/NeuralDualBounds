"""
IV_cont ATE bounds through every route, on the outcome scale.

    python -m Z3.run_iv --k 6
    python -m Z3.run_iv --k 10 --routes reduced-highs reduced-scip reduced
    python -m Z3.run_iv --k 6 --routes primal reduced reduced-z3

"""

import argparse
import os

import numpy as np

from Data.IV_cont.LP_construction import (empirical_distribution_IV,
                                          generate_data_IV)

from . import problems, reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = ("primal", "full-highs", "full-scip", "full-ipm",
          "reduced-highs", "reduced-scip", "reduced", "reduced-z3")
TRUE_ATE = 3.0


def load_iv(k, n=10000, lam=0.5, seed=2020):
    """``(P, edges, source)``: P(T, Y | Z) on ``k`` bins and the Y-bin edges."""
    np.random.seed(seed)
    sample = generate_data_IV(n, lam)
    P = empirical_distribution_IV(sample, k)
    Y = sample[:, 2]
    edges = np.linspace(Y.min(), Y.max(), k + 1)     # as LP_construction.discretize
    path = os.path.join(ROOT, "Data", "IV_cont", f"P{k}.npy")
    if not os.path.exists(path):
        return P, edges, f"generate(n={n}, seed={seed})"
    stored = np.load(path)
    if stored.shape != P.shape or not np.allclose(stored, P, rtol=0, atol=1e-12):
        raise RuntimeError(f"redrawing seed={seed}, n={n} does not reproduce "
                           f"{path}; its Y-bin edges are unknown")
    return stored, edges, f"preload(P{k}.npy)"


def main():
    p = argparse.ArgumentParser(
        description="IV_cont ATE bounds, every route side by side.")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--units", choices=["y", "bins"], default="y",
                   help="y: Y-bin centers (comparable with TRUE ATE = 3); "
                        "bins: the bin indices 0..k-1")
    p.add_argument("--eps", type=float, default=0.0,
                   help="slack on the observational constraints")
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=None,
                   help="default: all but reduced-z3 (slow: about 20 s "
                        "per oracle call at k=6), minus the 2^k-sized ones "
                        "above k=8")
    args = p.parse_args()

    k = args.k
    P, edges, src = load_iv(k)
    centers = (edges[:-1] + edges[1:]) / 2
    yv = centers if args.units == "y" else None
    routes = args.routes or tuple(
        r for r in ROUTES if r != "reduced-z3" and
        (k <= 8 or not r.startswith(("primal", "full"))))

    def make(sign):
        return problems.IVCont(P, k, sign=sign, eps=args.eps, y_values=yv)

    reference.compare(make, routes=routes, baseline=routes[0],
                      label=f"IV_cont k={k} {src} units={args.units}")
    w = edges[1] - edges[0]
    if args.units == "y":
        print(f"\nTRUE ATE = {TRUE_ATE:g}")
    else:
        print(f"\nTRUE ATE = {TRUE_ATE:g} on the outcome scale, about "
              f"{TRUE_ATE / w:.3f} in bin indices (bin width {w:.4f})")


if __name__ == "__main__":
    main()