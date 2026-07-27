"""
Sweep the IV-cont ATE bound solvers across discretization levels k and
compare bounds and runtimes between the exponential primal formulations
and the polynomial O(k^3) dual.

CLI usage (from anywhere):

    python Data/IV_cont/sweep_solvers.py --ks 4 5 6 --solvers primal implicit dual
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Data.IV_cont.LP_construction import (
    generate_data_IV,
    empirical_distribution_IV,
    build_constraints_IV,
)
from Data.IV_cont.autobound_scip_solver import (
    solve_lp_scip,
    solve_lp_scip_implicit,
    solve_dual_lp_scip_poly,
)

SOLVERS = ("primal", "implicit", "dual")


def get_P(k, data="auto", n=10000, lam=0.5, P_path=None):
    """
    Return (P, source_label) for discretization level k.

    data="preload"  : load P{k}.npy (or P_path), error if missing
    data="generate" : draw n samples and compute the empirical distribution
    data="auto"     : preload if the file exists, otherwise generate
    """
    if data in ("auto", "preload"):
        path = P_path or os.path.join(_HERE, f"Data/IV_cont/P{k}.npy")
        if os.path.exists(path):
            P = np.load(path)
            if P.shape != (k, 2, k):
                raise ValueError(f"{path} has shape {P.shape}, expected {(k, 2, k)}")
            return P, f"preload({os.path.basename(path)})"
        if data == "preload":
            raise FileNotFoundError(path)
    samples = generate_data_IV(n, lam)
    return empirical_distribution_IV(samples, k=k), f"generate(n={n})"


def solve(P, k, solver, eps=1e-6):
    """Return (lower, upper) ATE bounds for the chosen solver."""
    if solver == "primal":
        A, b, c, labels = build_constraints_IV(P, k=k)
        lower, upper = solve_lp_scip(c, A, b, eps=eps)
        return lower["obj"], upper["obj"]
    if solver == "implicit":
        return solve_lp_scip_implicit(P, k, eps=eps)
    if solver == "dual":
        return solve_dual_lp_scip_poly(P, k, eps=eps)
    raise ValueError(f"unknown solver {solver!r}")


def run_sweep(ks=(4, 5, 6), solvers=SOLVERS, data="auto", n=10000, lam=0.5,
              eps=1e-6, seed=2020, max_exponential_k=8, tol=1e-4):
    """
    Run every solver at every k and print a comparison table.

    The primal/implicit formulations have 2^k * k^2 variables, so they are
    skipped for k > max_exponential_k (the dual scales as O(k^3) and always
    runs). Returns a pandas DataFrame if pandas is available, otherwise a
    list of row dicts.
    """
    rows = []
    for k in ks:
        np.random.seed(seed)  # same data for every solver at this k
        P, src = get_P(k, data=data, n=n, lam=lam)
        for solver in solvers:
            if solver in ("primal", "implicit") and k > max_exponential_k:
                print(f"[k={k}] skipping {solver}: 2^{k}*{k}^2 = "
                      f"{2**k * k * k} variables exceeds max_exponential_k={max_exponential_k}")
                continue
            print(f"[k={k}] solving with {solver} ({src})...")
            t0 = time.time()
            lower, upper = solve(P, k, solver, eps=eps)
            dt = time.time() - t0
            rows.append({"k": k, "data": src, "solver": solver,
                         "ATE_lower": lower, "ATE_upper": upper, "time_s": dt})

    print_table(rows)
    check_agreement(rows, tol=tol)

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def print_table(rows):
    """Print the sweep results as a markdown table."""
    print("\n| k | data | solver | ATE lower | ATE upper | time (s) |")
    print("|---|------|--------|-----------|-----------|----------|")
    for r in rows:
        print(f"| {r['k']} | {r['data']} | {r['solver']} "
              f"| {r['ATE_lower']:.6f} | {r['ATE_upper']:.6f} "
              f"| {r['time_s']:.2f} |")
    print("\nTRUE ATE = 3")


def check_agreement(rows, tol=1e-4):
    """Verify all solvers agree on the bounds at each k."""
    ks = sorted({r["k"] for r in rows})
    for k in ks:
        group = [r for r in rows if r["k"] == k]
        if len(group) < 2:
            continue
        lo = [r["ATE_lower"] for r in group]
        up = [r["ATE_upper"] for r in group]
        if max(lo) - min(lo) < tol and max(up) - min(up) < tol:
            print(f"k={k}: all solvers MATCH (tol={tol})")
        else:
            print(f"k={k}: MISMATCH  lower spread={max(lo) - min(lo):.2e}, "
                  f"upper spread={max(up) - min(up):.2e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep IV-cont ATE bound solvers over k and compare.")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 5, 6],
                        help="discretization levels to sweep")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS,
                        default=list(SOLVERS), help="solvers to compare")
    parser.add_argument("--data", choices=["auto", "preload", "generate"],
                        default="auto",
                        help="auto: preload P{k}.npy if present else generate")
    parser.add_argument("--n", type=int, default=10000,
                        help="sample size when generating data")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="mixture weight lambda of the noise distribution")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="slack on the observational constraints")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--max-exponential-k", type=int, default=8,
                        help="skip primal/implicit solvers above this k")
    args = parser.parse_args()

    run_sweep(ks=args.ks, solvers=args.solvers, data=args.data, n=args.n,
              lam=args.lam, eps=args.eps, seed=args.seed,
              max_exponential_k=args.max_exponential_k)


if __name__ == "__main__":
    main()
