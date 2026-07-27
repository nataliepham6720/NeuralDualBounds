"""
Solve the exact (canonical) education--voting LP and its reductions.

Run:
    python exact_lp_solver.py --kx 3 --ky 3 --n 50000
    python exact_lp_solver.py --kx 3 --ky 3 --backend scip
    python exact_lp_solver.py --kx 4 --ky 4 --skip-exact   # pruned LP only
"""

import argparse
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from LP_construction import build_constraints_EV
from LP_construction_exact import (
    generate_data_EV,
    empirical_distribution_EV,
    marginal_and_conditional,
    bin_centers,
    build_constraints_EV_exact,
    prune_duplicate_columns,
    n_strata,
    dual_objective,
    dual_violation_reduced,
)


# ============================================================
# Backends
# ============================================================

def solve_lp_highs(c, A, b):
    """Both senses through HiGHS; returns (lower, upper, duals)."""
    out = {}
    for sense, sign in (("min", 1.0), ("max", -1.0)):
        res = linprog(sign * c, A_eq=A, b_eq=b, bounds=(0, None),
                      method="highs")
        if not res.success:
            raise RuntimeError(f"HiGHS failed ({sense}): {res.message}")
        out[sense] = (sign * res.fun, sign * res.eqlin.marginals)
    return out["min"][0], out["max"][0], {"min": out["min"][1],
                                          "max": out["max"][1]}


def solve_lp_scip(c, A, b, eps=1e-9):
    """Same interface, through SCIP (mirrors scip_solver.py)."""
    from pyscipopt import Model

    A = A.toarray() if sparse.issparse(A) else np.asarray(A)
    n = len(c)

    def solve_sense(sense):
        m = Model()
        m.hideOutput()
        p = [m.addVar(lb=0, ub=1) for _ in range(n)]
        for i in range(len(b)):
            expr = sum(A[i, j] * p[j] for j in range(n) if A[i, j] != 0)
            m.addCons(expr >= b[i] - eps)
            m.addCons(expr <= b[i] + eps)
        m.setObjective(sum(c[j] * p[j] for j in range(n)), sense)
        m.optimize()
        if m.getStatus() != "optimal":
            raise RuntimeError("SCIP infeasible")
        return m.getObjVal()

    return solve_sense("minimize"), solve_sense("maximize"), None


BACKENDS = {"highs": solve_lp_highs, "scip": solve_lp_scip}


# ============================================================
# Experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--kx", type=int, default=3)
    ap.add_argument("--ky", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--backend", choices=list(BACKENDS), default="highs")
    ap.add_argument("--skip-exact", action="store_true",
                    help="build the pruned LP only (for large kx, ky)")
    args = ap.parse_args()

    kx, ky = args.kx, args.ky
    solve = BACKENDS[args.backend]

    np.random.seed(args.seed)

    data, Y0, Y1 = generate_data_EV(args.n, tau=args.tau, seed=args.seed)
    print("True ATE:", np.mean(Y1 - Y0))

    P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
    px, p_cond = marginal_and_conditional(P)
    y_centers = bin_centers(y_bins)

    print(f"\nkx = {kx}, ky = {ky}")
    print(f"#strata = 2^{kx} * {ky}^{2 * kx} = {n_strata(kx, ky):,}")
    print(f"#distinct columns = (2*{ky})^{kx} = {(2 * ky) ** kx:,}")

    # ---- exact LP ---------------------------------------------------
    t0 = time.time()
    A, b, c, labels = build_constraints_EV_exact(P, kx, ky, y_bins)
    t_build = time.time() - t0
    print(f"\nA shape: {A.shape}  nnz: {A.nnz}  (built in {t_build:.2f}s)")

    if not args.skip_exact:
        t0 = time.time()
        lo, hi, duals = solve(c, A, b)
        t_exact = time.time() - t0
        print("\n=============== EXACT LP ===============")
        print(f"ATE LOWER: {lo:.6f}")
        print(f"ATE UPPER: {hi:.6f}")
        print(f"time: {t_exact:.2f}s")

        if duals is not None:
            for sense, bound in (("min", lo), ("max", hi)):
                lam_full = duals[sense]
                lam, lam_empty = lam_full[:-1], lam_full[-1]
                obj = dual_objective(lam_full[:-1], lam_empty, b)
                viol = dual_violation_reduced(lam, lam_empty, labels, kx, ky,
                                              px, y_centers, sense=sense)
                print(f"  dual[{sense}]: objective {obj:.6f} "
                      f"(gap {abs(obj - bound):.2e}), "
                      f"max reduced violation {viol:.2e}")

    # ---- pruned LP --------------------------------------------------
    print("\n=============== PRUNED LP ==============")
    for sense in ("min", "max"):
        t0 = time.time()
        Ap, cp, keep = prune_duplicate_columns(A, c, sense=sense)
        lo_p, hi_p, _ = solve(cp, Ap, b)
        val = lo_p if sense == "min" else hi_p
        print(f"{sense}: {val:.6f}   "
              f"({Ap.shape[1]:,} columns kept, {time.time() - t0:.2f}s)")

    # ---- collapsed LP (LP_construction.build_constraints_EV) --------
    t0 = time.time()
    Ac, bc, cc, _ = build_constraints_EV(P, kx, ky, y_bins)
    lo_c, hi_c, _ = solve(cc, Ac, bc)
    print("\n============= COLLAPSED LP =============")
    print(f"ATE LOWER: {lo_c:.6f}")
    print(f"ATE UPPER: {hi_c:.6f}")
    print(f"({len(cc):,} columns, {time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()