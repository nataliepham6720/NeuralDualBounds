"""
Education--voting design: full primal LP versus reduced dual.

Emits the rows of Table `tab:ev` -- ATE lower, ATE upper and wall-clock time
for each solver at each discretization -- and saves the discretized draw for
every configuration so the numbers are reproducible without re-sampling.

    python table_ev.py                      # table rows (SCIP primal)
    python table_ev.py --backend highs      # both sides through HiGHS
    python table_ev.py --verify             # losslessness checks at small k

The two solvers being compared are

  primal        the LP of LP_construction.build_constraints_EV: one variable
                per (x, d, y0, y1), i.e. 2 kx ky^2 columns and 2 kx ky + 1
                rows.  This is the tractable writing of the primal -- the
                canonical enumeration behind it has 2^kx ky^(2kx) strata
                (3.3e29 at kx = ky = 12), so it is never formed.  At small k
                the two are checked to give identical bounds (--verify).

  reduced dual  the dual of that program after the collapse: both sides of
                the constraint (A^T lambda)_(d,y) <= c_y are additive over x
                and each d_x is free, so the ky^(2kx) dual constraints reduce
                to 2 kx ky + 1 rows over 2 kx ky + kx + 1 variables.  The
                inner max over the unconstrained arm is taken analytically,
                which is where the y-loop disappears:

                    t_x >= lambda_{x,0,y} + px (yc[y] - yc_min)
                    t_x >= lambda_{x,1,y} + px (yc_max - yc[y])
                    sum_x t_x + lambda_empty <= 0

                (senses flip for the upper bound).  Lossless by LP duality.
"""

import argparse
import os
import time

import numpy as np
from typing import NamedTuple

from scipy import sparse
from scipy.optimize import linprog

from LP_construction import build_constraints_EV
from LP_cons import (
    generate_data_EV,
    empirical_distribution_EV,
    marginal_and_conditional,
    bin_centers,
    build_constraints_EV_exact,
    n_strata,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# (kx, ky) configurations reported in the table
CONFIGS = [(3, 3), (4, 6)]

N_POINTS = 10000
TAU = 0.5
SEED = 2020


# ============================================================
# Data: generate once, save, reuse
# ============================================================

def load_or_make(kx, ky, n=N_POINTS, tau=TAU, seed=SEED, data_dir=DATA_DIR,
                 refresh=False):
    """
    Discretized draw for (kx, ky, n, seed), cached to
    data/EV_kx{kx}_ky{ky}_n{n}_s{seed}.npz.

    Stores P / x_bins / y_bins -- the same layout as the P12_EV.npz and
    P16_EV.npz already in this directory -- plus the true ATE of the draw, so
    a rerun reproduces the table without re-sampling and without depending on
    numpy's RNG staying fixed.
    """
    path = os.path.join(data_dir, f"EV_kx{kx}_ky{ky}_n{n}_s{seed}.npz")

    if os.path.exists(path) and not refresh:
        with np.load(path) as z:
            P, x_bins, y_bins = z["P"], z["x_bins"], z["y_bins"]
            ate = float(z["ate_true"])
        made = False
    else:
        data, Y0, Y1 = generate_data_EV(n, tau=tau, seed=seed)
        P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
        ate = float(np.mean(Y1 - Y0))
        os.makedirs(data_dir, exist_ok=True)
        np.savez(path, P=P, x_bins=x_bins, y_bins=y_bins,
                 ate_true=ate, n=n, tau=tau, seed=seed, kx=kx, ky=ky)
        made = True

    return P, x_bins, y_bins, ate, path, made


# ============================================================
# Solvers
# ============================================================

class timings(NamedTuple):
    """
    One solver's result at one configuration.

    t_matrix  assembling the coefficient arrays in numpy
    t_model   handing the program to the solver (SCIP only; linprog fuses
              this into the solve, and it is where scip_solver.py's dense
              `sum(A[i,j]*p[j] for j in range(n))` spends its time)
    t_opt     the simplex itself
    """
    lo: float
    hi: float
    t_matrix: float
    t_model: float
    t_opt: float
    shape: tuple

    @property
    def total(self):
        return self.t_matrix + self.t_model + self.t_opt


def solve_highs(c, A_eq, b_eq, sense, A_ub=None, b_ub=None, bounds=(0, None)):
    """Returns (value, model_seconds, optimize_seconds); linprog fuses the two."""
    sign = 1.0 if sense == "min" else -1.0
    t0 = time.time()
    res = linprog(sign * np.asarray(c), A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"HiGHS failed ({sense}): {res.message}")
    return sign * res.fun, 0.0, time.time() - t0


def solve_scip(c, A, b, sense, eps=1e-9):
    """
    Mirrors scip_solver.py: equality constraints as a two-sided pair.

    Note eps genuinely relaxes the program -- scip_solver.py's 1e-6 widens the
    bounds by ~6e-5 here -- so the table uses 1e-9.
    """
    from pyscipopt import Model

    A = A.toarray() if sparse.issparse(A) else np.asarray(A)
    n = len(c)

    t0 = time.time()
    m = Model()
    m.hideOutput()
    p = [m.addVar(lb=0, ub=1) for _ in range(n)]
    m.addCons(sum(p) == 1)
    for i in range(len(b)):
        expr = sum(A[i, j] * p[j] for j in range(n))
        m.addCons(expr >= b[i] - eps)
        m.addCons(expr <= b[i] + eps)
    m.setObjective(sum(c[j] * p[j] for j in range(n)),
                   "minimize" if sense == "min" else "maximize")
    t1 = time.time()
    m.optimize()
    print("solve sucessfully")
    t2 = time.time()
    if m.getStatus() != "optimal":
        raise RuntimeError("SCIP infeasible")
    return m.getObjVal(), t1 - t0, t2 - t1


# ------------------------------------------------------------
# Primal
# ------------------------------------------------------------

def run_primal(P, kx, ky, y_bins, backend, eps=1e-9):
    """Build + solve both senses.  See `timings` for the returned fields."""
    t0 = time.time()
    A, b, c, _ = build_constraints_EV(P, kx, ky, y_bins)
    t_matrix = time.time() - t0

    t_model = t_opt = 0.0
    vals = {}
    for sense in ("min", "max"):
        if backend == "scip":
            v, tm, to = solve_scip(c, A, b, sense, eps=eps)
        else:
            v, tm, to = solve_highs(c, A, b, sense)
        vals[sense] = v
        t_model += tm
        t_opt += to

    return timings(vals["min"], vals["max"], t_matrix, t_model, t_opt, A.shape)


# ------------------------------------------------------------
# Reduced dual
# ------------------------------------------------------------

def build_reduced_dual(P, kx, ky, y_bins, sense):
    """
    Compact LP for the dual.

    Variables, in order: lambda[x,d,y]  (2 kx ky),  t[x]  (kx),  lambda_empty.
    Rows: 2 kx ky linear constraints + 1 aggregate.  Everything is free.
    """
    px, p_cond = marginal_and_conditional(P)
    yc = bin_centers(y_bins)
    yc_min, yc_max = yc.min(), yc.max()

    n_lam = 2 * kx * ky
    n_var = n_lam + kx + 1
    i_t = n_lam                       # offset of t
    i_0 = n_lam + kx                  # index of lambda_empty

    def lam_idx(x, d, y):
        return (x * 2 + d) * ky + y

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for x in range(kx):
        for y in range(ky):
            for d in (0, 1):
                # min: t_x >= lambda + shift    ->   lambda - t_x <= -shift
                # max: s_x <= lambda + shift    ->   s_x - lambda <= shift
                if d == 0:
                    shift = px[x] * (yc[y] - (yc_min if sense == "min"
                                              else yc_max))
                else:
                    shift = px[x] * ((yc_max if sense == "min" else yc_min)
                                     - yc[y])
                s = 1.0 if sense == "min" else -1.0
                rows += [r, r]
                cols += [lam_idx(x, d, y), i_t + x]
                vals += [s, -s]
                rhs.append(-s * shift)
                r += 1

    # sum_x t_x + lambda_empty <= 0   (>= 0 for the upper bound)
    s = 1.0 if sense == "min" else -1.0
    rows += [r] * (kx + 1)
    cols += list(range(i_t, i_t + kx)) + [i_0]
    vals += [s] * (kx + 1)
    rhs.append(0.0)
    r += 1

    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    b_ub = np.array(rhs)

    # objective  sum_{x,d,y} p(d,y|x) lambda_{x,d,y} + lambda_empty
    c = np.zeros(n_var)
    for x in range(kx):
        for d in (0, 1):
            for y in range(ky):
                c[lam_idx(x, d, y)] = p_cond[x, d, y]
    c[i_0] = 1.0

    return c, A_ub, b_ub


def run_reduced_dual(P, kx, ky, y_bins):
    """
    Both senses.

    The dual of the minimizing primal is a maximization and vice versa, so the
    lower bound comes from the 'min' system solved as a max, and conversely.
    """
    t_matrix = t_opt = 0.0
    shape = None
    out = {}
    for sense in ("min", "max"):
        t0 = time.time()
        c, A_ub, b_ub = build_reduced_dual(P, kx, ky, y_bins, sense)
        t_matrix += time.time() - t0
        shape = A_ub.shape
        v, _, to = solve_highs(c, None, None,
                               "max" if sense == "min" else "min",
                               A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
        out[sense] = v
        t_opt += to
    return timings(out["min"], out["max"], t_matrix, 0.0, t_opt, shape)


# ============================================================
# Verification (small k, where the canonical LP is formable)
# ============================================================

def verify(tol=1e-7):
    print("Losslessness checks: exact canonical LP == primal == reduced dual")
    for kx, ky in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 3)):
        P, x_bins, y_bins, _, _, _ = load_or_make(kx, ky, n=20000, seed=7)

        A, b, c, _ = build_constraints_EV_exact(P, kx, ky, y_bins)
        ex_lo, _, _ = solve_highs(c, A, b, "min")
        ex_hi, _, _ = solve_highs(c, A, b, "max")

        pr = run_primal(P, kx, ky, y_bins, "highs")
        du = run_reduced_dual(P, kx, ky, y_bins)

        for name, lo, hi in (("primal", pr.lo, pr.hi),
                             ("dual", du.lo, du.hi)):
            assert abs(lo - ex_lo) < tol, (kx, ky, name, lo, ex_lo)
            assert abs(hi - ex_hi) < tol, (kx, ky, name, hi, ex_hi)

        print(f"  [ok] kx={kx} ky={ky}: [{ex_lo:.6f}, {ex_hi:.6f}] "
              f"from all three ({n_strata(kx, ky):,} canonical strata)")
    print()


# ============================================================
# Table
# ============================================================

def k_label(kx, ky):
    return f"${kx}$" if kx == ky else f"$k_x{{=}}{kx},\\,k_y{{=}}{ky}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("scip", "highs"), default="scip",
                    help="solver for the primal (the dual is always HiGHS)")
    ap.add_argument("--n", type=int, default=N_POINTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--refresh", action="store_true",
                    help="regenerate the saved draws")
    ap.add_argument("--eps", type=float, default=1e-9,
                    help="SCIP equality slack (scip_solver.py uses 1e-6)")
    ap.add_argument("--verify", action="store_true",
                    help="run the small-k losslessness checks first")
    args = ap.parse_args()

    if args.verify:
        verify()

    rows = []
    for kx, ky in CONFIGS:
        P, x_bins, y_bins, ate, path, made = load_or_make(
            kx, ky, n=args.n, seed=args.seed, refresh=args.refresh)

        print(f"kx = {kx}, ky = {ky}   "
              f"({'saved' if made else 'loaded'} {os.path.relpath(path, HERE)})")
        print(f"  true ATE {ate:.4f};  canonical strata "
              f"2^{kx} * {ky}^{2 * kx} = {n_strata(kx, ky):.3e}")

        pr = run_primal(P, kx, ky, y_bins, args.backend, eps=args.eps)
        du = run_reduced_dual(P, kx, ky, y_bins)

        for name, res, be in (("primal      ", pr, args.backend),
                              ("reduced dual", du, "highs")):
            print(f"  {name} {res.shape[0]:4d} x {res.shape[1]:<5d} "
                  f"[{res.lo:.6f}, {res.hi:.6f}]  {res.total:8.3f}s  "
                  f"(matrix {res.t_matrix:.3f} + model {res.t_model:.3f} "
                  f"+ opt {res.t_opt:.3f}, {be})")
        print(f"  |gap| lower {abs(pr.lo - du.lo):.2e}   "
              f"upper {abs(pr.hi - du.hi):.2e}   "
              f"speedup {pr.total / max(du.total, 1e-12):.0f}x total, "
              f"{pr.t_opt / max(du.t_opt, 1e-12):.0f}x on the solve alone\n")

        rows.append((kx, ky, pr, du))

    # ---- LaTeX ------------------------------------------------------
    print("% --- rows for tab:ev "
          f"(primal: {args.backend}; reduced dual: highs) ---")
    for i, (kx, ky, pr, du) in enumerate(rows):
        if i:
            print(r"\midrule")
        print(f"\\multirow{{2}}{{*}}{{{k_label(kx, ky)}}} & primal       "
              f"& ${pr.lo:.3f}$ & ${pr.hi:.3f}$ & ${pr.total:.2f}$\\\\")
        print(f"                      & reduced dual "
              f"& ${du.lo:.3f}$ & ${du.hi:.3f}$ & ${du.total:.4f}$\\\\")


if __name__ == "__main__":
    main()