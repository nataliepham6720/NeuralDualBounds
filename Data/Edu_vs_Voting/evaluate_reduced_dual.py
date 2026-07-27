"""
Education--voting design: canonical primal LP versus original dual versus
reduced dual.

Emits the rows of Table `tab:ev` -- ATE lower, ATE upper and wall-clock time
for each solver at each discretization -- and saves the discretized draw for
every configuration so the numbers are reproducible without re-sampling.

    python evaluate_reduced_dual.py                   # table rows (HiGHS)
    python evaluate_reduced_dual.py --backend scip    # primal through SCIP
    python evaluate_reduced_dual.py --verify          # losslessness checks

Sizes.  Write

    m = |X| |D| |Y| = 2 kx ky            observable cells
    N = |D|^|X| |Y|^{|D||X|} = 2^kx ky^(2kx)     canonical strata

The three programs timed against each other, and why only the last one is
solvable at every discretization in the table:

  primal        min/max c^T q   s.t.  A q = b,  q >= 0
                (m + 1) x N.  One row per observable cell plus normalization
                -- *linear* in the data -- and one column per canonical
                stratum -- *exponential* in kx.  This is the program an
                autobound-style pipeline hands to SCIP/HiGHS, and it is what
                `run_primal` builds (`build_constraints_EV_exact`) and times.
                At kx = ky = 6 it already has 1.4e11 columns, so `--max-cols`
                refuses to form it and the row is reported as o.o.m.

  dual          max/min b^T lambda  s.t.  A^T lambda <= c   (senses flip)
                N x (m + 1).  The transpose: the exponent moves from the
                columns to the rows.  Variables are lambda_{x,d,y} and
                lambda_empty -- linearly many -- against one constraint per
                stratum.  `run_dual_original` forms A^T explicitly and hands
                it to HiGHS, so the table shows what the *unreduced* dual
                costs; it hits the same `--max-cols` wall as the primal (the
                exponent just moves to the rows), which is exactly the point.

  reduced dual  the same dual after the collapse, (m + 1) x (m + kx + 1):
                *linear in both directions*.  Both sides of the constraint
                (A^T lambda)_(d,y) <= c_y are additive over x and each d_x is
                free, so the N dual constraints reduce to a single separable
                one; writing each per-x inner maximum as an epigraph variable
                t_x and taking the max over the *unconstrained* arm
                analytically (which is where the y-loop disappears) leaves

                    t_x >= lambda_{x,0,y} + px (yc[y] - yc_min)
                    t_x >= lambda_{x,1,y} + px (yc_max - yc[y])
                    sum_x t_x + lambda_empty <= 0

                i.e. 2 kx ky + 1 rows over 2 kx ky + kx + 1 variables (senses
                flip for the upper bound).  Lossless by LP duality, and
                cross-checked against the exponential dual constraint set by
                `--verify`.

Rows and variables of every program are indexed by the *occupied* x-bins only,
matching `build_constraints_EV_exact`, which drops rows for empty bins.
"""

import argparse
import os
import time

import numpy as np
from typing import NamedTuple, Optional

from scipy import sparse
from scipy.optimize import linprog

from LP_construction import (
    generate_data_EV,
    empirical_distribution_EV,
    marginal_and_conditional,
    bin_centers,
    build_constraints_EV_exact,
    dual_violation_bruteforce,
    dual_violation_reduced,
    # lp_shapes,
    n_strata,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# (kx, ky) configurations reported in the table
CONFIGS = [(4, 4), (4, 5)]

N_POINTS = 10000
TAU = 0.5
SEED = 2020

# Refuse to materialize a program with an exponential dimension beyond this
# (columns of the primal, rows of the original dual -- the same count N).
# At (4,4) N is 1.0e6, at (4,6) 2.7e7, at (6,6) 1.4e11.
MAX_PRIMAL_COLS = 5e10 #0_000_000


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
    shape     (rows, columns) of the program actually solved
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
    t2 = time.time()
    if m.getStatus() != "optimal":
        raise RuntimeError("SCIP infeasible")
    return m.getObjVal(), t1 - t0, t2 - t1


# ------------------------------------------------------------
# Primal:  (m + 1) rows, 2^kx ky^(2kx) columns
# ------------------------------------------------------------

def run_primal(P, kx, ky, y_bins, backend, eps=1e-9,
               max_cols=MAX_PRIMAL_COLS) -> Optional[timings]:
    """
    Build + solve the canonical primal in both senses.

    Returns None when the column count exceeds `max_cols`, i.e. when the
    exponential side of the program cannot be materialized at all -- that is
    the failure mode the table is there to exhibit, so it is reported rather
    than worked around.
    """
    n_cols = n_strata(kx, ky)
    if n_cols > max_cols:
        return None

    t0 = time.time()
    A, b, c, _ = build_constraints_EV_exact(P, kx, ky, y_bins)
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
# Original dual:  2^kx ky^(2kx) rows, (m + 1) columns
# ------------------------------------------------------------

def run_dual_original(P, kx, ky, y_bins,
                      max_rows=MAX_PRIMAL_COLS) -> Optional[timings]:
    """
    Build + solve the *unreduced* dual in both senses, through HiGHS.

    The program is the literal transpose of the primal: free variables
    lambda_{x,d,y} and lambda_empty (m + 1 of them, one per primal row), and
    one inequality per canonical stratum (d, y):

        lower bound:  max  b^T lambda   s.t.  A^T lambda <= c
        upper bound:  min  b^T lambda   s.t.  A^T lambda >= c

    A^T is formed by transposing the sparse A of `build_constraints_EV_exact`,
    so this run demonstrates what the dual costs *before* the collapse: the
    exponent moves from the primal's columns to the dual's rows, and the same
    `max_rows` guard trips at the same configurations.  Returns None when the
    2^kx ky^(2kx) rows exceed the guard.
    """
    n_rows = n_strata(kx, ky)
    if n_rows > max_rows:
        return None

    t0 = time.time()
    A, b, c, _ = build_constraints_EV_exact(P, kx, ky, y_bins)
    At = A.T.tocsr()
    t_matrix = time.time() - t0

    t_opt = 0.0
    out = {}
    for sense in ("min", "max"):
        if sense == "min":
            # dual of the minimizing primal: maximize b^T lambda, A^T lambda <= c
            v, _, to = solve_highs(b, None, None, "max",
                                   A_ub=At, b_ub=c, bounds=(None, None))
        else:
            # dual of the maximizing primal: minimize b^T lambda, A^T lambda >= c
            v, _, to = solve_highs(b, None, None, "min",
                                   A_ub=-At, b_ub=-c, bounds=(None, None))
        out[sense] = v
        t_opt += to

    return timings(out["min"], out["max"], t_matrix, 0.0, t_opt, At.shape)


# ------------------------------------------------------------
# Reduced dual:  (m + 1) rows, (m + kx + 1) columns
# ------------------------------------------------------------

def build_reduced_dual(P, kx, ky, y_bins, sense):
    """
    Compact LP for the dual.

    Variables, in order: lambda[x,d,y]  (2 ns ky),  t[x]  (ns),  lambda_empty,
    where ns = #occupied x-bins.  Rows: 2 ns ky linearized epigraph
    constraints + 1 aggregate.  Everything is free.

    The lambda block is exactly the dual variable per primal row, so the
    column count matches the primal's row count up to the ns epigraph
    variables: O(|X||D||Y|) columns against O(|X||D||Y|) rows, in place of the
    dual's 2^kx ky^(2kx) rows.

    Empty x-bins carry no primal row, hence no dual variable: they are dropped
    here too.  (Keeping them would leave lambda_{x,d,y} free and unpriced, and
    the program would be unbounded.)
    """
    px, p_cond = marginal_and_conditional(P)
    yc = bin_centers(y_bins)
    yc_min, yc_max = yc.min(), yc.max()

    x_support = [x for x in range(kx) if px[x] > 0]
    ns = len(x_support)

    n_lam = 2 * ns * ky
    n_var = n_lam + ns + 1
    i_t = n_lam                       # offset of t
    i_0 = n_lam + ns                  # index of lambda_empty

    def lam_idx(xi, d, y):
        return (xi * 2 + d) * ky + y

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for xi, x in enumerate(x_support):
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
                cols += [lam_idx(xi, d, y), i_t + xi]
                vals += [s, -s]
                rhs.append(-s * shift)
                r += 1

    # sum_x t_x + lambda_empty <= 0   (>= 0 for the upper bound)
    s = 1.0 if sense == "min" else -1.0
    rows += [r] * (ns + 1)
    cols += list(range(i_t, i_t + ns)) + [i_0]
    vals += [s] * (ns + 1)
    rhs.append(0.0)
    r += 1

    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    b_ub = np.array(rhs)

    # objective  sum_{x,d,y} p(d,y|x) lambda_{x,d,y} + lambda_empty
    c = np.zeros(n_var)
    for xi, x in enumerate(x_support):
        for d in (0, 1):
            for y in range(ky):
                c[lam_idx(xi, d, y)] = p_cond[x, d, y]
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


def reduced_dual_solution(P, kx, ky, y_bins, sense):
    """
    The optimal (lambda grid, lambda_empty) of the reduced dual, in the
    (x, d, y) layout `dual_violation_*` expect.  Used by --verify to check the
    collapsed constraint set against the exponential one.
    """
    px, _ = marginal_and_conditional(P)
    x_support = [x for x in range(kx) if px[x] > 0]

    c, A_ub, b_ub = build_reduced_dual(P, kx, ky, y_bins, sense)
    sign = -1.0 if sense == "min" else 1.0   # dual of min-primal is a max
    res = linprog(sign * c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None),
                  method="highs")
    if not res.success:
        raise RuntimeError(f"reduced dual failed ({sense}): {res.message}")

    grid = np.zeros((kx, 2, ky))
    for xi, x in enumerate(x_support):
        for d in (0, 1):
            for y in range(ky):
                grid[x, d, y] = res.x[(xi * 2 + d) * ky + y]
    lam_empty = float(res.x[-1])
    return grid, lam_empty, sign * res.fun


# ============================================================
# Verification (small k, where the canonical LP is formable)
# ============================================================

def verify(tol=1e-7):
    """
    Three checks at sizes where the exponential program can still be written:

      values   canonical primal == original dual == reduced dual, both senses
               (strong duality plus losslessness of the collapse);
      rows     the reduced dual's optimum is feasible for the full set of
               2^kx ky^(2kx) dual constraints, checked by brute force -- i.e.
               dropping those rows in favour of 2 kx ky + 1 loses nothing.
    """
    print("Losslessness checks: canonical primal == original dual == "
          "reduced dual, and the collapsed rows imply the exponential ones")
    for kx, ky in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 3)):
        P, x_bins, y_bins, _, _, _ = load_or_make(kx, ky, n=20000, seed=7)
        px, _ = marginal_and_conditional(P)
        yc = bin_centers(y_bins)

        A, b, c, labels = build_constraints_EV_exact(P, kx, ky, y_bins)
        ex_lo, _, _ = solve_highs(c, A, b, "min")
        ex_hi, _, _ = solve_highs(c, A, b, "max")

        og = run_dual_original(P, kx, ky, y_bins)
        assert abs(og.lo - ex_lo) < tol, (kx, ky, "dual", og.lo, ex_lo)
        assert abs(og.hi - ex_hi) < tol, (kx, ky, "dual", og.hi, ex_hi)

        du = run_reduced_dual(P, kx, ky, y_bins)
        assert abs(du.lo - ex_lo) < tol, (kx, ky, du.lo, ex_lo)
        assert abs(du.hi - ex_hi) < tol, (kx, ky, du.hi, ex_hi)

        worst = -np.inf
        for sense in ("min", "max"):
            grid, lam_empty, val = reduced_dual_solution(P, kx, ky, y_bins,
                                                         sense)
            lam = np.array([grid[x, d, y] for (x, d, y) in labels[:-1]])
            bf = dual_violation_bruteforce(lam, lam_empty, labels, kx, ky,
                                           px, yc, sense=sense)
            rd = dual_violation_reduced(lam, lam_empty, labels, kx, ky,
                                        px, yc, sense=sense)
            assert bf <= tol, (kx, ky, sense, "infeasible for the full dual", bf)
            assert abs(bf - rd) < tol, (kx, ky, sense, bf, rd)
            worst = max(worst, bf)

        # shapes = lp_shapes(kx, ky, kx_support=int((px > 0).sum()))
        print(f"  [ok] kx={kx} ky={ky}: [{ex_lo:.6f}, {ex_hi:.6f}]  "
              f"primal {shapes['primal'][0]}x{shapes['primal'][1]:,} | "
              f"dual {shapes['dual'][0]:,}x{shapes['dual'][1]} | "
              f"reduced {shapes['reduced_dual'][0]}x"
              f"{shapes['reduced_dual'][1]}   "
              f"max dual violation {worst:.1e}")
    print()


# ============================================================
# Table
# ============================================================

def k_label(kx, ky):
    return f"${kx}$" if kx == ky else f"$k_x{{=}}{kx},\\,k_y{{=}}{ky}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("scip", "highs"), default="highs",
                    help="solver for the primal (the duals are always HiGHS)")
    ap.add_argument("--n", type=int, default=N_POINTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--refresh", action="store_true",
                    help="regenerate the saved draws")
    ap.add_argument("--eps", type=float, default=1e-9,
                    help="SCIP equality slack (scip_solver.py uses 1e-6)")
    ap.add_argument("--max-cols", type=int, default=MAX_PRIMAL_COLS,
                    help="skip the primal / original dual beyond this many "
                         "columns / rows")
    ap.add_argument("--verify", action="store_true",
                    help="run the small-k losslessness checks first")
    args = ap.parse_args()

    if args.verify:
        verify()

    rows = []
    for kx, ky in CONFIGS:
        P, x_bins, y_bins, ate, path, made = load_or_make(
            kx, ky, n=args.n, seed=args.seed, refresh=args.refresh)
        px, _ = marginal_and_conditional(P)

        print(f"kx = {kx}, ky = {ky}")

        pr = run_primal(P, kx, ky, y_bins, args.backend, eps=args.eps,
                        max_cols=args.max_cols)
        og = run_dual_original(P, kx, ky, y_bins, max_rows=args.max_cols)
        du = run_reduced_dual(P, kx, ky, y_bins)

        if pr is None:
            print(f"  primal / original dual not formable "
                  f"(N = {n_strata(kx, ky):.2e} > "
                  f"--max-cols {args.max_cols:.2e})")
        for name, res, be in (("primal       ", pr, args.backend),
                              ("original dual", og, "highs"),
                              ("reduced dual ", du, "highs")):
            if res is None:
                continue
            print(f"  {name} {res.shape[0]:9,} x {res.shape[1]:<12,} "
                  f"[{res.lo:.6f}, {res.hi:.6f}]  {res.total:8.3f}s  "
                  f"(matrix {res.t_matrix:.3f} + model {res.t_model:.3f} "
                  f"+ opt {res.t_opt:.3f}, {be})")
        if pr is not None:
            print(f"  |gap| lower {abs(pr.lo - du.lo):.2e}   "
                  f"upper {abs(pr.hi - du.hi):.2e}   "
                  f"speedup vs primal {pr.total / max(du.total, 1e-12):.0f}x, "
                  f"vs original dual "
                  f"{og.total / max(du.total, 1e-12):.0f}x")
        print()

        rows.append((kx, ky, pr, og, du))

    # ---- LaTeX ------------------------------------------------------
    def cells(res):
        if res is None:
            return r"& --- & --- & \multicolumn{1}{c}{o.o.m.}"
        return f"& ${res.lo:.3f}$ & ${res.hi:.3f}$ & ${res.total:.4f}$"

    print("% --- rows for tab:ev "
          f"(primal: {args.backend}; duals: highs) ---")
    for i, (kx, ky, pr, og, du) in enumerate(rows):
        if i:
            print(r"\midrule")
        print(f"\\multirow{{3}}{{*}}{{{k_label(kx, ky)}}} "
              f"& primal        {cells(pr)}\\\\")
        print(f"                      & original dual {cells(og)}\\\\")
        print(f"                      & reduced dual  {cells(du)}\\\\")


if __name__ == "__main__":
    main()