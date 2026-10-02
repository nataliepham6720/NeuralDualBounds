"""
latent_dual.reference
=====================
Reference solves of the *same* dual, for cross-checking any
:class:`~latent_dual.main.LatentDual`.

Two of these need no auxiliary variables (the full dual has one linear row per
stratum and no ``max`` anywhere -- the ``max`` is created *by* the reduction).
The lifted route is the one that does, and it is here only to show the cost of
undoing the reduction for an LP solver.
"""

import time

import numpy as np


def full_dual_lp(prob, backend="highs-ipm"):
    """
    The full dual -- every stratum as an explicit row -- via HiGHS or SCIP.

    ``2^k``-ish rows, ``n_obs + 1`` columns, no auxiliary variables.
    """
    idx, rw = prob.all_rows()
    m, nnz = idx.shape[0], prob.nnz

    if backend == "scip":
        from pyscipopt import Model
        mod = Model()
        mod.hideOutput()
        v = [mod.addVar(lb=None, ub=None) for _ in range(prob.n)]
        for r in range(m):
            mod.addCons(sum(v[j] for j in idx[r]) + v[-1] <= float(rw[r]))
        mod.setObjective(sum(float(prob.b[i]) * v[i] for i in range(prob.n)),
                         "maximize")
        mod.optimize()
        if mod.getStatus() != "optimal":
            raise RuntimeError(f"SCIP status: {mod.getStatus()}")
        return mod.getObjVal()

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    rows = np.repeat(np.arange(m), nnz + 1)
    cols = np.column_stack(
        [idx, np.full(m, prob.n - 1, dtype=np.intp)]).reshape(-1)
    A_ub = coo_matrix((np.ones(m * (nnz + 1)), (rows, cols)),
                      shape=(m, prob.n)).tocsr()
    res = linprog(-prob.b, A_ub=A_ub, b_ub=rw, bounds=(None, None),
                  method=backend)
    if not res.success:
        raise RuntimeError(f"{backend}: {res.message}")
    return -res.fun


def full_primal_lp(prob, backend="highs"):
    """
    The exact primal over all strata -- ground truth.

    Built by transposing ``all_rows()``: column ``j`` has a 1 in each row of
    ``idx[j]`` plus the normalisation row, ``b`` is the dual objective and the
    primal cost is ``sign * rhs``.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    idx, rw = prob.all_rows()
    m, nnz = idx.shape[0], prob.nnz
    cols = np.repeat(np.arange(m), nnz + 1)
    rows = np.column_stack(
        [idx, np.full(m, prob.n - 1, dtype=np.intp)]).reshape(-1)
    A_eq = coo_matrix((np.ones(m * (nnz + 1)), (rows, cols)),
                      shape=(prob.n, m)).tocsr()
    # rhs = sign * c, so the primal cost in this problem's sense is rhs itself
    res = linprog(rw, A_eq=A_eq, b_eq=prob.b, bounds=(0, None), method=backend)
    if not res.success:
        raise RuntimeError(f"{backend}: {res.message}")
    return res.fun


def reduced_aux_lp(prob, backend="highs-ipm"):
    """
    The reduced dual as an LP, each ``max`` lifted to an auxiliary variable.

    The analytical route: the same reduced constraints that ``separate``
    evaluates, written out with the epigraph variables of ``prob.reduced_lp()``
    so that HiGHS or SCIP can take them.  Polynomial in size -- no stratum is
    enumerated -- but with auxiliary columns the oracle route never needs
    (``k^3`` of them for IV).  ``eps > 0`` adds ``tau >= |lam|`` as well.
    """
    lp = prob.reduced_lp()
    tau = lp.aux(prob.n_obs) if prob.eps else np.arange(0)
    for i, t in enumerate(tau):
        lp.leq([i, t], 0.0, [1.0, -1.0])
        lp.leq([i, t], 0.0, [-1.0, -1.0])
    A, r = lp.matrix()
    cost = np.zeros(lp.n_cols)
    cost[:prob.n] = prob.b
    cost[tau] = -prob.eps

    if backend == "scip":
        from pyscipopt import Model, quicksum
        mod = Model()
        mod.hideOutput()
        v = [mod.addVar(lb=None, ub=None) for _ in range(lp.n_cols)]
        for i in range(A.shape[0]):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            mod.addCons(quicksum(float(a) * v[j] for j, a in
                                 zip(A.indices[lo:hi], A.data[lo:hi]))
                        <= float(r[i]))
        mod.setObjective(quicksum(float(cost[j]) * v[j]
                                  for j in np.flatnonzero(cost)), "maximize")
        mod.optimize()
        if mod.getStatus() != "optimal":
            raise RuntimeError(f"SCIP status: {mod.getStatus()}")
        return mod.getObjVal()

    from scipy.optimize import linprog
    res = linprog(-cost, A_ub=A, b_ub=r, bounds=(None, None), method=backend)
    if not res.success:
        raise RuntimeError(f"{backend}: {res.message}")
    return -res.fun


def bounds_full_dual_lp(make_problem, backend="highs-ipm"):
    return (full_dual_lp(make_problem(+1), backend),
            -full_dual_lp(make_problem(-1), backend))


def bounds_reduced_aux_lp(make_problem, backend="highs-ipm"):
    return (reduced_aux_lp(make_problem(+1), backend),
            -reduced_aux_lp(make_problem(-1), backend))


def bounds_full_primal_lp(make_problem, backend="highs"):
    """``(lower, upper)``: min of ``+c`` and ``-min`` of ``-c``."""
    return (full_primal_lp(make_problem(+1), backend),
            -full_primal_lp(make_problem(-1), backend))


def check_oracle(make_problem, trials=3, scale=1.5, seed=0, tol=1e-9):
    """
    Verify ``separate`` against the explicit stratum enumeration.

    For random dual points this checks that (a) the reduction returns the same
    maximum as looping over every row of ``all_rows()``, and (b) each row the
    oracle hands back really attains the value it claims.  Any problem whose
    oracle passes this can be trusted by the solver; one that fails it would
    produce a wrong bound silently.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for sign in (+1, -1):
        prob = make_problem(sign)
        idx, rw = prob.all_rows()
        for _ in range(trials):
            x = rng.standard_normal(prob.n) * scale
            ref = float((x[idx].sum(axis=1) + x[-1] - rw).max())
            h, sel, srhs = prob.separate(x)
            sel = np.asarray(sel).reshape(-1, prob.nnz)
            got = float((x[sel].sum(axis=1) + x[-1]
                         - np.ravel(srhs)).max())
            worst = max(worst, abs(h - ref), abs(got - ref))
    if worst > tol:
        raise AssertionError(f"oracle disagrees with enumeration by {worst:.2e}")
    return worst


def compare(make_problem, routes=("primal", "full-highs", "full-scip",
                                  "full-ipm", "reduced-highs", "reduced-scip",
                                  "reduced"), label="",
            backend="highs-ipm", baseline="primal", oracle_kw=None, **kw):
    """
    Run the same bounds through several routes and print a comparison table.

    ``routes``:
      ``primal``         exact primal over all strata (HiGHS)       -- ground truth
      ``full-highs``     full dual, every stratum a row (HiGHS)     -- no aux vars
      ``full-scip``      the same, SCIP                             -- no aux vars
      ``full-ipm``       the same rows, our interior-point method   -- no aux vars
      ``reduced-highs``  reduced dual, each max an aux var (HiGHS)  -- aux vars
      ``reduced-scip``   the same, SCIP                             -- aux vars
      ``reduced``        reduced dual, rows generated by the oracle -- no aux vars

    ``baseline`` names the route the error columns are measured against.
    """
    from . import main
    oracle_kw = oracle_kw or {}
    probe = make_problem(+1)
    rows, base = [], None

    def run(route, name, solver, fmt, size):
        nonlocal base
        t0 = time.time()
        try:
            lo, up = solver()
        except Exception as exc:
            rows.append((fmt, name, None, None, f"{type(exc).__name__}: {exc}",
                         time.time() - t0))
            return
        dt = time.time() - t0
        rows.append((fmt, name, lo, up, size, dt))
        if base is None and baseline in (route, name):
            base = (lo, up)

    n_strata = None
    if any(r.startswith(("primal", "full")) for r in routes):
        try:
            n_strata = probe.all_rows()[0].shape[0]
        except NotImplementedError:
            n_strata = None

    aux_size = None
    if any(r.startswith("reduced-") for r in routes):
        try:
            lp = probe.reduced_lp()
            aux_size = f"{lp.n_cols - probe.n} aux, {lp.n_rows} rows"
        except NotImplementedError:
            aux_size = "n/a"

    if "primal" in routes:
        run("primal", "HiGHS (primal)",
            lambda: bounds_full_primal_lp(make_problem),
            "exact primal", f"{n_strata} cols")
    if "full-highs" in routes:
        run("full-highs", "HiGHS IPM",
            lambda: bounds_full_dual_lp(make_problem, backend),
            "full dual", f"{n_strata} rows")
    if "full-scip" in routes:
        run("full-scip", "SCIP",
            lambda: bounds_full_dual_lp(make_problem, "scip"),
            "full dual", f"{n_strata} rows")
    if "full-ipm" in routes:
        run("full-ipm", "our IPM",
            lambda: main.solve_bounds(make_problem, full_dual=True, **kw),
            "full dual", f"{n_strata} rows")
    if "reduced-highs" in routes:
        run("reduced-highs", "HiGHS IPM",
            lambda: bounds_reduced_aux_lp(make_problem, backend),
            "reduced dual + aux", aux_size)
    if "reduced-scip" in routes:
        run("reduced-scip", "SCIP",
            lambda: bounds_reduced_aux_lp(make_problem, "scip"),
            "reduced dual + aux", aux_size)
    if "reduced" in routes:
        run("reduced", "our IPM + oracle",
            lambda: main.solve_bounds(make_problem, **{**kw, **oracle_kw}),
            "reduced dual", f"{probe.n_reduced_rows()} max-con")

    head = f"{label}  " if label else ""
    print(f"{head}vars={probe.n}  groups={probe.nnz}  "
          f"strata={n_strata if n_strata is not None else 'n/a'}")
    if base:
        print(f"baseline ({baseline}): [{base[0]:+.10f}, {base[1]:+.10f}]")
    print()
    print("| formulation | solver | lower | upper | err lower | err upper "
          "| size | time |")
    print("|---|---|---|---|---|---|---|---|")
    for fmt, name, lo, up, size, dt in rows:
        if lo is None:
            print(f"| {fmt} | {name} | FAILED | | | | {size} | {dt:.1f}s |")
            continue
        el = f"{lo - base[0]:+.1e}" if base else "-"
        eu = f"{up - base[1]:+.1e}" if base else "-"
        print(f"| {fmt} | {name} | {lo:+.10f} | {up:+.10f} | {el} | {eu} "
              f"| {size} | {dt:.1f}s |")
    return rows