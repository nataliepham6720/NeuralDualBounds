"""
reduced_ipm_solver.py
=====================
Interior-point solver for the *reduced* IV-cont dual, with the ``max``
operator evaluated by brute force and **no auxiliary variables**.

The dual
--------
The primal LP over latent response types ``j = (f_T, y0, y1)`` has
``2^k * k^2`` columns.  Its dual has one constraint per latent type,

    sum_z lam[z, f_T(z), y_{f_T(z)}] + lam0  <=  y1 - y0 ,          (*)

so the dual has only ``2k^2 + 1`` variables but exponentially many rows.
Maximising the left side over ``f_T in {0,1}^k`` decouples across ``z``:

    R(y0,y1):  sum_z max( lam[z,0,y0], lam[z,1,y1] ) + lam0 <= s*(y1-y0)

which is the **reduced dual** -- ``k^2`` constraints carrying a ``max``.
With ``s = +1`` for the lower bound and ``s = -1`` for the upper bound,

    V(s) = max   sum_{z,t,y} P[z,t,y] lam[z,t,y] + lam0 - eps*||lam||_1
           s.t.  sum_z max(lam[z,0,y0], lam[z,1,y1]) + lam0 <= s*(y1-y0)

    ATE_lower = V(+1),      ATE_upper = -V(-1)

(the two-sided ``b +- eps`` primal slack becomes the L1 penalty; the
normalisation row is an exact equality so ``lam0`` is unpenalised).

How the ``max`` is handled
--------------------------
``max`` is not an LP expression, so an LP solver can only take the reduced
dual after lifting it with ``k^3`` epigraph variables
``aux[z,y0,y1] >= lam[z,0,y0], lam[z,1,y1]``.  This module does not lift it.
Instead :func:`max_oracle` evaluates the ``max`` **by brute force**, which for
the reduced form costs only

    k^2 pairs (y0,y1)  x  k values of z  =  O(k^3)

per call -- the maximising ``f_T`` is read off per ``z`` independently, so the
``2^k`` maps are never enumerated.  The oracle returns, for every pair, the
exact constraint body *and the latent type that attains it*.  That attaining
type is a single linear row of form (*), and only those rows are handed to the
interior-point method.  The IPM therefore always works on

    max  b^T x - eps*||lam||_1   s.t.   A_W x <= r_W ,     x = (lam, lam0)

a smooth log-barrier problem in the bare ``2k^2 + 1`` dual variables, with
``|W|`` linear rows grown on demand.  Rows are only ever *added*, each is a
row of the true exponential dual, so every subproblem is a relaxation of the
full dual and the loop terminates when the oracle finds nothing violated --
i.e. when the rows already in ``W`` attain the ``max`` at the current point.

    outer approximation:  W <- W + {argmax rows at x}   (<= k^2 per round)
    interior point:       Newton on -b^T x + eps*psi(lam) - mu*sum log(r-Ax)
    oracle:               O(k^3) brute force over the max

Degenerate directions
---------------------
Because ``sum_{t,y} P[z,t,y] = 1`` for every ``z``, the directions

    delta_z :  lam[z,t,y] += 1 for all (t,y),   lam0 -= 1

leave *every* row of form (*) and the ``eps = 0`` objective exactly unchanged:
the dual has ``k`` intrinsically flat directions.  They are projected out of
the Newton system (:attr:`ReducedDual.gauge_basis`) instead of being left to
poison the linear algebra, and the gauge is then fixed optimally in closed
form -- along ``delta_z`` only ``sum_{t,y}|lam[z,t,y]+s|`` varies, minimised
by the per-block median (:meth:`ReducedDual.gauge_optimize`).

Certification
-------------
``lam0`` appears in every constraint with coefficient ``+1`` and in the
objective with coefficient ``+1``, so a point violating the exact constraints
by ``v > 0`` is repaired by ``lam0 <- lam0 - v`` at a cost of exactly ``v``.
:meth:`ReducedDual.certify` applies that repair and re-evaluates with the
exact ``max`` and exact ``|lam|``, so every reported number is the objective
of a provably dual-feasible point: a valid bound, not a solver claim.
``ATE_lower`` can therefore only be under-stated and ``ATE_upper``
over-stated -- errors widen the interval, never narrow it.

``eps`` defaults to ``0``, which solves the dual exactly (agreement with the
lifted reference LP is ~1e-10).  ``eps > 0`` is supported and keeps the dual
bounded when the empirical ``P`` violates the IV model's testable
restrictions, but the ``eps*||lam||_1`` term has negligible curvature beside
``P`` so the barrier does not itself minimise it; the gauge move and the
coordinate polish recover most of it, leaving the certified value up to about
``eps*||lam||_1`` conservative (again, outward).  With ``eps = 0`` an
infeasible ``P`` makes the dual unbounded; that is detected exactly (by
duality the value cannot exceed the span of the ``y`` values) and reported.

Cost
----
``n = 2k^2 + 1`` variables, ``O(k^3)`` per oracle call, ``O(|W| k^2)`` Hessian
assembly, ``O(n^3) = O(k^6)`` per Newton solve.  Polynomial in ``k``, against
``2^k k^2`` columns for the primal.

CLI
---
    python Data/IV_cont/reduced_ipm_solver.py --k 8
    python Data/IV_cont/reduced_ipm_solver.py --k 6 --reference highs-ipm
    python Data/IV_cont/reduced_ipm_solver.py --self-test        # oracle vs 2^k
"""

import argparse
import itertools
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# the brute-force max oracle
# ---------------------------------------------------------------------------

def max_oracle(lam, lam0, rhs):
    """
    Evaluate the reduced constraints exactly, by brute force over the ``max``.

    For every pair ``(y0, y1)`` this returns

        h[y0,y1] = sum_z max(lam[z,0,y0], lam[z,1,y1]) + lam0 - rhs[y0,y1]

    together with the latent type attaining it.  ``h[y0,y1] > 0`` means the
    dual point violates the constraint for that pair; ``h.max()`` is exactly
    the quantity the reference ``brute(lam, lam0)`` computes.

    Cost is ``O(k^3)``: ``k^2`` pairs times ``k`` values of ``z``.  The ``2^k``
    treatment maps are never enumerated -- for a fixed pair the maximising
    ``f_T(z)`` is chosen independently for each ``z``.

    Parameters
    ----------
    lam  : (k, 2, k) dual variables, indexed ``[z, t, y]``
    lam0 : dual variable of the normalisation row
    rhs  : (k, k) right-hand side ``s*(y_{y1} - y_{y0})``

    Returns
    -------
    h   : (k, k)     constraint bodies (positive = violated)
    idx : (k, k, k)  ``idx[y0,y1,z]`` = flat index of ``lam[z, f_T(z), y_.]``
                     for the attaining type, i.e. the support of the linear
                     row that the IPM receives
    """
    k = lam.shape[0]
    a0 = lam[:, 0, :].T[:, None, :]        # [y0, 1,  z]  -> lam[z, 0, y0]
    a1 = lam[:, 1, :].T[None, :, :]        # [1,  y1, z]  -> lam[z, 1, y1]

    take1 = a0 < a1                                        # f_T(z) = 1 there
    h = np.where(take1, a1, a0).sum(axis=2) + lam0 - rhs   # [y0, y1]

    take1 = np.broadcast_to(take1, (k, k, k))
    y0 = np.arange(k)[:, None, None]
    y1 = np.arange(k)[None, :, None]
    z = np.arange(k)[None, None, :]
    # flat index of lam[z, t, y] is z*2k + t*k + y
    idx = z * 2 * k + np.where(take1, k + y1, y0)
    return h, idx


def max_oracle_enumerate(lam, lam0, rhs):
    """
    Same quantity, obtained by enumerating all ``2^k`` treatment maps.

    Exponential and for testing only: it checks the reduction that
    :func:`max_oracle` relies on, i.e. that maximising ``sum_z lam[z,f_T(z),.]``
    over ``f_T in {0,1}^k`` really does decouple into a per-``z`` ``max``.
    This is the same agreement the z3 model checks symbolically.
    """
    k = lam.shape[0]
    h = np.full((k, k), -np.inf)
    for y0 in range(k):
        for y1 in range(k):
            for f_T in itertools.product((0, 1), repeat=k):
                val = sum(lam[z, f_T[z], y1 if f_T[z] else y0] for z in range(k))
                h[y0, y1] = max(h[y0, y1], val)
    return h + lam0 - rhs


# ---------------------------------------------------------------------------
# cut pool: rows of the true exponential dual, added on demand
# ---------------------------------------------------------------------------

class CutPool:
    """
    The working set ``W``.

    Every row of the dual has the form ``sum_z lam[z,t_z,y_z] + lam0 <= r``,
    i.e. exactly ``k`` unit coefficients on ``lam`` plus one on ``lam0``, so a
    row is stored as its ``k`` flat indices and its right-hand side.
    """

    def __init__(self, k):
        self.k = k
        self._idx = []
        self._rhs = []
        self._seen = set()

    def __len__(self):
        return len(self._rhs)

    def add(self, idx, rhs):
        """Add rows (``idx`` of shape ``(m, k)``), skipping duplicates."""
        n_new = 0
        for row, r in zip(np.asarray(idx).reshape(-1, self.k), np.ravel(rhs)):
            key = (tuple(sorted(row.tolist())), float(r))
            if key in self._seen:
                continue
            self._seen.add(key)
            self._idx.append(row.copy())
            self._rhs.append(float(r))
            n_new += 1
        return n_new

    def arrays(self):
        return (np.asarray(self._idx, dtype=np.intp).reshape(-1, self.k),
                np.asarray(self._rhs, dtype=float))


# ---------------------------------------------------------------------------
# the reduced dual
# ---------------------------------------------------------------------------

def _abs_smooth(u, delta):
    """``sqrt(u^2+delta^2) - delta`` and its first two derivatives."""
    if delta <= 0.0:
        return np.abs(u), np.sign(u), np.zeros_like(u)
    root = np.sqrt(u * u + delta * delta)
    return root - delta, u / root, (delta * delta) / root ** 3


class ReducedDual:
    """
    The reduced IV-cont dual, variables ``x = [lam.ravel(), lam0]`` only.

    ``lam`` has shape ``(k, 2, k)`` indexed ``[z, t, y]``, flat index
    ``z*2k + t*k + y``; ``lam0`` is the last entry.  There are no epigraph,
    absolute-value or slack variables anywhere in this class.
    """

    def __init__(self, P, k, sign=+1, eps=1e-6, y_values=None, abs_smooth=1e-6):
        P = np.ascontiguousarray(P, dtype=float)
        if P.shape != (k, 2, k):
            raise ValueError(f"P has shape {P.shape}, expected {(k, 2, k)}")
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 (lower bound) or -1 (upper bound)")

        self.k = k
        self.sign = int(sign)
        self.eps = float(eps)
        self.delta = float(abs_smooth)
        self.P = P
        self.n_lam = 2 * k * k
        self.n = self.n_lam + 1
        self.b = np.concatenate([P.reshape(-1), [1.0]])   # objective gradient

        yv = np.arange(k, dtype=float) if y_values is None \
            else np.asarray(y_values, dtype=float)
        if yv.shape != (k,):
            raise ValueError(f"y_values must have shape {(k,)}")
        self.y_values = yv
        self.rhs = self.sign * (yv[None, :] - yv[:, None])   # [y0, y1]

        # --- the k intrinsically flat directions ---------------------------
        # delta_z: +1 on all of block z, -1 on lam0.  Leaves every dual row
        # and the eps=0 objective exactly invariant.
        D = np.zeros((self.n, k))
        for z in range(k):
            D[:self.n_lam].reshape(k, 2, k, k)[z, :, :, z] = 1.0
        D[-1, :] = -1.0
        self.gauge_basis = D
        self.gauge_proj = D @ np.linalg.inv(D.T @ D) @ D.T   # onto span{delta}

    # -- packing ----------------------------------------------------------
    def unpack(self, x):
        return x[:self.n_lam].reshape(self.k, 2, self.k), float(x[-1])

    # -- exact evaluation --------------------------------------------------
    def oracle(self, x):
        """:func:`max_oracle` at ``x``."""
        lam, lam0 = self.unpack(x)
        return max_oracle(lam, lam0, self.rhs)

    def violation(self, x):
        """Largest violation of the exact reduced constraints (``>= 0``)."""
        return max(0.0, float(self.oracle(x)[0].max()))

    def objective(self, x, exact_abs=True):
        lam, lam0 = self.unpack(x)
        val = float((self.P * lam).sum() + lam0)
        if self.eps:
            pen = np.abs(lam).sum() if exact_abs \
                else _abs_smooth(lam, self.delta)[0].sum()
            val -= self.eps * float(pen)
        return val

    def gauge_optimize(self, x):
        """
        Move along the flat directions to the best gauge, exactly.

        Along ``delta_z`` the constraints and the ``eps = 0`` objective do not
        move at all, so only ``sum_{t,y} |lam[z,t,y] + s|`` matters; it is
        minimised at ``s = -median(lam[z])``.  Feasibility is preserved
        exactly and the objective can only improve.
        """
        if not self.eps:
            return x
        lam, lam0 = self.unpack(x)
        shift = -np.median(lam.reshape(self.k, -1), axis=1)
        out = x.copy()
        out[:self.n_lam].reshape(self.k, 2, self.k)[:] = lam + shift[:, None, None]
        out[-1] = lam0 - shift.sum()
        return out if self.objective(out) > self.objective(x) else x

    def polish(self, x, sweeps=4):
        """
        Exact coordinate ascent on the true objective, preserving feasibility.

        ``x`` must already satisfy the exact reduced constraints.  Each
        coordinate is moved to its exactly optimal value given the others,
        which is available in closed form: ``lam[z,t,y]`` enters only the
        ``k`` constraints of its own ``y``-slice, each through a two-argument
        ``max``, so the largest feasible value is

            hi = min_p ( max(lam[z,t,y], other_p) + slack_p ) ,

        and along that coordinate the objective ``P_i*w - eps*|w|`` is
        concave piecewise linear with its only kink at ``0``.  Lowering a
        coordinate is always feasible (it can only lower a ``max``), so the
        optimum is ``hi`` when ``P_i > eps`` and ``min(0, hi)`` otherwise.

        This is what recovers the ``eps*||lam||_1`` term: the barrier is
        essentially flat in it (``eps`` is tiny beside ``P``), so the
        interior-point iterate sits at the analytic centre of the optimal
        face rather than at its minimum-L1 point.  Cost is ``O(k^3)`` per
        sweep, with the constraint bodies updated incrementally.
        """
        k, eps = self.k, self.eps
        lam, lam0 = self.unpack(x)
        lam = lam.copy()
        h = max_oracle(lam, lam0, self.rhs)[0]        # <= 0 on entry

        def best(P_i, v, hi):
            """Maximiser of P_i*w - eps*|w| over w <= hi."""
            if not eps or P_i > eps:
                return hi
            return min(0.0, hi)

        for _ in range(sweeps):
            moved = 0.0
            # lam0 first: coefficient 1 in the objective and in every row
            room = float(-h.max())
            if room > 0.0:
                lam0 += room
                h += room
                moved = max(moved, room)
            for z in range(k):
                for y0 in range(k):                   # t = 0
                    v = lam[z, 0, y0]
                    cur = np.maximum(v, lam[z, 1, :])         # over y1
                    hi = float((cur - h[y0, :]).min())
                    w = best(self.P[z, 0, y0], v, hi)
                    if w == v:
                        continue
                    gain = (self.P[z, 0, y0] * (w - v)
                            - eps * (abs(w) - abs(v)))
                    if gain <= 0.0:
                        continue
                    h[y0, :] += np.maximum(w, lam[z, 1, :]) - cur
                    lam[z, 0, y0] = w
                    moved = max(moved, abs(w - v))
                for y1 in range(k):                   # t = 1
                    v = lam[z, 1, y1]
                    cur = np.maximum(lam[z, 0, :], v)         # over y0
                    hi = float((cur - h[:, y1]).min())
                    w = best(self.P[z, 1, y1], v, hi)
                    if w == v:
                        continue
                    gain = (self.P[z, 1, y1] * (w - v)
                            - eps * (abs(w) - abs(v)))
                    if gain <= 0.0:
                        continue
                    h[:, y1] += np.maximum(lam[z, 0, :], w) - cur
                    lam[z, 1, y1] = w
                    moved = max(moved, abs(w - v))
            if moved <= 1e-15:
                break

        out = np.empty_like(x)
        out[:self.n_lam] = lam.reshape(-1)
        out[-1] = lam0
        return out

    def certify(self, x):
        """
        Return ``(value, x_feasible, violation)`` for a *provably* dual
        feasible point.

        ``lam0`` is repaired exactly, the gauge is fixed in closed form, and
        the point is polished by exact coordinate ascent -- all three keep the
        exact reduced constraints satisfied, so the value returned is attained
        by a dual feasible point and is a valid bound.
        """
        x = self.gauge_optimize(x)
        v = self.violation(x)
        xc = x.copy()
        xc[-1] -= v
        xp = self.polish(xc)
        if self.objective(xp) > self.objective(xc) and self.violation(xp) <= 1e-9:
            xc = xp
            xc[-1] -= self.violation(xc)
        return self.objective(xc, exact_abs=True), xc, v

    # -- log-barrier subproblem over the working set -----------------------
    def barrier(self, x, idx, rw, mu, rho, x_ref, grad=True, hess=True):
        """
        ``F = -obj + (rho/2)||x-x_ref||^2 - mu*sum_i log(r_i - a_i^T x)``
        for the linear rows in the working set, and its derivatives.

        ``F = inf`` when ``x`` is not strictly feasible for the working set.
        No auxiliary variables enter: ``x`` is ``(lam, lam0)``.
        """
        n, nl = self.n, self.n_lam
        u = rw - (x[idx].sum(axis=1) + x[-1])          # slacks, shape (m,)
        if not np.all(u > 0.0):
            return np.inf, None, None

        lam, _ = self.unpack(x)
        dx = x - x_ref
        F = (-self.objective(x, exact_abs=False)
             + 0.5 * rho * float(dx @ dx)
             - mu * float(np.log(u).sum()))
        if not (grad or hess):
            return F, None, None

        g = None
        if grad:
            w = mu / u
            g = np.zeros(n)
            np.add.at(g, idx.reshape(-1), np.repeat(w, self.k))
            g[-1] += w.sum()
            g -= self.b
            if self.eps:
                g[:nl] += self.eps * _abs_smooth(lam, self.delta)[1].reshape(-1)
            g += rho * dx

        H = None
        if hess:
            d = mu / (u * u)
            H = np.zeros((n, n))
            rows = np.broadcast_to(idx[:, :, None], (len(rw), self.k, self.k))
            cols = np.broadcast_to(idx[:, None, :], (len(rw), self.k, self.k))
            np.add.at(H, (rows.reshape(-1), cols.reshape(-1)),
                      np.broadcast_to(d[:, None, None],
                                      (len(rw), self.k, self.k)).reshape(-1))
            cross = np.zeros(n)
            np.add.at(cross, idx.reshape(-1), np.repeat(d, self.k))
            H[-1, :] += cross
            H[:, -1] += cross
            H[-1, -1] += float(d.sum())
            H[np.arange(n), np.arange(n)] += rho
            if self.eps:
                H[np.arange(nl), np.arange(nl)] += \
                    self.eps * _abs_smooth(lam, self.delta)[2].reshape(-1)

        return F, g, H

    def restore_interior(self, x, idx, rw, margin):
        """
        Push ``x`` back strictly inside the working set by lowering ``lam0``.

        Newly added rows are violated by construction; ``lam0`` sits in every
        row with coefficient ``+1``, so one shift restores strict feasibility
        for all of them at once, at a cost of exactly that shift.
        """
        if len(rw) == 0:
            return x
        head = rw - x[idx].sum(axis=1)
        need = float(head.min()) - margin
        if x[-1] <= need:
            return x
        out = x.copy()
        out[-1] = need
        return out


# ---------------------------------------------------------------------------
# Newton direction, with the flat directions projected out
# ---------------------------------------------------------------------------

def _newton_direction(H, g, gauge_proj, kappa_rel=1e8):
    """
    Solve ``H d = -g`` restricted to the complement of the flat subspace.

    The flat directions carry no information (the constraints and the
    ``eps = 0`` objective are invariant along them) but would otherwise absorb
    the tiny ``eps`` gradient at near-zero curvature and blow the step up, so
    the gradient is projected off them and the Hessian is stiffened along
    them.  The gauge itself is set exactly by
    :meth:`ReducedDual.gauge_optimize`.
    """
    from scipy.linalg import cho_factor, cho_solve
    n = H.shape[0]
    idx = np.arange(n)
    scale = max(1.0, float(np.abs(np.diag(H)).max()))

    g = g - gauge_proj @ g
    H = H + (kappa_rel * scale) * gauge_proj

    reg = 0.0
    for _ in range(40):
        M = H.copy()
        if reg:
            M[idx, idx] += reg * scale
        try:
            d = cho_solve(cho_factor(M, lower=True, check_finite=False), -g,
                          check_finite=False)
            if np.all(np.isfinite(d)):
                return d
        except Exception:
            pass
        reg = 1e-12 if reg == 0.0 else reg * 100.0
    return -g


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------

def solve_reduced_dual_ipm(P, k, eps=0.0, sign=+1, y_values=None,
                           rounds=60, mu0=1.0, mu_min=1e-12, mu_factor=0.5,
                           rho0=1.0, rho_min=1e-12, rho_factor=0.5,
                           newton_iters=50, newton_tol=1e-14,
                           armijo=1e-4, max_backtrack=60, abs_smooth=1e-6,
                           tol=1e-10, stall_rounds=6, bound_limit=None,
                           verbose=0, return_state=False):
    """
    Interior-point solve of the reduced IV-cont dual.

    The ``max`` is evaluated by brute force in ``O(k^3)`` per round
    (:func:`max_oracle`) and only the attaining latent types are added to the
    interior-point subproblem as linear rows -- the ``2^k k^2`` rows of the
    full dual are never enumerated, and no auxiliary variable is introduced.

    Returns the certified value of ``V(sign)``; with ``return_state=True``
    returns ``(value, info)``.
    """
    prob = ReducedDual(P, k, sign=sign, eps=eps, y_values=y_values,
                       abs_smooth=abs_smooth)
    pool = CutPool(k)

    # When the primal is feasible its objective c_j = s*(y1-y0) lies in
    # [-span, span], so by duality V(sign) does too.  A certified value above
    # that can only mean the dual is unbounded, i.e. the primal is infeasible.
    span = float(prob.y_values.max() - prob.y_values.min())
    limit = span + 1.0 if bound_limit is None else float(bound_limit)

    # Seed with the two constant treatment maps, f_T == 0 and f_T == 1, for
    # every pair: 2k^2 rows that touch every dual coordinate.
    z = np.arange(k)[None, None, :]
    y0 = np.arange(k)[:, None, None]
    y1 = np.arange(k)[None, :, None]
    seed0 = np.broadcast_to(z * 2 * k + y0, (k, k, k))
    seed1 = np.broadcast_to(z * 2 * k + k + y1, (k, k, k))
    pool.add(seed0.reshape(-1, k), np.repeat(prob.rhs.reshape(-1), 1))
    pool.add(seed1.reshape(-1, k), np.repeat(prob.rhs.reshape(-1), 1))

    x = np.zeros(prob.n)
    idx, rw = pool.arrays()
    x = prob.restore_interior(x, idx, rw, margin=1.0)
    x_ref = x.copy()

    mu, rho = float(mu0), float(rho0)
    best_val, best_x, best_viol = prob.certify(x)
    n_newton, best_round, t0 = 0, 0, time.time()

    for rnd in range(rounds):
        # ---- brute-force max oracle -> new linear rows -------------------
        h, sel = prob.oracle(x)
        n_new = pool.add(sel.reshape(-1, k), prob.rhs.reshape(-1))
        idx, rw = pool.arrays()
        x = prob.restore_interior(x, idx, rw, margin=max(1e-10, mu))

        # ---- interior-point solve of the relaxation ----------------------
        inner = 0
        for inner in range(newton_iters):
            F, g, H = prob.barrier(x, idx, rw, mu, rho, x_ref)
            if not np.isfinite(F):
                break
            d = _newton_direction(H, g, prob.gauge_proj)
            slope = float(g @ d)
            if slope > 0.0:
                d, slope = -g, -float(g @ g)
            if -0.5 * slope <= newton_tol:
                break
            alpha, ok = 1.0, False
            for _ in range(max_backtrack):
                Fn, _, _ = prob.barrier(x + alpha * d, idx, rw, mu, rho, x_ref,
                                        grad=False, hess=False)
                if np.isfinite(Fn) and Fn <= F + armijo * alpha * slope:
                    ok = True
                    break
                alpha *= 0.5
            if not ok:
                break
            x = x + alpha * d
            n_newton += 1

        # ---- certify -----------------------------------------------------
        val, xc, viol = prob.certify(x)
        if val > best_val + tol:
            best_val, best_x, best_viol, best_round = val, xc, viol, rnd
        elif val > best_val:
            best_val, best_x, best_viol = val, xc, viol
        x_ref = x.copy()

        # converged: the oracle found nothing to add and nothing violated, and
        # the certified value has stopped moving
        converged = (n_new == 0 and float(h.max()) <= tol
                     and rnd - best_round >= stall_rounds)

        if verbose >= 2:
            print(f"    round {rnd:3d} |W|={len(pool):5d} (+{n_new:3d}) "
                  f"mu={mu:8.2e} rho={rho:8.2e} newton={inner:3d} "
                  f"viol={float(h.max()):+9.2e} certified={best_val:+.10f}")

        if best_val > limit:
            raise RuntimeError(
                f"reduced dual is unbounded: the certified value {best_val:.4g} "
                f"exceeds the largest value any feasible primal can attain "
                f"({span:.4g}). The primal is infeasible -- the empirical P "
                f"violates the IV model's testable restrictions at k={k}. "
                f"Relax the observational constraints with a larger eps "
                f"(currently {eps:g}), or change the discretization.")
        if converged or (mu <= mu_min and rho <= rho_min
                         and rnd - best_round >= stall_rounds):
            break
        mu = max(mu_min, mu * mu_factor)
        rho = max(rho_min, rho * rho_factor)

    info = {
        "x": best_x,
        "lam": best_x[:prob.n_lam].reshape(k, 2, k),
        "lam0": float(best_x[-1]),
        "violation_repaired": best_viol,
        "newton_steps": n_newton,
        "rounds": rnd + 1,
        "n_vars": prob.n,
        "n_rows_used": len(pool),
        "n_rows_full_dual": 2 ** k * k * k,
        "time_s": time.time() - t0,
        "sign": sign,
    }
    if verbose >= 1:
        print(f"  [ipm sign={sign:+d}] value={best_val:+.10f}  "
              f"vars={prob.n}  rows used={len(pool)} of {2 ** k * k * k}  "
              f"newton={n_newton}  repair={best_viol:.1e}  "
              f"{info['time_s']:.2f}s")
    return (best_val, info) if return_state else best_val


def solve_bounds_ipm(P, k, eps=0.0, y_values=None, verbose=0, **kw):
    """ATE ``(lower, upper)`` from the reduced max-form dual via the IPM."""
    if verbose:
        print("Solving lower bound (reduced dual, brute-force max, no aux vars)...")
    lower = solve_reduced_dual_ipm(P, k, eps=eps, sign=+1, y_values=y_values,
                                   verbose=verbose, **kw)
    if verbose:
        print("Solving upper bound (reduced dual, brute-force max, no aux vars)...")
    upper = -solve_reduced_dual_ipm(P, k, eps=eps, sign=-1, y_values=y_values,
                                    verbose=verbose, **kw)
    return lower, upper


# ---------------------------------------------------------------------------
# reference: the same reduced dual, but lifted with epigraph variables
# ---------------------------------------------------------------------------

def solve_reduced_dual_reference(P, k, eps=1e-6, sign=+1, y_values=None,
                                 backend="highs-ipm"):
    """
    Cross-check only -- this *does* introduce auxiliary variables.

    ``max`` is lifted to ``aux[z,y0,y1] >= lam[z,0,y0], lam[z,1,y1]`` and
    ``|lam|`` to ``tau >= +-lam``, giving the plain LP that HiGHS or SCIP can
    take.  ``backend`` is ``"highs"``/``"highs-ipm"``/``"highs-ds"`` (scipy's
    HiGHS, ``-ipm`` selecting its interior-point solver) or ``"scip"``.
    """
    yv = np.arange(k, dtype=float) if y_values is None \
        else np.asarray(y_values, dtype=float)
    rhs = sign * (yv[None, :] - yv[:, None])
    P = np.asarray(P, dtype=float)

    if backend == "scip":
        from pyscipopt import Model
        m = Model()
        m.hideOutput()
        lam = {(z, t, y): m.addVar(lb=None, ub=None)
               for z in range(k) for t in (0, 1) for y in range(k)}
        tau = {key: m.addVar(lb=0, ub=None) for key in lam}
        for key, v in lam.items():
            m.addCons(tau[key] >= v)
            m.addCons(tau[key] >= -v)
        lam0 = m.addVar(lb=None, ub=None)
        aux = {}
        for z in range(k):
            for y0 in range(k):
                for y1 in range(k):
                    a = m.addVar(lb=None, ub=None)
                    m.addCons(a >= lam[z, 0, y0])
                    m.addCons(a >= lam[z, 1, y1])
                    aux[z, y0, y1] = a
        for y0 in range(k):
            for y1 in range(k):
                m.addCons(sum(aux[z, y0, y1] for z in range(k)) + lam0
                          <= float(rhs[y0, y1]))
        m.setObjective(sum(P[z, t, y] * lam[z, t, y]
                           for z in range(k) for t in (0, 1) for y in range(k))
                       + lam0 - eps * sum(tau.values()), "maximize")
        m.optimize()
        if m.getStatus() != "optimal":
            raise RuntimeError(f"SCIP status: {m.getStatus()}")
        return m.getObjVal()

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    nl = 2 * k * k
    n = nl + 1 + k ** 3 + nl
    o_aux, o_tau = nl + 1, nl + 1 + k ** 3

    def L(z, t, y):
        return z * 2 * k + t * k + y

    def AUX(z, y0, y1):
        return o_aux + (z * k + y0) * k + y1

    cobj = np.zeros(n)
    cobj[:nl] = -P.reshape(-1)
    cobj[nl] = -1.0
    cobj[o_tau:] = eps

    rows, cols, vals, b_ub, row = [], [], [], [], 0

    def put(entries, rhs_val):
        nonlocal row
        for j, v in entries:
            rows.append(row)
            cols.append(j)
            vals.append(v)
        b_ub.append(rhs_val)
        row += 1

    for z in range(k):
        for y0 in range(k):
            for y1 in range(k):
                put([(L(z, 0, y0), 1.0), (AUX(z, y0, y1), -1.0)], 0.0)
                put([(L(z, 1, y1), 1.0), (AUX(z, y0, y1), -1.0)], 0.0)
    for y0 in range(k):
        for y1 in range(k):
            put([(AUX(z, y0, y1), 1.0) for z in range(k)] + [(nl, 1.0)],
                float(rhs[y0, y1]))
    for i in range(nl):
        put([(i, 1.0), (o_tau + i, -1.0)], 0.0)
        put([(i, -1.0), (o_tau + i, -1.0)], 0.0)

    A_ub = coo_matrix((vals, (rows, cols)), shape=(row, n)).tocsr()
    res = linprog(cobj, A_ub=A_ub, b_ub=np.array(b_ub), bounds=(None, None),
                  method=backend)
    if not res.success:
        raise RuntimeError(f"{backend}: {res.message}")
    return -res.fun


def solve_bounds_reference(P, k, eps=1e-6, y_values=None, backend="highs-ipm"):
    """ATE ``(lower, upper)`` from the epigraph LP (auxiliary variables)."""
    lower = solve_reduced_dual_reference(P, k, eps=eps, sign=+1,
                                         y_values=y_values, backend=backend)
    upper = -solve_reduced_dual_reference(P, k, eps=eps, sign=-1,
                                          y_values=y_values, backend=backend)
    return lower, upper


# ---------------------------------------------------------------------------
# self-test: the O(k^3) oracle against the 2^k enumeration
# ---------------------------------------------------------------------------

def self_test(ks=(2, 3, 4, 5), trials=5, seed=0):
    """Check ``max_oracle`` against brute force over all ``2^k`` maps."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for k in ks:
        for _ in range(trials):
            lam = rng.standard_normal((k, 2, k)) * rng.uniform(0.2, 3.0)
            lam0 = float(rng.standard_normal())
            rhs = np.arange(k)[None, :] - np.arange(k)[:, None]
            h_fast, idx = max_oracle(lam, lam0, rhs.astype(float))
            h_slow = max_oracle_enumerate(lam, lam0, rhs.astype(float))
            err = float(np.abs(h_fast - h_slow).max())
            # the returned support must reproduce the same value
            h_sel = lam.reshape(-1)[idx].sum(axis=2) + lam0 - rhs
            err = max(err, float(np.abs(h_sel - h_slow).max()))
            worst = max(worst, err)
        print(f"  k={k}: O(k^3) oracle vs 2^k={2 ** k} enumeration -> "
              f"max |diff| = {worst:.2e}")
    assert worst < 1e-12, f"oracle mismatch {worst}"
    print("  oracle matches the exponential enumeration")
    return worst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_P(k, data="preload", n=10000, lam=0.5, P_path=None):
    if data == "preload":
        path = P_path or os.path.join(_HERE, f"P{k}.npy")
        P = np.load(path)
        if P.shape != (k, 2, k):
            raise ValueError(f"{path} has shape {P.shape}, expected {(k, 2, k)}")
        return P, f"preload({os.path.basename(path)})"
    from LP_construction import generate_data_IV, empirical_distribution_IV
    return empirical_distribution_IV(generate_data_IV(n, lam), k), f"generate(n={n})"


def main():
    p = argparse.ArgumentParser(
        description="Interior-point solve of the reduced IV-cont dual with a "
                    "brute-force max oracle (no auxiliary variables).")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--data", choices=["preload", "generate"], default="preload")
    p.add_argument("--P-path", default=None)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.0,
                   help="slack on the observational constraints; 0 solves "
                        "the dual exactly, >0 widens the bounds by about "
                        "eps*||lam||_1")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--mu0", type=float, default=1.0)
    p.add_argument("--mu-factor", type=float, default=0.5)
    p.add_argument("--rho0", type=float, default=1.0)
    p.add_argument("--reference", choices=["none", "highs", "highs-ipm",
                                           "highs-ds", "scip"], default="none")
    p.add_argument("--self-test", action="store_true",
                   help="check the O(k^3) max oracle against the 2^k enumeration")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    if args.self_test:
        print("Self-test of the brute-force max oracle:")
        self_test()
        print()

    np.random.seed(args.seed)
    P, src = load_P(args.k, data=args.data, n=args.n, lam=args.lam,
                    P_path=args.P_path)
    k = args.k
    print(f"k={k}  P from {src}  eps={args.eps}")
    print(f"reduced dual: {2 * k ** 2 + 1} variables, {k ** 2} max-constraints, "
          f"oracle cost {k ** 3} (full dual has {2 ** k * k ** 2} rows)")

    t0 = time.time()
    lower, upper = solve_bounds_ipm(P, k, eps=args.eps, rounds=args.rounds,
                                    mu0=args.mu0, mu_factor=args.mu_factor,
                                    rho0=args.rho0, verbose=args.verbose)
    dt = time.time() - t0

    print("\n==============================")
    print(f"ATE LOWER: {lower:.10f}")
    print(f"ATE UPPER: {upper:.10f}")
    print("TRUE ATE = 3")
    print("==============================")
    print(f"Time taken (reduced IPM, no aux vars): {dt:.2f}s")

    if args.reference != "none":
        t0 = time.time()
        rl, ru = solve_bounds_reference(P, k, eps=args.eps,
                                        backend=args.reference)
        dtr = time.time() - t0
        print(f"\nreference ({args.reference}, epigraph LP with aux vars):")
        print(f"  ATE LOWER: {rl:.10f}   (diff {lower - rl:+.2e})")
        print(f"  ATE UPPER: {ru:.10f}   (diff {upper - ru:+.2e})")
        print(f"  Time taken: {dtr:.2f}s")


if __name__ == "__main__":
    main()
