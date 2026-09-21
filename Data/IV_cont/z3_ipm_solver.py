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