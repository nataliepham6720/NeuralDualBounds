from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# cut pool
# ---------------------------------------------------------------------------

class CutPool:
    """
    A working set of dual rows.

    A row is ``sum_g lam[idx_g] + lam_norm <= rhs``: exactly ``nnz`` unit
    coefficients, so only the indices and the right-hand side are stored.
    """

    def __init__(self, nnz):
        self.nnz = int(nnz)
        self._idx, self._rhs, self._seen = [], [], set()

    def __len__(self):
        return len(self._rhs)

    def add(self, idx, rhs):
        """Add rows (``idx`` of shape ``(m, nnz)``); returns how many were new."""
        idx = np.asarray(idx, dtype=np.intp).reshape(-1, self.nnz)
        rhs = np.ravel(np.asarray(rhs, dtype=float))
        n_new = 0
        for row, r in zip(idx, rhs):
            key = (tuple(sorted(row.tolist())), round(float(r), 12))
            if key in self._seen:
                continue
            self._seen.add(key)
            self._idx.append(row.copy())
            self._rhs.append(float(r))
            n_new += 1
        return n_new

    def arrays(self):
        return (np.asarray(self._idx, dtype=np.intp).reshape(-1, self.nnz),
                np.asarray(self._rhs, dtype=float))


# ---------------------------------------------------------------------------
# smooth |.| for the optional eps-slack L1 term
# ---------------------------------------------------------------------------

def abs_smooth(u, delta):
    """``sqrt(u^2+delta^2) - delta`` and its first two derivatives."""
    if delta <= 0.0:
        return np.abs(u), np.sign(u), np.zeros_like(u)
    root = np.sqrt(u * u + delta * delta)
    return root - delta, u / root, (delta * delta) / root ** 3


# ---------------------------------------------------------------------------
# the problem interface
# ---------------------------------------------------------------------------

class LatentDual(ABC):
    """
    Base class for a latent-stratum dual.  See the module docstring.

    Subclasses must call :meth:`_finalise` once ``n_obs``, ``b``, ``groups``
    and ``span`` are set.
    """

    sign: int = +1
    eps: float = 0.0
    delta: float = 1e-6

    # -- setup ------------------------------------------------------------
    def _finalise(self):
        self.n_obs = int(self.n_obs)
        self.n = self.n_obs + 1
        self.b = np.asarray(self.b, float)
        if self.b.shape != (self.n,):
            raise ValueError(f"b must have shape {(self.n,)}, got {self.b.shape}")
        self.nnz = len(self.groups)

        covered = np.concatenate([np.asarray(g, dtype=np.intp)
                                  for g in self.groups])
        if covered.size != self.n_obs or set(covered.tolist()) != set(range(self.n_obs)):
            raise ValueError("groups must partition range(n_obs); got "
                             f"{covered.size} indices for n_obs={self.n_obs}")

        # A group whose b-entries sum to b_norm gives an exactly flat
        # direction.  Verify rather than assume: a group that fails this is a
        # modelling error, and silently projecting it out would bias the bound.
        flat = []
        for g in self.groups:
            g = np.asarray(g, dtype=np.intp)
            if abs(self.b[g].sum() - self.b[-1]) < 1e-9:
                flat.append(g)
        self.flat_groups = flat
        if flat:
            D = np.zeros((self.n, len(flat)))
            for c, g in enumerate(flat):
                D[g, c] = 1.0
                D[-1, c] = -1.0
            self.gauge_proj = D @ np.linalg.inv(D.T @ D) @ D.T
        else:
            self.gauge_proj = np.zeros((self.n, self.n))

    # -- what a problem provides -----------------------------------------
    @abstractmethod
    def separate(self, x):
        """
        Exact maximum of the dual constraint body over all strata.

        Returns ``(h, idx, rhs)``: ``h`` the largest violation (``> 0`` means
        infeasible), and rows to add to the working set.  Must be exact -- the
        certified bound relies on it.
        """

    def all_rows(self):
        """Every stratum as a row. Optional; only the full-dual route uses it."""
        raise NotImplementedError(f"{type(self).__name__} has no all_rows()")

    # -- generic pieces ---------------------------------------------------
    def objective(self, x, exact_abs=True):
        val = float(self.b @ x)
        if self.eps:
            lam = x[:self.n_obs]
            pen = np.abs(lam).sum() if exact_abs \
                else abs_smooth(lam, self.delta)[0].sum()
            val -= self.eps * float(pen)
        return val

    def violation(self, x):
        return max(0.0, float(self.separate(x)[0]))

    def gauge_optimize(self, x):
        """
        Exact move along the flat directions (only ``eps > 0`` cares).

        Along ``delta_g`` nothing moves but ``sum_{i in g}|lam_i + t|``, which
        is minimised at the group median.
        """
        if not self.eps or not self.flat_groups:
            return x
        out = x.copy()
        for g in self.flat_groups:
            t = -np.median(out[g])
            out[g] += t
            out[-1] -= t
        return out if self.objective(out) > self.objective(x) else x

    def certify(self, x):
        """``(value, x_feasible, violation)`` at a provably feasible point."""
        x = self.gauge_optimize(x)
        v = self.violation(x)
        xc = x.copy()
        xc[-1] -= v
        # lam_norm may now have slack to give back; taking it is always valid
        room = -float(self.separate(xc)[0])
        if room > 0.0:
            xc[-1] += room
        return self.objective(xc, exact_abs=True), xc, v

    # -- log-barrier over a working set ----------------------------------
    def barrier(self, x, idx, rw, mu, rho, x_ref, grad=True, hess=True):
        """``F = -obj + (rho/2)||x-x_ref||^2 - mu*sum log(rhs - A_W x)``."""
        n, nl, nnz = self.n, self.n_obs, self.nnz
        u = rw - (x[idx].sum(axis=1) + x[-1])
        if not np.all(u > 0.0):
            return np.inf, None, None
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
            np.add.at(g, idx.reshape(-1), np.repeat(w, nnz))
            g[-1] += w.sum()
            g -= self.b
            if self.eps:
                g[:nl] += self.eps * abs_smooth(x[:nl], self.delta)[1]
            g += rho * dx

        H = None
        if hess:
            d = mu / (u * u)
            m = len(rw)
            H = np.zeros((n, n))
            rows = np.broadcast_to(idx[:, :, None], (m, nnz, nnz))
            cols = np.broadcast_to(idx[:, None, :], (m, nnz, nnz))
            np.add.at(H, (rows.reshape(-1), cols.reshape(-1)),
                      np.broadcast_to(d[:, None, None],
                                      (m, nnz, nnz)).reshape(-1))
            cross = np.zeros(n)
            np.add.at(cross, idx.reshape(-1), np.repeat(d, nnz))
            H[-1, :] += cross
            H[:, -1] += cross
            H[-1, -1] += float(d.sum())
            H[np.arange(n), np.arange(n)] += rho
            if self.eps:
                H[np.arange(nl), np.arange(nl)] += \
                    self.eps * abs_smooth(x[:nl], self.delta)[2]
        return F, g, H

    def restore_interior(self, x, idx, rw, margin):
        """One ``lam_norm`` shift puts ``x`` strictly inside the working set."""
        if len(rw) == 0:
            return x
        need = float((rw - x[idx].sum(axis=1)).min()) - margin
        if x[-1] <= need:
            return x
        out = x.copy()
        out[-1] = need
        return out


# ---------------------------------------------------------------------------
# Newton direction with the flat directions projected out
# ---------------------------------------------------------------------------

def newton_direction(H, g, gauge_proj, kappa_rel=1e8):
    """
    ``H d = -g`` restricted to the complement of the flat subspace.

    The flat directions carry no information, but would otherwise absorb a
    tiny gradient at near-zero curvature and blow the step up (measured: step
    norms of 1e6 and condition numbers of 1e17 before this projection).
    """
    from scipy.linalg import cho_factor, cho_solve
    n = H.shape[0]
    diag = np.arange(n)
    scale = max(1.0, float(np.abs(np.diag(H)).max()))
    g = g - gauge_proj @ g
    H = H + (kappa_rel * scale) * gauge_proj
    reg = 0.0
    for _ in range(40):
        M = H.copy()
        if reg:
            M[diag, diag] += reg * scale
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

def solve(prob, oracle=None, full_dual=False, rounds=None, mu0=1.0, mu_min=1e-12,
          mu_factor=None, rho0=1.0, rho_min=1e-12, rho_factor=0.5,
          newton_iters=200, newton_tol=1e-14, inner_mu_frac=0.01,
          armijo=1e-4, max_backtrack=60, tol=1e-10, stall_rounds=8,
          seed_rows=None, verbose=0, return_state=False):
    """
    Interior-point solve of ``prob``: ``max b^T x`` over the stratum rows.

    ``oracle`` is any callable ``(x) -> (h, idx, rhs)``; it defaults to the
    problem's closed-form ``prob.separate``.  Pass
    ``latent_dual.z3_oracle.Z3Oracle(prob)`` to have z3 find the worst stratum
    by reasoning over the index arithmetic instead.  Both must be exact -- the
    certified bound relies on it, which is what ``check_oracle`` verifies.

    ``full_dual=True`` seeds the barrier with ``prob.all_rows()`` and never
    calls any oracle -- the comparison baseline.

    Notes on the schedule, both learned from measurement rather than theory:

    * ``newton_iters`` is a cap, and ``inner_mu_frac`` is what normally ends a
      round.  A fixed decrement tolerance is unreachable in floating point, so
      the cap would silently decide the effort per round; a round that stops at
      the cap leaves its subproblem unsolved while ``mu`` anneals anyway, and
      the iterate falls behind the central path.  Tying the inner tolerance to
      ``mu`` fixed a 7.2e-3 error at k=14 in the IV example and made k<=10
      about 2.5x faster at the same accuracy.  ``inner_mu_frac`` is the dial:
      0.1 left the mediation problem at kM=2, kY=4 stuck 5.2e-5 low no matter
      how many rounds it was given (the subproblems were simply not solved
      tightly enough), while 0.01 reaches 3e-11 there and changes nothing
      elsewhere except 1.3-2.5x more time.  Hence the default 0.01.
    * an oracle that returns few rows per round needs many more rounds (it
      needs roughly as many as the model needs rows), so pass a larger
      ``rounds`` and a ``mu_factor`` closer to 1 for such problems.
    """
    if rounds is None:
        rounds = 60
    if mu_factor is None:
        mu_factor = 0.5

    separate = prob.separate if oracle is None else oracle
    pool = CutPool(prob.nnz)
    if full_dual:
        pool.add(*prob.all_rows())
    elif seed_rows is not None:
        pool.add(*seed_rows)
    else:
        pool.add(*prob.seed_rows())

    x = np.zeros(prob.n)
    idx, rw = pool.arrays()
    x = prob.restore_interior(x, idx, rw, margin=1.0)
    x_ref = x.copy()
    mu, rho = float(mu0), float(rho0)
    best_val, best_x, best_viol = prob.certify(x)
    n_newton, best_round, t0 = 0, 0, time.time()

    for rnd in range(rounds):
        if full_dual:
            h, n_new = -np.inf, 0
        else:
            h, sel, srhs = separate(x)
            n_new = pool.add(sel, srhs)
            idx, rw = pool.arrays()
        x = prob.restore_interior(x, idx, rw, margin=max(1e-10, mu))

        inner = 0
        for inner in range(newton_iters):
            F, g, H = prob.barrier(x, idx, rw, mu, rho, x_ref)
            if not np.isfinite(F):
                break
            d = newton_direction(H, g, prob.gauge_proj)
            slope = float(g @ d)
            if slope > 0.0:
                d, slope = -g, -float(g @ g)
            if -0.5 * slope <= max(newton_tol, inner_mu_frac * mu):
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

        val, xc, viol = prob.certify(x)
        if val > best_val + tol:
            best_val, best_x, best_viol, best_round = val, xc, viol, rnd
        elif val > best_val:
            best_val, best_x, best_viol = val, xc, viol
        x_ref = x.copy()

        if verbose >= 2:
            print(f"    round {rnd:4d} |W|={len(pool):6d} (+{n_new:3d}) "
                  f"mu={mu:8.2e} newton={inner:3d} viol={float(h):+10.3e} "
                  f"certified={best_val:+.10f}")
        if best_val > prob.span + 1.0:
            raise RuntimeError(
                f"dual unbounded: certified {best_val:.4g} exceeds the largest "
                f"value any feasible primal attains ({prob.span:.4g}); the "
                f"primal is infeasible. Relax with eps > 0.")
        if mu <= mu_min and rho <= rho_min and rnd - best_round >= stall_rounds:
            break
        mu = max(mu_min, mu * mu_factor)
        rho = max(rho_min, rho * rho_factor)

    info = {"x": best_x, "violation_repaired": best_viol,
            "newton_steps": n_newton, "rounds": rnd + 1, "n_vars": prob.n,
            "n_rows_used": len(pool), "time_s": time.time() - t0,
            "sign": prob.sign, "full_dual": full_dual}
    if verbose >= 1:
        print(f"  [{type(prob).__name__} sign={prob.sign:+d}] "
              f"value={best_val:+.10f} vars={prob.n} rows={len(pool)} "
              f"inner_iter={n_newton} repair={best_viol:.1e} "
              f"{info['time_s']:.2f}s")
    return (best_val, info) if return_state else best_val


def solve_bounds(make_problem, make_oracle=None, verbose=0, **kw):
    """
    ``(lower, upper)`` from one call per sense.

    ``make_problem(sign)`` returns a fresh :class:`LatentDual`.  ``make_oracle``,
    if given, is called as ``make_oracle(prob)`` to build that problem's oracle
    (a z3 oracle has to be constructed per problem, since the sign is baked
    into its symbolic objective).
    """
    out = []
    for sign in (+1, -1):
        prob = make_problem(sign)
        orc = None if make_oracle is None else make_oracle(prob)
        out.append(solve(prob, oracle=orc, verbose=verbose, **kw))
    return out[0], -out[1]