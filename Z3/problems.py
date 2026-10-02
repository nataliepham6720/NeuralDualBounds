import itertools

import numpy as np

from .main import LatentDual, LiftedLP


# ===========================================================================
# 1. continuous IV
# ===========================================================================

class IVCont(LatentDual):
    """
    Dual of the continuous-IV ATE LP (``Data/IV_cont``).

    Dual variables ``lam[z,t,y]`` (flat index ``z*2k + t*k + y``) plus
    ``lam_norm``.  A stratum is ``(f_T, y0, y1)`` with ``f_T in {0,1}^k``::

        sum_z lam[z, f_T(z), y_{f_T(z)}] + lam_norm <= s*(yv[y1] - yv[y0])

    Groups: one per ``z`` (every stratum hits exactly one ``(t,y)`` per ``z``).

    ``yv`` (``y_values``) are the outcome values of the ``k`` Y-bins.  They
    default to the bin *indices* ``0..k-1``, which bounds
    ``E[bin(Y1) - bin(Y0)]`` -- not the ATE.  Pass the bin centers to bound
    the ATE on the outcome scale; ``Z3/run_iv_cont.py`` recovers them for the
    stored ``Data/IV_cont/P{k}.npy``.
    """

    def __init__(self, P, k, sign=+1, eps=0.0, y_values=None):
        P = np.ascontiguousarray(P, float)
        if P.shape != (k, 2, k):
            raise ValueError(f"P must be {(k, 2, k)}, got {P.shape}")
        self.k, self.sign, self.eps = k, int(sign), float(eps)
        self.yv = np.arange(k, dtype=float) if y_values is None \
            else np.asarray(y_values, float)
        self.P = P
        self.n_obs = 2 * k * k
        self.b = np.concatenate([P.reshape(-1), [1.0]])
        self.groups = [np.arange(z * 2 * k, (z + 1) * 2 * k) for z in range(k)]
        self.rhs = self.sign * (self.yv[None, :] - self.yv[:, None])
        self.span = float(np.abs(self.rhs).max())
        self._finalise()

    def n_reduced_rows(self):
        return self.k * self.k

    def separate(self, x):
        k = self.k
        lam = x[:self.n_obs].reshape(k, 2, k)
        a0 = lam[:, 0, :].T[:, None, :]          # [y0, 1,  z]
        a1 = lam[:, 1, :].T[None, :, :]          # [1,  y1, z]
        take1 = a0 < a1
        h = np.where(take1, a1, a0).sum(axis=2) + x[-1] - self.rhs
        take1 = np.broadcast_to(take1, (k, k, k))
        y0 = np.arange(k)[:, None, None]
        y1 = np.arange(k)[None, :, None]
        z = np.arange(k)[None, None, :]
        idx = z * 2 * k + np.where(take1, k + y1, y0)
        return float(h.max()), idx.reshape(-1, k), self.rhs.reshape(-1)

    def seed_rows(self):
        """Both constant treatment maps at every pair: 2k^2 rows."""
        k = self.k
        z = np.arange(k)[None, None, :]
        y0 = np.arange(k)[:, None, None]
        y1 = np.arange(k)[None, :, None]
        i0 = np.broadcast_to(z * 2 * k + y0, (k, k, k)).reshape(-1, k)
        i1 = np.broadcast_to(z * 2 * k + k + y1, (k, k, k)).reshape(-1, k)
        r = self.rhs.reshape(-1)
        return np.vstack([i0, i1]), np.concatenate([r, r])

    # -- z3: the same maximisation, by logic over the index arithmetic ----
    def z3_config_keys(self):
        """One configuration per ``(y0, y1)``; z3 then maximises over ``f_T``."""
        return [(y0, y1) for y0 in range(self.k) for y1 in range(self.k)]

    def z3_pin(self, z3, key):
        y0, y1 = key
        return [self._z3y(0) == y0, self._z3y(1) == y1]

    def z3_model(self, z3, opt, lam_arr, cost_0):
        k = self.k
        t = z3.Function("t", z3.IntSort(), z3.IntSort())
        y = z3.Function("y", z3.IntSort(), z3.IntSort())
        self._z3t, self._z3y = t, y
        for z_ in range(k):
            opt.add(t(z_) >= 0, t(z_) < 2)
        for t_ in (0, 1):
            opt.add(y(t_) >= 0, y(t_) < k)
        # the cost is read out of a second Array so that arbitrary y_values and
        # the sign flip need no change to the encoding
        yv = z3.Array("yv", z3.IntSort(), z3.RealSort())
        cont = z3.K(z3.IntSort(), z3.RealVal(0))
        for i, v in enumerate(self.yv):
            cont = z3.Store(cont, i, z3.RealVal(repr(float(v))))
        opt.add(yv == cont)
        sgn = z3.RealVal(repr(float(self.sign)))
        cost = cost_0 - sgn * (z3.Select(yv, y(1)) - z3.Select(yv, y(0)))
        for z_ in range(k):
            tz = t(z_)
            cost = cost + z3.Select(lam_arr, z_ * 2 * k + tz * k + y(tz))

        def read(model, key):
            y0, y1 = key
            tsel = [model.eval(t(z_), model_completion=True).as_long()
                    for z_ in range(k)]
            idx = [z_ * 2 * k + tz * k + (y1 if tz else y0)
                   for z_, tz in enumerate(tsel)]
            return np.array(idx, dtype=np.intp), float(self.rhs[y0, y1])

        return cost, read

    # -- the same reduction as an LP, each max lifted ----------------------
    def reduced_lp(self):
        """
        ``k^3`` epigraph variables, one per ``max`` in ``separate``::

            aux[z,y0,y1] >= lam[z,0,y0],    aux[z,y0,y1] >= lam[z,1,y1]
            sum_z aux[z,y0,y1] + lam_norm <= s*(yv[y1] - yv[y0])

        ``2k^3 + k^2`` rows over ``2k^2 + 1 + k^3`` columns.
        """
        k = self.k
        lp = LiftedLP(self.n)
        aux = lp.aux(k ** 3).reshape(k, k, k)           # [z, y0, y1]
        for z in range(k):
            for y0 in range(k):
                for y1 in range(k):
                    lp.epigraph(aux[z, y0, y1], [z * 2 * k + y0])
                    lp.epigraph(aux[z, y0, y1], [z * 2 * k + k + y1])
        for y0 in range(k):
            for y1 in range(k):
                lp.leq(list(aux[:, y0, y1]) + [self.n - 1], self.rhs[y0, y1])
        return lp

    def all_rows(self):
        k = self.k
        T = np.array(list(itertools.product((0, 1), repeat=k)), dtype=np.intp)
        tz = T[:, None, None, :]
        y0 = np.arange(k, dtype=np.intp)[None, :, None, None]
        y1 = np.arange(k, dtype=np.intp)[None, None, :, None]
        z = np.arange(k, dtype=np.intp)[None, None, None, :]
        idx = z * 2 * k + tz * k + np.where(tz == 1, y1, y0)
        rhs = np.broadcast_to(self.rhs[None, :, :], (T.shape[0], k, k))
        return (np.ascontiguousarray(idx).reshape(-1, k),
                np.ascontiguousarray(rhs).reshape(-1))


# ===========================================================================
# 2. education vs voting
# ===========================================================================

class EduVsVoting(LatentDual):
    """
    Dual of the education--voting ATE LP (``Data/Edu_vs_Voting``).

    Dual variables ``lam[i,d,y]`` over the supported x-bins (flat index
    ``i*2ky + d*ky + y``) plus ``lam_norm``.  A stratum is ``(d, y)`` with
    ``d in {0,1}^nx`` and ``y`` indexed by ``(x,d)``::

        sum_i lam[i, d_i, y_{i,d_i}] + lam_norm
            <= s * sum_i px[i] * (yc[y_{i,1}] - yc[y_{i,0}])

    Groups: one per supported ``x``.  Both maxima decouple, so the reduced
    form is a single constraint.
    """

    def __init__(self, P, kx, ky, y_bins, sign=+1, eps=0.0):
        P = np.asarray(P, float)
        px = P.sum(axis=(1, 2))
        self.xs = [x for x in range(kx) if px[x] > 0]
        self.nx, self.kx, self.ky = len(self.xs), kx, ky
        self.sign, self.eps = int(sign), float(eps)
        p_cond = np.zeros_like(P)
        p_cond[px > 0] = P[px > 0] / px[px > 0][:, None, None]
        self.px = px[self.xs]
        self.p_cond = p_cond[self.xs]
        yb = np.asarray(y_bins, float)
        self.yc = (yb[:-1] + yb[1:]) / 2
        self.n_obs = 2 * self.nx * ky
        self.b = np.concatenate([self.p_cond.reshape(-1), [1.0]])
        self.groups = [np.arange(i * 2 * ky, (i + 1) * 2 * ky)
                       for i in range(self.nx)]
        self.C = (self.sign * self.px[:, None, None]
                  * (self.yc[None, None, :] - self.yc[None, :, None]))
        self.span = float(self.px.sum() * (self.yc.max() - self.yc.min()))
        self._finalise()

    def n_reduced_rows(self):
        return 1

    def _grids(self, x):
        lam = x[:self.n_obs].reshape(self.nx, 2, self.ky)
        M = np.maximum(lam[:, 0, :][:, :, None],
                       lam[:, 1, :][:, None, :]) - self.C
        take1 = lam[:, 0, :][:, :, None] < lam[:, 1, :][:, None, :]
        return M, take1

    def _row(self, choice, take1):
        ky, nx = self.ky, self.nx
        y0, y1 = choice // ky, choice % ky
        d = take1[np.arange(nx), y0, y1].astype(np.intp)
        idx = np.arange(nx) * 2 * ky + d * ky + np.where(d == 1, y1, y0)
        rhs = float((self.sign * self.px * (self.yc[y1] - self.yc[y0])).sum())
        return idx, rhs

    def separate(self, x, runners_up=True):
        M, take1 = self._grids(x)
        flat = M.reshape(self.nx, -1)
        order = np.argsort(-flat, axis=1)
        best = order[:, 0]
        h = float(x[-1] + flat[np.arange(self.nx), best].sum())
        rows, rhss = [], []
        i0, r0 = self._row(best, take1)
        rows.append(i0)
        rhss.append(r0)
        if runners_up and flat.shape[1] > 1:
            # the reduction collapses to ONE constraint, so the argmax alone
            # would add a single row per round and starve the outer
            # approximation; the per-x runners-up give nx+1 rows instead.
            for i in range(self.nx):
                alt = best.copy()
                alt[i] = order[i, 1]
                ii, rr = self._row(alt, take1)
                rows.append(ii)
                rhss.append(rr)
        return h, np.array(rows, dtype=np.intp), np.array(rhss, float)

    def seed_rows(self):
        ky, nx = self.ky, self.nx
        base = np.arange(nx) * 2 * ky
        idx, rhs = [], []
        for dv in (0, 1):
            for yv in range(ky):
                idx.append(base + dv * ky + yv)
                rhs.append(0.0)      # y_{i,0}=y_{i,1}=yv  =>  cost 0
        return np.array(idx, dtype=np.intp), np.array(rhs, float)

    # -- z3 ---------------------------------------------------------------
    def z3_config_keys(self):
        """Everything decouples, so there is a single configuration."""
        return [None]

    def z3_pin(self, z3, key):
        return []

    def z3_model(self, z3, opt, lam_arr, cost_0):
        nx, ky = self.nx, self.ky
        d = z3.Function("d", z3.IntSort(), z3.IntSort())
        y0 = z3.Function("y0", z3.IntSort(), z3.IntSort())
        y1 = z3.Function("y1", z3.IntSort(), z3.IntSort())
        for i in range(nx):
            opt.add(d(i) >= 0, d(i) < 2)
            opt.add(y0(i) >= 0, y0(i) < ky)
            opt.add(y1(i) >= 0, y1(i) < ky)
        yc = z3.Array("yc", z3.IntSort(), z3.RealSort())
        cont = z3.K(z3.IntSort(), z3.RealVal(0))
        for i, v in enumerate(self.yc):
            cont = z3.Store(cont, i, z3.RealVal(repr(float(v))))
        opt.add(yc == cont)

        cost = cost_0
        for i in range(nx):
            di = d(i)
            yi = z3.If(di == 1, y1(i), y0(i))
            cost = cost + z3.Select(lam_arr, i * 2 * ky + di * ky + yi)
            spx = z3.RealVal(repr(float(self.sign * self.px[i])))
            cost = cost - spx * (z3.Select(yc, y1(i)) - z3.Select(yc, y0(i)))

        def read(model, key):
            dd = np.array([model.eval(d(i), model_completion=True).as_long()
                           for i in range(nx)], dtype=np.intp)
            a0 = np.array([model.eval(y0(i), model_completion=True).as_long()
                           for i in range(nx)], dtype=np.intp)
            a1 = np.array([model.eval(y1(i), model_completion=True).as_long()
                           for i in range(nx)], dtype=np.intp)
            idx = (np.arange(nx) * 2 * ky + dd * ky
                   + np.where(dd == 1, a1, a0))
            rhs = float((self.sign * self.px
                         * (self.yc[a1] - self.yc[a0])).sum())
            return idx, rhs

        return cost, read

    def reduced_lp(self):
        """
        One epigraph variable per supported ``x``; the arm ``d`` does not
        take is free, so its ``y`` is maximised out in closed form::

            t_i >= lam[i,0,y] - min_y' C[i,y,y']
            t_i >= lam[i,1,y] - min_y' C[i,y',y]
            sum_i t_i + lam_norm <= 0

        ``2 nx ky + 1`` rows over ``2 nx ky + 1 + nx`` columns.
        """
        nx, ky = self.nx, self.ky
        lp = LiftedLP(self.n)
        t = lp.aux(nx)
        lo0 = self.C.min(axis=2)             # [i, y0]: best y1 for arm 0
        lo1 = self.C.min(axis=1)             # [i, y1]: best y0 for arm 1
        for i in range(nx):
            for y in range(ky):
                lp.epigraph(t[i], [i * 2 * ky + y], -lo0[i, y])
                lp.epigraph(t[i], [i * 2 * ky + ky + y], -lo1[i, y])
        lp.leq(list(t) + [self.n - 1], 0.0)
        return lp

    def all_rows(self):
        nx, ky = self.nx, self.ky
        base = np.arange(nx) * 2 * ky
        idx, rhs = [], []
        for d_vec in itertools.product((0, 1), repeat=nx):
            d = np.array(d_vec, dtype=np.intp)
            for y_vec in itertools.product(range(ky), repeat=2 * nx):
                y = np.array(y_vec, dtype=np.intp).reshape(nx, 2)
                idx.append(base + d * ky + y[np.arange(nx), d])
                rhs.append(self.sign * float(
                    (self.px * (self.yc[y[:, 1]] - self.yc[y[:, 0]])).sum()))
        return np.array(idx, dtype=np.intp), np.array(rhs, float)


# ===========================================================================
# 3. mediation: NIE(0) under the parallel design
# ===========================================================================

class Mediation(LatentDual):
    """
    Dual of the mediation NIE(0) LP (``Data/Mediate Analysis``).

    Observational cells come in two blocks, matching ``full_primal`` there:

        lam0[d, m, y]  for the natural design   (rows ``d*kM*kY + m*kY + y``)
        lam1[d, j, y]  for the manipulated one  (offset by ``2*kM*kY``)

    A stratum is ``(m0, m1, y)`` with ``y`` indexed by ``(d, j)``, and its row is

        lam0[0,m0,y_{0,m0}] + lam0[1,m1,y_{1,m1}]
          + sum_{d,j} lam1[d,j,y_{d,j}] + lam_norm
          <= s * ( vY[y_{0,m1}] - vY[y_{0,m0}] )

    so ``nnz = 2 + 2*kM``.  Groups: one per ``d`` in the ``lam0`` block (every
    stratum hits one ``(m,y)`` cell there per ``d``) and one per ``(d,j)`` in
    the ``lam1`` block.

    The reduction.  Fix ``(m0, m1)``.  Every outcome slot ``y_{d,j}`` then
    appears in its own ``lam1[d,j,.]`` and, for at most three slots, in a term
    that depends on ``(m0,m1)``::

        slot (0,m0):  + lam0[0,m0,.]  + s*vY[.]
        slot (1,m1):  + lam0[1,m1,.]
        slot (0,m1):  - s*vY[.]

    Every other slot contributes ``max_y lam1[d,j,y]``, independent of
    ``(m0,m1)``.  So with ``S = sum_{d,j} max_y lam1[d,j,y]`` precomputed, each
    of the ``kM^2`` pairs costs ``O(kY)``: total ``O(kM^2 kY)`` against
    ``kM^2 kY^(2kM)`` strata.  When ``m0 == m1`` the first and third
    adjustments land on the same slot and cancel to ``lam0[0,m0,.]``; the code
    accumulates them additively so that case needs no special handling.
    """

    def __init__(self, p0, p1, vY, sign=+1, eps=0.0):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if p0.shape != p1.shape or p0.ndim != 3 or p0.shape[0] != 2:
            raise ValueError(f"p0, p1 must both be (2, kM, kY); got {p0.shape}")
        _, kM, kY = p0.shape
        self.kM, self.kY = kM, kY
        self.sign, self.eps = int(sign), float(eps)
        self.p0, self.p1 = p0, p1
        self.vY = np.asarray(vY, float)
        self.blk = 2 * kM * kY                       # size of the lam0 block
        self.n_obs = 2 * self.blk
        self.b = np.concatenate([p0.reshape(-1), p1.reshape(-1), [1.0]])
        self.groups = (
            [np.arange(d * kM * kY, (d + 1) * kM * kY) for d in (0, 1)] +
            [self.blk + np.arange((d * kM + j) * kY, (d * kM + j + 1) * kY)
             for d in (0, 1) for j in range(kM)])
        self.span = float(self.vY.max() - self.vY.min())
        self._finalise()

    # -- index helpers ----------------------------------------------------
    def i0(self, d, m, y):
        return d * self.kM * self.kY + m * self.kY + y

    def i1(self, d, j, y):
        return self.blk + (d * self.kM + j) * self.kY + y

    def n_reduced_rows(self):
        return self.kM * self.kM

    # -- the reduction ----------------------------------------------------
    def separate(self, x):
        kM, kY, s = self.kM, self.kY, self.sign
        lam0 = x[:self.blk].reshape(2, kM, kY)
        lam1 = x[self.blk:self.n_obs].reshape(2, kM, kY)

        base_arg = lam1.reshape(-1, kY).argmax(axis=1)            # (2kM,)
        base_max = lam1.reshape(-1, kY).max(axis=1)
        S = float(base_max.sum())

        h = np.empty((kM, kM))
        idx = np.empty((kM, kM, 2 + 2 * kM), dtype=np.intp)
        rhs = np.empty((kM, kM))
        slot = lambda d, j: d * kM + j                            # noqa: E731

        for m0 in range(kM):
            for m1 in range(kM):
                # additive adjustments per affected slot
                adj = {}
                adj.setdefault(slot(0, m0), np.zeros(kY))
                adj[slot(0, m0)] += lam0[0, m0] + s * self.vY
                adj.setdefault(slot(1, m1), np.zeros(kY))
                adj[slot(1, m1)] += lam0[1, m1]
                adj.setdefault(slot(0, m1), np.zeros(kY))
                adj[slot(0, m1)] -= s * self.vY

                total = S
                ystar = base_arg.copy()
                for sl, a in adj.items():
                    w = lam1.reshape(-1, kY)[sl] + a
                    j = int(w.argmax())
                    ystar[sl] = j
                    total += float(w[j]) - float(base_max[sl])
                h[m0, m1] = total + x[-1]

                # assemble the attaining stratum as a dual row
                yg = ystar.reshape(2, kM)
                row = [self.i0(0, m0, yg[0, m0]), self.i0(1, m1, yg[1, m1])]
                row += [self.i1(d, j, yg[d, j])
                        for d in (0, 1) for j in range(kM)]
                idx[m0, m1] = row
                rhs[m0, m1] = s * (self.vY[yg[0, m1]] - self.vY[yg[0, m0]])

        return (float(h.max()), idx.reshape(-1, 2 + 2 * kM), rhs.reshape(-1))

    def seed_rows(self):
        """Constant outcome responses ``y_{d,j} = v`` at every ``(m0,m1)``."""
        kM, kY, s = self.kM, self.kY, self.sign
        idx, rhs = [], []
        for v in range(kY):
            for m0 in range(kM):
                for m1 in range(kM):
                    row = [self.i0(0, m0, v), self.i0(1, m1, v)]
                    row += [self.i1(d, j, v)
                            for d in (0, 1) for j in range(kM)]
                    idx.append(row)
                    rhs.append(0.0)          # vY[v] - vY[v] = 0
        return np.array(idx, dtype=np.intp), np.array(rhs, float)

    # -- z3 ---------------------------------------------------------------
    def z3_config_keys(self):
        """One configuration per ``(m0, m1)``; z3 maximises over ``y``."""
        return [(m0, m1) for m0 in range(self.kM) for m1 in range(self.kM)]

    def z3_pin(self, z3, key):
        m0, m1 = key
        return [self._z3m[0] == m0, self._z3m[1] == m1]

    def z3_model(self, z3, opt, lam_arr, cost_0):
        kM, kY, blk = self.kM, self.kY, self.blk
        m0 = z3.Int("m0")
        m1 = z3.Int("m1")
        self._z3m = (m0, m1)
        opt.add(m0 >= 0, m0 < kM, m1 >= 0, m1 < kM)
        # y is a single uninterpreted function over the flat slot d*kM + j
        yf = z3.Function("yf", z3.IntSort(), z3.IntSort())
        for slot in range(2 * kM):
            opt.add(yf(slot) >= 0, yf(slot) < kY)
        # the slots reached through a symbolic m need their own bounds; z3 links
        # them to the concrete ones by congruence once m is fixed
        opt.add(yf(m0) >= 0, yf(m0) < kY, yf(m1) >= 0, yf(m1) < kY,
                yf(kM + m1) >= 0, yf(kM + m1) < kY)
        vy = z3.Array("vy", z3.IntSort(), z3.RealSort())
        cont = z3.K(z3.IntSort(), z3.RealVal(0))
        for i, v in enumerate(self.vY):
            cont = z3.Store(cont, i, z3.RealVal(repr(float(v))))
        opt.add(vy == cont)

        sgn = z3.RealVal(repr(float(self.sign)))
        # rhs = s * ( vY[y_{0,m1}] - vY[y_{0,m0}] ); slot (0, m) is just m
        cost = cost_0 - sgn * (z3.Select(vy, yf(m1)) - z3.Select(vy, yf(m0)))
        # lam0 block: one cell per d
        cost = cost + z3.Select(lam_arr, m0 * kY + yf(m0))
        cost = cost + z3.Select(lam_arr, kM * kY + m1 * kY + yf(kM + m1))
        # lam1 block: one cell per (d, j)
        for d in (0, 1):
            for j in range(kM):
                slot = d * kM + j
                cost = cost + z3.Select(lam_arr,
                                        blk + slot * kY + yf(slot))

        def read(model, key):
            a0, a1 = key
            ys = np.array([model.eval(yf(slot), model_completion=True).as_long()
                           for slot in range(2 * kM)], dtype=np.intp)
            yg = ys.reshape(2, kM)
            idx = [self.i0(0, a0, yg[0, a0]), self.i0(1, a1, yg[1, a1])]
            idx += [self.i1(d, j, yg[d, j])
                    for d in (0, 1) for j in range(kM)]
            rhs = self.sign * (self.vY[yg[0, a1]] - self.vY[yg[0, a0]])
            return np.array(idx, dtype=np.intp), float(rhs)

        return cost, read

    def reduced_lp(self):
        """
        The slot maxima of ``separate`` as six epigraph families of ``kM``::

            S1[j] >= lam1[1,j,y]                          slot (1,j), plain
            S0[j] >= lam1[0,j,y]                          slot (0,j), plain
            U1[j] >= lam1[1,j,y] + lam0[1,j,y]            slot (1,m1),  j = m1
            U0[j] >= lam1[0,j,y] + lam0[0,j,y]            slot (0,m0),  m0 = m1
            Pp[j] >= lam1[0,j,y] + lam0[0,j,y] + s*vY[y]  slot (0,m0),  m0 != m1
            Nn[j] >= lam1[0,j,y] - s*vY[y]                slot (0,m1),  m0 != m1

        plus one row per ``(m0, m1)`` summing the family each slot takes and
        ``lam_norm``, ``<= 0``.  ``6 kM kY + kM^2`` rows over ``n + 6 kM``.
        """
        kM, kY, s = self.kM, self.kY, self.sign
        lp = LiftedLP(self.n)
        S1, S0, U1, U0, Pp, Nn = (lp.aux(kM) for _ in range(6))
        for j in range(kM):
            for y in range(kY):
                a0, a1 = self.i1(0, j, y), self.i1(1, j, y)
                b0, b1 = self.i0(0, j, y), self.i0(1, j, y)
                lp.epigraph(S1[j], [a1])
                lp.epigraph(S0[j], [a0])
                lp.epigraph(U1[j], [a1, b1])
                lp.epigraph(U0[j], [a0, b0])
                lp.epigraph(Pp[j], [a0, b0], s * self.vY[y])
                lp.epigraph(Nn[j], [a0], -s * self.vY[y])
        for m0 in range(kM):
            for m1 in range(kM):
                cols = [U1[m1]] + [S1[j] for j in range(kM) if j != m1]
                if m0 == m1:
                    cols += [U0[m0]] + [S0[j] for j in range(kM) if j != m0]
                else:
                    cols += [Pp[m0], Nn[m1]]
                    cols += [S0[j] for j in range(kM) if j not in (m0, m1)]
                lp.leq(cols + [self.n - 1], 0.0)
        return lp

    def all_rows(self):
        kM, kY, s = self.kM, self.kY, self.sign
        idx, rhs = [], []
        for m0 in range(kM):
            for m1 in range(kM):
                for y_vec in itertools.product(range(kY), repeat=2 * kM):
                    yg = np.array(y_vec, dtype=np.intp).reshape(2, kM)
                    row = [self.i0(0, m0, yg[0, m0]), self.i0(1, m1, yg[1, m1])]
                    row += [self.i1(d, j, yg[d, j])
                            for d in (0, 1) for j in range(kM)]
                    idx.append(row)
                    rhs.append(s * (self.vY[yg[0, m1]] - self.vY[yg[0, m0]]))
        return np.array(idx, dtype=np.intp), np.array(rhs, float)