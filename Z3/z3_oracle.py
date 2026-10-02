import numpy as np


class Z3Oracle:
    """
    Wrap a problem's z3 model as a callable oracle ``(x) -> (h, idx, rhs)``.

    The problem must implement ``z3_model``, ``z3_config_keys`` and ``z3_pin``
    (see the examples in :mod:`latent_dual.problems`).  ``top_up`` asks the
    problem's closed-form ``separate`` for extra rows when z3 returns only one
    -- needed for a reduction that collapses to a single constraint, since one
    row per round starves the outer approximation.
    """

    def __init__(self, prob, top_up=True, timeout_ms=0):
        import z3
        self._z3 = z3
        self.prob = prob
        self.top_up = top_up
        self.n_solves = 0

        self.opt = z3.Optimize()
        if timeout_ms:
            self.opt.set("timeout", int(timeout_ms))
        self.lam_arr = z3.Array("matrix", z3.IntSort(), z3.RealSort())
        self.cost_0 = z3.Real("cost0")
        # the problem declares its choice variables, bounds them, and returns
        # the symbolic objective plus a reader for the attaining stratum
        self.cost, self.read = prob.z3_model(z3, self.opt, self.lam_arr,
                                             self.cost_0)
        self.opt.maximize(self.cost)
        self.keys = prob.z3_config_keys()

    # -- helpers ----------------------------------------------------------
    def _container(self, vals):
        """``vals`` as a z3 constant Array (values passed as text, per the ref)."""
        z3 = self._z3
        arr = z3.K(z3.IntSort(), z3.RealVal(0))
        for i, v in enumerate(np.asarray(vals, float).reshape(-1)):
            arr = z3.Store(arr, i, z3.RealVal(repr(float(v))))
        return arr

    def _value(self, model):
        val = model.eval(self.cost, model_completion=True)
        try:
            return float(val.as_fraction())
        except AttributeError:
            return float(val.as_decimal(30).rstrip("?"))

    # -- the oracle -------------------------------------------------------
    def __call__(self, x):
        z3 = self._z3
        prob = self.prob
        x = np.asarray(x, float)

        self.opt.push()
        self.opt.add(self.lam_arr == self._container(x[:prob.n_obs]),
                     self.cost_0 == z3.RealVal(repr(float(x[-1]))))

        h, rows, rhss = -np.inf, [], []
        for key in self.keys:
            self.opt.push()
            for con in prob.z3_pin(z3, key):
                self.opt.add(con)
            sat = self.opt.check()
            self.n_solves += 1
            if sat != z3.sat:
                self.opt.pop()
                self.opt.pop()
                raise RuntimeError(f"z3 returned {sat} for configuration {key}")
            model = self.opt.model()
            h = max(h, self._value(model))
            idx, rhs = self.read(model, key)
            rows.append(np.asarray(idx, dtype=np.intp))
            rhss.append(float(rhs))
            self.opt.pop()
        self.opt.pop()

        if self.top_up and len(rows) == 1:
            _, extra_idx, extra_rhs = prob.separate(x)
            extra_idx = np.asarray(extra_idx).reshape(-1, prob.nnz)
            rows.extend(list(extra_idx[1:]))
            rhss.extend(list(np.ravel(extra_rhs)[1:]))

        return (float(h), np.array(rows, dtype=np.intp),
                np.array(rhss, dtype=float))


def check_z3_oracle(make_problem, trials=2, scale=1.5, seed=0, tol=1e-8):
    """
    Check the z3 oracle against the explicit stratum enumeration.

    Same contract as :func:`latent_dual.reference.check_oracle`: the value must
    match the maximum over ``all_rows()``, and the row z3 hands back must
    reproduce the value it claims.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for sign in (+1, -1):
        prob = make_problem(sign)
        idx, rw = prob.all_rows()
        orc = Z3Oracle(prob, top_up=False)
        for _ in range(trials):
            x = rng.standard_normal(prob.n) * scale
            ref = float((x[idx].sum(axis=1) + x[-1] - rw).max())
            h, sel, srhs = orc(x)
            got = float((x[sel].sum(axis=1) + x[-1] - np.ravel(srhs)).max())
            worst = max(worst, abs(h - ref), abs(got - ref))
    if worst > tol:
        raise AssertionError(f"z3 oracle disagrees with enumeration by {worst:.2e}")
    return worst
