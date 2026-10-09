"""
Mediation NIE(0) bounds through every route, on the outcome scale.

    python -m Z3.run_medanalysis --kM 2 --kY 4
    python -m Z3.run_medanalysis --kM 3 --kY 6 --routes reduced-highs reduced
    python -m Z3.run_medanalysis --kM 2 --kY 3 --routes primal reduced reduced-z3

"""

import argparse
import importlib.util
import os

from . import problems, reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = ("primal", "full-highs", "full-scip", "full-ipm",
          "reduced-highs", "reduced-scip", "reduced", "reduced-z3")
MAX_STRATA = 20000      # about where run_iv drops the 2^k-sized routes (k=8)


def _data_module():
    """``Data/Mediate Analysis/reduced_dual_solver.py`` (the space rules out
    a plain import)."""
    path = os.path.join(ROOT, "Data", "Mediate Analysis",
                        "reduced_dual_solver.py")
    spec = importlib.util.spec_from_file_location("mediate_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mediation(kM, kY, n=200000, seed=0):
    """``(p0, p1, vY, true_nie0, source)`` from the data module's SCM."""
    p0, p1, vY, nie = _data_module().make_data(kM, kY, N=n, seed=seed)
    return p0, p1, vY, nie, f"generate(n={n}, seed={seed})"


def main():
    p = argparse.ArgumentParser(
        description="Mediation NIE(0) bounds, every route side by side.")
    p.add_argument("--kM", type=int, default=2)
    p.add_argument("--kY", type=int, default=4)
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=0.0,
                   help="slack on the observational constraints")
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=None,
                   help="default: all but reduced-z3 (slow), minus the "
                        f"kM^2 kY^(2kM)-sized ones above {MAX_STRATA} strata")
    args = p.parse_args()

    kM, kY = args.kM, args.kY
    p0, p1, vY, nie, src = load_mediation(kM, kY, n=args.n, seed=args.seed)
    n_strata = kM ** 2 * kY ** (2 * kM)
    routes = args.routes or tuple(
        r for r in ROUTES if r != "reduced-z3" and
        (n_strata <= MAX_STRATA or not r.startswith(("primal", "full"))))

    def make(sign):
        return problems.Mediation(p0, p1, vY, sign=sign, eps=args.eps)

    reference.compare(make, routes=routes, baseline=routes[0],
                      label=f"Mediation kM={kM} kY={kY} {src}")
    print(f"\nTRUE NIE(0) = {nie:g}  (this draw)")


if __name__ == "__main__":
    main()