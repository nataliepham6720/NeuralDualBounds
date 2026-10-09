"""
Edu_vs_Voting ATE bounds through every route, on the outcome scale.

    python -m 'Z3.run_edu&voting' --kx 4 --ky 4
    python -m 'Z3.run_edu&voting' --kx 12 --ky 12 --routes reduced-highs reduced
    python -m 'Z3.run_edu&voting' --kx 3 --ky 3 --routes primal reduced reduced-z3

The module name holds an ``&``, so quote it in the shell.
"""

import argparse
import os

import numpy as np

from Data.Edu_vs_Voting.LP_construction import (empirical_distribution_EV,
                                                generate_data_EV, n_strata)

from . import problems, reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "Edu_vs_Voting")
ROUTES = ("primal", "full-highs", "full-scip", "full-ipm",
          "reduced-highs", "reduced-scip", "reduced", "reduced-z3")
TRUE_ATE = 0.5          # Y1 - Y0 = tau exactly in generate_data_EV
MAX_STRATA = 20000      # about where run_iv drops the 2^k-sized routes (k=8)


def load_ev(kx, ky, n=10000, tau=TRUE_ATE, seed=2020):
    """``(P, y_bins, source)``: P(X, D, Y) on ``kx`` x ``ky`` bins."""
    path = os.path.join(DATA, f"EV_kx{kx}_ky{ky}_n{n}_s{seed}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return z["P"], z["y_bins"], f"preload({os.path.basename(path)})"
    path = os.path.join(DATA, f"P{kx}_EV.npz")
    if kx == ky and os.path.exists(path):
        # n and seed of these draws are not recorded; used as stored
        with np.load(path) as z:
            return z["P"], z["y_bins"], f"preload(P{kx}_EV.npz)"
    data, _, _ = generate_data_EV(n, tau=tau, seed=seed)
    P, _, y_bins = empirical_distribution_EV(data, kx, ky)
    return P, y_bins, f"generate(n={n}, seed={seed})"


def main():
    p = argparse.ArgumentParser(
        description="Edu_vs_Voting ATE bounds, every route side by side.")
    p.add_argument("--kx", type=int, default=4)
    p.add_argument("--ky", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.0,
                   help="slack on the observational constraints")
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=None,
                   help="default: all but reduced-z3 (slow), minus the "
                        f"2^kx ky^(2kx)-sized ones above {MAX_STRATA} strata")
    args = p.parse_args()

    kx, ky = args.kx, args.ky
    P, y_bins, src = load_ev(kx, ky)
    routes = args.routes or tuple(
        r for r in ROUTES if r != "reduced-z3" and
        (n_strata(kx, ky) <= MAX_STRATA or
         not r.startswith(("primal", "full"))))

    def make(sign):
        return problems.EduVsVoting(P, kx, ky, y_bins, sign=sign, eps=args.eps)

    reference.compare(make, routes=routes, baseline=routes[0],
                      label=f"Edu_vs_Voting kx={kx} ky={ky} {src}")
    print(f"\nTRUE ATE = {TRUE_ATE:g}")


if __name__ == "__main__":
    main()