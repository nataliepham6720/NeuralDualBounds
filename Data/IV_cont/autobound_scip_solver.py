import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import truncnorm
from pyscipopt import Model
import time

from LP_construction import *

SEED = 2020

np.random.seed(SEED)
# torch.manual_seed(SEED)

# def solve_lp_scip(c, A, b, eps=1e-6):

#     n = len(c)

#     def solve_sense(sense):
#         m = Model()
#         m.hideOutput()

#         p = [m.addVar(lb=0, ub=1) for _ in range(n)]

#         # Sum to 1
#         m.addCons(sum(p) == 1)

#         # Conditional constraints
#         for i in range(len(b)):
#             expr = sum(A[i,j] * p[j] for j in range(n))
#             m.addCons(expr >= b[i] - eps)
#             m.addCons(expr <= b[i] + eps)

#         obj = sum(c[j] * p[j] for j in range(n))
#         m.setObjective(obj, sense)

#         m.optimize()

#         if m.getStatus() != "optimal":
#             raise RuntimeError("SCIP infeasible — discretization too coarse")

#         return m.getObjVal()
#     print("Solve lower bound")
#     lower = solve_sense("minimize")
#     print("Solve upper bound")
#     upper = solve_sense("maximize")

#     return lower, upper

def solve_lp_scip(c, A, b, eps=1e-6):

    n = len(c)

    def solve_sense(sense):
        m = Model()
        m.hideOutput()

        p = [m.addVar(lb=0, ub=1) for _ in range(n)]

        # normalization
        m.addCons(sum(p) == 1)

        # constraints
        for i in range(len(b)):
            expr = sum(A[i,j]*p[j] for j in range(n))
            m.addCons(expr >= b[i]-eps)
            m.addCons(expr <= b[i]+eps)

        obj = sum(c[j]*p[j] for j in range(n))
        m.setObjective(obj, sense)

        m.optimize()

        status = m.getStatus()
        if status != "optimal":
            raise RuntimeError(f"SCIP status: {status}")
        print("obj", m.getObjVal(),         # optimal objective
            "primal", m.getPrimalbound(), # primal bound
            "dual", m.getDualbound(),     # dual bound
            "gap",m.getGap())
        return {
            "obj": m.getObjVal(),         # optimal objective
            "primal": m.getPrimalbound(), # primal bound
            "dual": m.getDualbound(),     # dual bound
            "gap": m.getGap()
        }

    print("Solve lower bound")
    lower = solve_sense("minimize")

    print("Solve upper bound")
    upper = solve_sense("maximize")

    return lower, upper


def solve_lp_scip_implicit(P, k, eps=1e-6):
    """
    Solve the primal LP directly from P and k without materialising A.

    Constraint structure (from ImplicitLP_IV):
        sum_{Tt[z]==t, Yt[t]==y} p[Tt, Yt]  ==  P[z, t, y]   for all z, t, y
        sum_{Tt, Yt}             p[Tt, Yt]  ==  1              (normalization)

    Precomputing index lookup tables reduces constraint-building cost from
    O(m * n) = O(k^2 * 2^k * k^2) down to O(k^3 * 2^k).
    """
    T_types = list(product([0, 1], repeat=k))   # 2^k
    Y_types = list(product(range(k), repeat=2)) # k^2
    n_Tt = len(T_types)
    n_Yt = len(Y_types)
    n    = n_Tt * n_Yt

    # Precompute: Tt indices where T_types[i][z] == t
    tt_for = {
        (z, t): [i for i, Tt in enumerate(T_types) if Tt[z] == t]
        for z in range(k) for t in [0, 1]
    }
    # Precompute: Yt indices where Y_types[j][t] == y
    yt_for = {
        (t, y): [j for j, Yt in enumerate(Y_types) if Yt[t] == y]
        for t in [0, 1] for y in range(k)
    }

    def solve_sense(sense):
        model = Model()
        model.hideOutput()

        p = [model.addVar(lb=0, ub=1) for _ in range(n)]

        # Normalization
        model.addCons(sum(p) == 1)

        # Observational constraints built from implicit structure
        for z in range(k):
            for t in [0, 1]:
                for y in range(k):
                    expr = sum(
                        p[i * n_Yt + j]
                        for i in tt_for[z, t]
                        for j in yt_for[t, y]
                    )
                    model.addCons(expr >= P[z, t, y] - eps)
                    model.addCons(expr <= P[z, t, y] + eps)

        # Objective: E[Y1 - Y0]
        obj = sum(
            (Y_types[j][1] - Y_types[j][0]) * p[i * n_Yt + j]
            for i in range(n_Tt)
            for j in range(n_Yt)
        )
        model.setObjective(obj, sense)
        model.optimize()

        if model.getStatus() != "optimal":
            raise RuntimeError("SCIP infeasible — discretization too coarse")

        return model.getObjVal()

    print("Solving lower bound (implicit A)...")
    lower = solve_sense("minimize")
    print("Solving upper bound (implicit A)...")
    upper = solve_sense("maximize")
    return lower, upper

def solve_dual_lp_scip_poly(P, k, eps=1e-6):
    """
    Solve the dual LP with a polynomial number of constraints.

    The dual of the primal LP has one constraint per latent type
    j = (f_T, y0, y1), i.e. 2^k * k^2 constraints:

        sum_z lam[z, t_z, y_{t_z}] + lam_norm <= y1 - y0.

    Since the bound must hold for every treatment map f_T in {0,1}^k, the
    binding constraint for a fixed (y0, y1) is obtained by maximizing over
    t_z independently for each z:

        sum_z max(lam[z,0,y0], lam[z,1,y1]) + lam_norm <= y1 - y0.

    Each max is linearised with an auxiliary variable aux[z,y0,y1] >= both
    arguments, so the 2^k * k^2 type constraints collapse to k^2 grouped
    constraints plus 2k^3 linearisation constraints: O(k^3) instead of
    exponential in k.

    Lower ATE bound:  max  p_obs^T delta  s.t. A^T delta <= c
    Upper ATE bound:  min  p_obs^T delta  s.t. A^T delta >= c
    (for the upper bound the grouping uses min instead of max, linearised
    with aux <= both arguments).

    The primal slack (b - eps <= Aq <= b + eps) shows up in the dual as an
    L1 penalty eps * ||lam_obs||_1 on the objective, which also keeps the
    dual bounded when the empirical P is slightly infeasible.
    """

    def solve_sense(sense):
        m = Model()
        m.hideOutput()

        # dual variables for the 2k^2 observational constraints (free)
        lam = {}
        lam_abs = {}
        for z in range(k):
            for t in [0, 1]:
                for y in range(k):
                    lam[z, t, y] = m.addVar(lb=None, ub=None)
                    # |lam| for the eps-slack L1 penalty
                    lam_abs[z, t, y] = m.addVar(lb=0, ub=None)
                    m.addCons(lam_abs[z, t, y] >= lam[z, t, y])
                    m.addCons(lam_abs[z, t, y] >= -lam[z, t, y])
        # dual variable for the normalization constraint (free)
        lam_norm = m.addVar(lb=None, ub=None)

        # aux[z,y0,y1] linearises max/min(lam[z,0,y0], lam[z,1,y1])
        aux = {}
        for z in range(k):
            for y0 in range(k):
                for y1 in range(k):
                    aux[z, y0, y1] = m.addVar(lb=None, ub=None)
                    if sense == "maximize":
                        m.addCons(aux[z, y0, y1] >= lam[z, 0, y0])
                        m.addCons(aux[z, y0, y1] >= lam[z, 1, y1])
                    else:
                        m.addCons(aux[z, y0, y1] <= lam[z, 0, y0])
                        m.addCons(aux[z, y0, y1] <= lam[z, 1, y1])

        # one grouped constraint per (y0, y1) pair
        for y0 in range(k):
            for y1 in range(k):
                expr = sum(aux[z, y0, y1] for z in range(k)) + lam_norm
                if sense == "maximize":
                    m.addCons(expr <= y1 - y0)
                else:
                    m.addCons(expr >= y1 - y0)

        obj = sum(
            P[z, t, y] * lam[z, t, y]
            for z in range(k) for t in [0, 1] for y in range(k)
        ) + lam_norm
        penalty = eps * sum(lam_abs.values())

        if sense == "maximize":
            m.setObjective(obj - penalty, "maximize")
        else:
            m.setObjective(obj + penalty, "minimize")

        m.optimize()

        status = m.getStatus()
        if status != "optimal":
            raise RuntimeError(f"SCIP status: {status}")
        return m.getObjVal()

    print("Solving lower bound (poly dual)...")
    lower = solve_sense("maximize")
    print("Solving upper bound (poly dual)...")
    upper = solve_sense("minimize")
    return lower, upper


# ============================================================
# 8. Run Experiment (Matches Table 3)
# ============================================================
def main():
    import argparse

# Dual LP with O(k^3) constraints instead of 2^k * k^2
start = time.time()
lower, upper = solve_dual_lp_scip_poly(P, k)
end = time.time()
    parser = argparse.ArgumentParser(
        description="ATE bounds for the continuous IV setting via SCIP."
    )
    parser.add_argument("--k", type=int, default=8,
                        help="number of discretization bins for Z and Y")
    parser.add_argument("--n", type=int, default=10000,
                        help="sample size when generating data")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="mixture weight lambda of the noise distribution")
    parser.add_argument("--data", choices=["preload", "generate"],
                        default="preload",
                        help="preload P from .npy or generate data and "
                             "compute the empirical distribution")
    parser.add_argument("--P-path", default=None,
                        help="path to the preloaded P (.npy); defaults to "
                             "./Data/IV_cont/P{k}.npy")
    parser.add_argument("--solver", choices=["primal", "implicit", "dual"],
                        default="dual",
                        help="primal: explicit A (2^k k^2 variables); "
                             "implicit: primal without materialising A; "
                             "dual: polynomial O(k^3) dual LP")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="slack on the observational constraints")
    args = parser.parse_args()

print("\n==============================")
print("ATE LOWER:", lower)
print("ATE UPPER:", upper)
print("TRUE ATE = 3")
print("==============================")
    k = args.k
    print(f"Discretize into {k} bins")

print("Time taken (poly dual): ", end-start)
    if args.data == "preload":
        P_path = args.P_path or f"./Data/IV_cont/P{k}.npy"
        print(f"Loading P from {P_path}")
        P = np.load(P_path)
        if P.shape != (k, 2, k):
            raise ValueError(f"P has shape {P.shape}, expected {(k, 2, k)}")
    else:
        print(f"Generating data: n={args.n}, lam={args.lam}")
        data = generate_data_IV(args.n, args.lam)
        P = empirical_distribution_IV(data, k=k)

    start = time.time()
    if args.solver == "primal":
        A, b, c, labels = build_constraints_IV(P, k=k)
        print("Original vars:", A.shape)
        lower, upper = solve_lp_scip(c, A, b, eps=args.eps)
        lower, upper = lower["obj"], upper["obj"]
    elif args.solver == "implicit":
        lower, upper = solve_lp_scip_implicit(P, k, eps=args.eps)
    else:
        # Dual LP with O(k^3) constraints instead of 2^k * k^2
        lower, upper = solve_dual_lp_scip_poly(P, k, eps=args.eps)
    end = time.time()

    print("\n==============================")
    print("ATE LOWER:", lower)
    print("ATE UPPER:", upper)
    print("TRUE ATE = 3")
    print("==============================")

    print(f"Time taken ({args.solver}): ", end - start)


if __name__ == "__main__":
    main()