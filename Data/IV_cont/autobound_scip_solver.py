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

# ============================================================
# 8. Run Experiment (Matches Table 3)
# ============================================================
n = 10000
lam = 0.5 # np.random.rand()
k=8
print(f"Discretize into {k} bins")

data = generate_data_IV(n, lam)
# P = empirical_distribution_IV(data, k=k)
P = np.load("./Data/IV_cont/P8.npy")

A, b, c, labels = build_constraints_IV(P, k=k)
# A_pruned, c_pruned, keep_idx = prune_duplicate_columns(A, c)
print(A)
print("Original vars:", A.shape)
# print("Pruned vars:", A_pruned.shape)
# c = ate_vector(k=8)
start = time.time()
lower, upper = solve_lp_scip(c, A, b)
# print('solving pruned problem')
# lower_pruned, upper_pruned = solve_lp_scip(c, A, b)
end = time.time()

start2 = time.time()
lower_implicit, upper_implicit = solve_lp_scip_implicit(P, k)
print('solving implicit problem')
end2 = time.time()

print("\n==============================")
print("ATE LOWER:", lower, lower_pruned, lower_implicit)
print("ATE UPPER:", upper, upper_pruned, upper_implicit)
print("TRUE ATE = 3")
print("==============================")

print("Time taken: ", end-start)
print("Time taken (implicit): ", end2-start2)