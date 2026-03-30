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

def solve_lp_scip(c, A, b, eps=1e-6):

    n = len(c)

    def solve_sense(sense):
        m = Model()
        m.hideOutput()

        p = [m.addVar(lb=0, ub=1) for _ in range(n)]

        # Sum to 1
        m.addCons(sum(p) == 1)

        # Conditional constraints
        for i in range(len(b)):
            expr = sum(A[i,j] * p[j] for j in range(n))
            m.addCons(expr >= b[i] - eps)
            m.addCons(expr <= b[i] + eps)

        obj = sum(c[j] * p[j] for j in range(n))
        m.setObjective(obj, sense)

        m.optimize()

        if m.getStatus() != "optimal":
            raise RuntimeError("SCIP infeasible — discretization too coarse")

        return m.getObjVal()

    lower = solve_sense("minimize")
    upper = solve_sense("maximize")

    return lower, upper


# ============================================================
# 8. Run Experiment (Matches Table 3)
# ============================================================
n = 10000
lam = 0.5 # np.random.rand()

data = generate_data(n, lam)
P = empirical_distribution(data, k=8)

A, b, c, labels = build_constraints(P, k=8)
# c = ate_vector(k=8)
start = time.time()
lower, upper = solve_lp_scip(c, A, b)
end = time.time()

print("\n==============================")
print("ATE LOWER:", lower)
print("ATE UPPER:", upper)
print("TRUE ATE = 3")
print("==============================")

print("Time taken: ", end-start)