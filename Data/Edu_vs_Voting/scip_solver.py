import numpy as np
from itertools import product
from pyscipopt import Model
import time

from LP_construction import *

SEED = 2020
np.random.seed(SEED)

def solve_lp_scip(c, A, b, eps=1e-6):

    n = len(c)

    def solve_sense(sense):
        m = Model()
        m.hideOutput()

        p = [m.addVar(lb=0, ub=1) for _ in range(n)]

        # sum to 1
        m.addCons(sum(p) == 1)

        for i in range(len(b)):
            expr = sum(A[i,j] * p[j] for j in range(n))
            m.addCons(expr >= b[i] - eps)
            m.addCons(expr <= b[i] + eps)

        obj = sum(c[j] * p[j] for j in range(n))
        m.setObjective(obj, sense)

        m.optimize()

        if m.getStatus() != "optimal":
            raise RuntimeError("SCIP infeasible")

        return m.getObjVal()

    print("Solve lower bound")
    lower = solve_sense("minimize")
    print("Solve upper bound")
    upper = solve_sense("maximize")

    return lower, upper


# ============================================================
# 5. Run experiment
# ============================================================

n_pts = 10000
kx = 12
ky = 12

data, Y0, Y1 = generate_data_EV(n_pts, tau=0.5, seed=SEED)
ATE_true = np.mean(Y1 - Y0)
print("True ATE:", ATE_true)

P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky) 
A, b, c, labels = build_constraints_EV(P, kx, ky, y_bins)

print("A shape:", A.shape) 
print("b shape:", b.shape) 
print("c shape:", c.shape) 
print("Number of variables:", len(c))

start = time.time()
lower, upper = solve_lp_scip(c, A, b)
end = time.time()

print("\n==============================")
print("ATE LOWER:", lower)
print("ATE UPPER:", upper)
print("==============================")

print("Time taken:", end-start)