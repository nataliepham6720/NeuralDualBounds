import numpy as np
from itertools import product

# ============================================================
# Continuous Data Generator (controlled ATE)
# ============================================================

def generate_data(n, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)

    # Continuous covariate X
    X = rng.uniform(0, 1, size=(n, 1))

    # Baseline + treatment effect
    alpha = 2 * (X - 0.5)           # heterogeneity
    tau = tau                        # controls ATE

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Potential outcomes
    p_y0 = sigmoid(alpha)
    p_y1 = sigmoid(alpha + tau)

    Y0 = rng.binomial(1, p_y0)
    Y1 = rng.binomial(1, p_y1)

    # Treatment assignment (confounded via Y0)
    logits_T = 2 * X.squeeze() + 1.5 * Y0.squeeze()
    p_T = sigmoid(logits_T)

    T = rng.binomial(1, p_T).reshape(-1, 1)

    # Observed outcome
    Y = np.where(T == 1, Y1, Y0).reshape(-1, 1)

    return np.hstack([X, T, Y]), Y0, Y1


# ============================================================
# Discretization
# ============================================================

def discretize(x, k):
    bins = np.linspace(x.min(), x.max(), k+1)
    return np.clip(np.digitize(x, bins) - 1, 0, k-1)


# ============================================================
# Empirical Distribution P(X,D,Y)
# ============================================================

def empirical_distribution(data, k):
    X = discretize(data[:, 0], k)
    D = discretize(data[:, 1], k)   # D already binary but general
    Y = data[:, 2].astype(int)

    P = np.zeros((k, k, 2))

    n = len(data)
    for i in range(n):
        P[X[i], D[i], Y[i]] += 1

    P /= n
    return P


# ============================================================
# Latent Types
# ============================================================

def latent_types(k):
    # Potential outcomes (y0, y1)
    Y_types = list(product([0,1], repeat=2))   # 4 types

    # X bins and D bins
    X_types = list(range(k))
    D_types = list(range(k))

    # Full latent types: (x, d, y0, y1)
    T_types = list(product(X_types, D_types, Y_types))

    return T_types, Y_types


# ============================================================
# Build LP constraints
# ============================================================

def build_constraints(P, k):
    """
    Variables:
        p[x,d,y0,y1]
    """

    T_types, Y_types = latent_types(k)

    index = {t: i for i, t in enumerate(T_types)}
    n_vars = len(T_types)

    A = []
    b = []

    # --------------------------------------------------------
    # 1. Normalization
    # --------------------------------------------------------
    row = np.ones(n_vars)
    A.append(row)
    b.append(1.0)

    # --------------------------------------------------------
    # 2. Match observed distribution
    # P(X=x, D=d, Y=y)
    # --------------------------------------------------------
    for x in range(k):
        for d in range(k):
            for y in [0,1]:

                row = np.zeros(n_vars)

                for (xx, dd, (y0, y1)) in T_types:
                    if xx == x and dd == d:
                        y_obs = y1 if d > k//2 else y0
                        if y_obs == y:
                            row[index[(xx, dd, (y0, y1))]] = 1

                A.append(row)
                b.append(P[x,d,y])

    # --------------------------------------------------------
    # 3. Objective: ATE
    # --------------------------------------------------------
    c = np.zeros(n_vars)

    for (x, d, (y0, y1)) in T_types:
        c[index[(x,d,(y0,y1))]] = (y1 - y0)

    labels = T_types

    return np.array(A), np.array(b), np.array(c), labels