import numpy as np
from itertools import product

# ============================================================
# Continuous Data Generator (X, Y continuous; D binary)
# ============================================================

def generate_data_EV(n, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)

    # Continuous covariate
    X = rng.uniform(0, 1, size=(n, 1))
    U = rng.normal(0, 0.5, size=(n, 1))

    # Structural function
    alpha = 2 * (X - 0.5)

    # Continuous potential outcomes
    Y0 = alpha + U # rng.normal(0, 0.2, size=(n, 1))
    Y1 = alpha + tau + U # rng.normal(0, 0.2, size=(n, 1))

    # Confounded treatment assignment
    logits_T = 2 * X.squeeze() + 1.5 * Y0.squeeze()
    p_T = 1 / (1 + np.exp(-logits_T))

    D = rng.binomial(1, p_T).reshape(-1, 1)

    # Observed outcome
    Y = np.where(D == 1, Y1, Y0)

    return np.hstack([X, D, Y]), Y0, Y1


# ============================================================
# Discretization
# ============================================================

def discretize(x, k):
    bins = np.linspace(x.min(), x.max(), k + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, k - 1)
    return idx, bins


# ============================================================
# Empirical Distribution P(X,D,Y)
# ============================================================

def empirical_distribution_EV(data, kx, ky):
    X, x_bins = discretize(data[:, 0], kx)
    D = data[:, 1].astype(int)
    Y, y_bins = discretize(data[:, 2], ky)

    P = np.zeros((kx, 2, ky))
    n = len(data)

    for i in range(n):
        P[X[i], D[i], Y[i]] += 1

    P /= n

    return P, x_bins, y_bins


# ============================================================
# Latent Types
# ============================================================

def latent_types(kx, ky):
    Y_types = list(product(range(ky), repeat=2))  # (y0, y1)
    X_types = list(range(kx))
    D_types = [0, 1]

    T_types = list(product(X_types, D_types, Y_types))
    return T_types


# ============================================================
# Build LP constraints
# ============================================================

def build_constraints_EV(P, kx, ky, y_bins):
    """
    Variables:
        p[x,d,y0,y1]

    Returns:
        A, b, c, labels
    """

    T_types = latent_types(kx, ky)
    index = {t: i for i, t in enumerate(T_types)}
    n_vars = len(T_types)

    A = []
    b = []

    # --------------------------------------------------------
    # 1. Normalization
    # --------------------------------------------------------
    A.append(np.ones(n_vars))
    b.append(1.0)

    # --------------------------------------------------------
    # 2. Observational constraints
    # sum_{y0,y1: y = y_d} p(x,d,y0,y1) = P(x,d,y)
    # --------------------------------------------------------
    for x in range(kx):
        for d in [0, 1]:
            for y in range(ky):

                row = np.zeros(n_vars)

                for (xx, dd, (y0, y1)) in T_types:
                    if xx == x and dd == d:
                        y_obs = y1 if d == 1 else y0
                        if y_obs == y:
                            row[index[(xx, dd, (y0, y1))]] = 1

                A.append(row)
                b.append(P[x, d, y])

    # --------------------------------------------------------
    # 3. Objective: ATE using bin centers
    # --------------------------------------------------------
    c = np.zeros(n_vars)

    # compute bin centers for Y
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    for (x, d, (y0, y1)) in T_types:
        c[index[(x, d, (y0, y1))]] = y_centers[y1] - y_centers[y0]

    labels = T_types

    return np.array(A), np.array(b), np.array(c), labels