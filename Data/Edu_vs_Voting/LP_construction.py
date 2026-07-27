"""
Exact (Balke--Pearl canonical) LP for the education--voting DAG.

    X ---> D ---> Y
     \\     ^      ^
      \\    |      |
       \\-> |      |
            U -----+          (U unobserved, U -/- X)

Following the generalized principal-stratification recipe, every endogenous
mechanism gets a full response function.  X is a root we condition on, so it
enters as its realized value, not as a response function:

    f^u_D : {0,...,kx-1} -> {0,1}
            f^u_D(.)   = d = (d_0, ..., d_{kx-1})
    f^u_Y : {0,...,kx-1} x {0,1} -> {0,...,ky-1}
            f^u_Y(.,.) = y = (y_{x,d})_{x,d}

The edge X -> Y is what forces f^u_Y to carry the extra argument x, so the
outcome response is a two-index array.  A stratum is the pair (d, y) and U is
realized as the joint over these, with pmf q.  The number of strata is

    #strata = |D|^|X| * |Y|^{|D||X|} = 2^kx * ky^{2 kx},

exponential in kx.  This module builds that LP *exactly* -- no collapsing --
so it can be used as the ground truth against which reduced/neural
formulations are checked.

Primal (min/max over q):

    min_q / max_q   c^T q
    s.t.            sum_{(d,y): d_x = d, y_{x,d} = y} q_{d,y} = p(d,y | x)
                    sum_{d,y} q_{d,y} = 1,   q >= 0

with the ATE cost, which depends on y only (never on d):

    c_{(d,y)} = sum_x P(X=x) ( y_{x,1} - y_{x,0} ).

Dual (variables lambda_{x,d,y} and lambda_empty), with exactly |X| + 1
non-zero entries per column of A^T:

    (A^T lambda)_{(d,y)} = sum_x lambda_{x, d_x, y_{x,d_x}} + lambda_empty <= c_y.

Because there is no exclusion restriction here -- X -> Y is present, so nothing
couples one stratum of X to another -- conditioning on X separates the LP.  The
exponential column count is real but redundant: only (2 ky)^kx columns of A are
distinct (`prune_duplicate_columns`), and both sides of the dual constraint are
additive over x, so the ky^(2kx) dual constraints collapse to one
(`dual_violation_reduced`).  The resulting bound is the x-stratified Manski
bound, which telescopes to an x-free expression; the growth in kx shows up in
the size of the program, not in the value it returns.
"""

import numpy as np
from itertools import product
from collections import defaultdict
from scipy import sparse

__all__ = [
    "generate_data_EV",
    "discretize",
    "empirical_distribution_EV",
    "n_strata",
    "latent_types_EV",
    "marginal_and_conditional",
    "bin_centers",
    "objective_vector_EV",
    "build_constraints_EV_exact",
    "prune_duplicate_columns",
    "dual_objective",
    "dual_violation_bruteforce",
    "dual_violation_reduced",
]


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
# Stratum bookkeeping
# ============================================================

def n_strata(kx, ky):
    """|D|^|X| * |Y|^{|D||X|} = 2^kx * ky^(2 kx)  (eq. `fullcount`)."""
    return (2 ** kx) * (ky ** (2 * kx))


def y_slot(x, d):
    """Flat position of y_{x,d} inside the response array y."""
    return 2 * x + d


def latent_types_EV(kx, ky):
    """
    Enumerate the response functions.

        D_types : all d in {0,1}^kx           -- d[x] = d_x
        Y_types : all y in {0,...,ky-1}^(2kx) -- y[2x+d] = y_{x,d}

    The stratum (d, y) at position jd of D_types and jy of Y_types lives at
    column  jd * len(Y_types) + jy  (i.e. d is the outer loop, y the inner),
    matching the (T_types, Y_types) ordering used by the continuous-IV code.
    """
    D_types = list(product([0, 1], repeat=kx))
    Y_types = list(product(range(ky), repeat=2 * kx))
    return D_types, Y_types


# ============================================================
# Observational quantities
# ============================================================

def marginal_and_conditional(P):
    """
    Split the empirical joint P(x,d,y) into

        px[x]        = P(X = x)
        p_cond[x,d,y] = p(d, y | x)      (the LP right-hand side, `p_hat`)

    Rows for empty x-bins are dropped downstream (px = 0 there).
    """
    px = P.sum(axis=(1, 2))
    p_cond = np.zeros_like(P)
    support = px > 0
    p_cond[support] = P[support] / px[support][:, None, None]
    return px, p_cond


def bin_centers(y_bins):
    return (y_bins[:-1] + y_bins[1:]) / 2


# ============================================================
# Objective:  c_(d,y) = sum_x P(x) ( yc[y_{x,1}] - yc[y_{x,0}] )
# ============================================================

def objective_vector_EV(px, y_centers, kx, ky):
    """
    Build c over all 2^kx * ky^(2kx) strata without materializing the
    stratum list.  c is constant in d, so it is computed once over the
    ky^(2kx) outcome responses and tiled 2^kx times.

    The per-x contribution  px[x] * (yc[y_{x,1}] - yc[y_{x,0}])  is additive,
    so we accumulate it slot by slot with broadcasting.
    """
    n_y = ky ** (2 * kx)
    c_y = np.zeros(n_y)

    for x in range(kx):
        for d in (0, 1):
            slot = y_slot(x, d)
            # digit of the base-ky number at position `slot`
            # (slot 0 is the most significant, matching itertools.product)
            block = ky ** (2 * kx - slot - 1)
            digits = (np.arange(n_y) // block) % ky
            sign = 1.0 if d == 1 else -1.0
            c_y += sign * px[x] * y_centers[digits]

    return np.tile(c_y, 2 ** kx)


# ============================================================
# Constraint matrix
# ============================================================

def build_constraints_EV_exact(P, kx, ky, y_bins, sparse_A=True,
                               return_labels=True):
    """
    Exact LP:  A q = b,  q >= 0,  objective c^T q.

    Rows are indexed by (x, d, y) -- one per observable cell with P(X=x) > 0 --
    plus a final normalization row labelled (-1, -1, -1).
    Columns are indexed by the strata (d, y) in the order of `latent_types_EV`.

    Every column has exactly (#supported x) + 1 non-zeros: for each x the
    single cell (x, d_x, y_{x,d_x}) it is compatible with, plus the
    normalization row.  A is therefore built directly in COO form instead of
    scanning rows x columns.

    Returns
    -------
    A : (n_rows, n_cols) scipy.sparse.csr_matrix or np.ndarray
    b : (n_rows,) np.ndarray
    c : (n_cols,) np.ndarray
    labels : list of row labels (x, d, y), last one (-1, -1, -1)
    """
    px, p_cond = marginal_and_conditional(P)
    y_centers = bin_centers(y_bins)

    x_support = [x for x in range(kx) if px[x] > 0]

    # ---- rows -------------------------------------------------------
    row_of = {}
    labels = []
    for x in x_support:
        for d in (0, 1):
            for y in range(ky):
                row_of[(x, d, y)] = len(labels)
                labels.append((x, d, y))

    norm_row = len(labels)
    labels.append((-1, -1, -1))
    n_rows = norm_row + 1

    b = np.array([p_cond[x, d, y] for (x, d, y) in labels[:-1]] + [1.0])

    # ---- columns ----------------------------------------------------
    n_y = ky ** (2 * kx)
    n_d = 2 ** kx
    n_cols = n_d * n_y

    # digits[slot] : value of y at that slot for every outcome response
    idx_y = np.arange(n_y)
    digits = np.empty((2 * kx, n_y), dtype=np.int64)
    for slot in range(2 * kx):
        block = ky ** (2 * kx - slot - 1)
        digits[slot] = (idx_y // block) % ky

    rows = []
    cols = []
    D_types = list(product([0, 1], repeat=kx))

    for jd, d_vec in enumerate(D_types):
        offset = jd * n_y
        col_block = offset + idx_y
        for x in x_support:
            d = d_vec[x]
            # row index (x, d, y_{x,d}) for every outcome response at once
            base = row_of[(x, d, 0)]
            rows.append(base + digits[y_slot(x, d)])
            cols.append(col_block)

    # normalization row
    rows.append(np.full(n_cols, norm_row))
    cols.append(np.arange(n_cols))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.size)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    if not sparse_A:
        A = A.toarray()

    c = objective_vector_EV(px, y_centers, kx, ky)

    if return_labels:
        return A, b, c, labels
    return A, b, c


# ============================================================
# Column pruning
# ============================================================

def prune_duplicate_columns(A, c, sense="min", verbose=False):
    """
    Strata that agree on every observable cell give identical columns of A and
    differ only through the counterfactual arms y_{x, 1-d_x}, which enter c but
    no constraint.  Keeping one representative per distinct column is exact,
    provided the representative carries the extreme cost for the sense being
    solved: the cheapest column for a minimization, the dearest for a
    maximization.

    Distinct columns are the maps x -> (d_x, y_{x,d_x}), so this collapses
    2^kx * ky^(2kx) strata down to (2 ky)^kx.
    """
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")

    if sparse.issparse(A):
        Acsc = A.tocsc()
        patterns = (
            tuple(np.sort(Acsc.indices[Acsc.indptr[j]:Acsc.indptr[j + 1]]))
            for j in range(Acsc.shape[1])
        )
    else:
        patterns = (tuple(A[:, j]) for j in range(A.shape[1]))

    groups = defaultdict(list)
    for j, pattern in enumerate(patterns):
        groups[pattern].append(j)

    pick = np.argmin if sense == "min" else np.argmax

    keep = []
    for pattern, idxs in groups.items():
        best = idxs[int(pick([c[i] for i in idxs]))]
        keep.append(best)
        if verbose:
            print(f"group={idxs}, costs={[c[i] for i in idxs]}, keep={best}")

    keep = sorted(keep)
    A_new = A[:, keep]
    c_new = c[keep]
    return A_new, c_new, keep


# ============================================================
# Dual
# ============================================================

def dual_objective(lam, lam_empty, b):
    """b^T lambda, with b's normalization entry paired with lambda_empty."""
    return float(np.dot(lam, b[:-1]) + lam_empty * b[-1])


def _lam_grid(lam, labels, kx, ky):
    """Reshape the dual vector into grid[x, d, y] (0 for unsupported x)."""
    grid = np.zeros((kx, 2, ky))
    for val, (x, d, y) in zip(lam, labels[:len(lam)]):
        if x >= 0:
            grid[x, d, y] = val
    return grid


def dual_violation_bruteforce(lam, lam_empty, labels, kx, ky, px, y_centers,
                              sense="min"):
    """
    max over all 2^kx * ky^(2kx) strata of

        sum_x lambda_{x, d_x, y_{x,d_x}} + lambda_empty - c_y      (sense='min')

    (negated for sense='max', where the dual constraints point the other way).
    Feasible iff the returned value is <= 0.  Exponential -- reference only.
    """
    grid = _lam_grid(lam, labels, kx, ky)
    D_types, Y_types = latent_types_EV(kx, ky)

    worst = -np.inf
    for d_vec in D_types:
        for y_vec in Y_types:
            lhs = sum(grid[x, d_vec[x], y_vec[y_slot(x, d_vec[x])]]
                      for x in range(kx)) + lam_empty
            c_y = sum(px[x] * (y_centers[y_vec[y_slot(x, 1)]]
                               - y_centers[y_vec[y_slot(x, 0)]])
                      for x in range(kx))
            slack = lhs - c_y if sense == "min" else c_y - lhs
            worst = max(worst, slack)
    return worst


def dual_violation_reduced(lam, lam_empty, labels, kx, ky, px, y_centers,
                           sense="min"):
    """
    The same quantity in O(kx * ky^2).

    Both sides of the dual constraint are additive over x, and d_x may be
    chosen independently for each x, so

        max_d sum_x lambda_{x,d_x,y_{x,d_x}} = sum_x max_d lambda_{x,d,y_{x,d}},

    and the outer max over y then separates as well:

        max_{d,y} [ (A^T lambda)_(d,y) - c_y ]
            = lambda_empty
              + sum_x max_{y0,y1} [ max_d lambda_{x,d,y_d}
                                    - px[x] (yc[y1] - yc[y0]) ].

    So the ky^(2kx) dual constraints collapse to a single one -- this is the
    collapse that conditioning on X buys, and the reason the whole exponential
    family never has to be written down.
    """
    grid = _lam_grid(lam, labels, kx, ky)

    total = lam_empty if sense == "min" else -lam_empty
    for x in range(kx):
        # rows: y0 = y_{x,0}, cols: y1 = y_{x,1}
        c_x = px[x] * (y_centers[None, :] - y_centers[:, None])
        if sense == "min":
            # constraints A^T lambda <= c: the worst d maximizes the lhs
            lhs = np.maximum(grid[x, 0, :][:, None], grid[x, 1, :][None, :])
            slack = lhs - c_x
        else:
            # constraints A^T lambda >= c: the worst d minimizes the lhs
            lhs = np.minimum(grid[x, 0, :][:, None], grid[x, 1, :][None, :])
            slack = c_x - lhs
        total += slack.max()
    return total