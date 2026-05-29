"""
implicit_neural_solver.py
=========================
Matrix-free neural dual solver for causal partial-identification bounds.

Mathematical background
-----------------------
Both examples solve the same kind of LP: given an empirical distribution P
over observed variables, bound a causal estimand (ATE) by finding the
extremes of  c^T p  subject to  A p = b, p >= 0.

The DUAL of that LP is:
    max   b^T lam
    s.t.  A^T lam <= c

where lam in R^m is the vector of dual variables (one per observational
constraint), and the constraint  A^T lam <= c  has one row per latent type
(n rows, potentially n >> m).

The bottleneck in the central-path algorithm is evaluating  A^T lam  at
every iteration.  Both examples have DAG structure that lets us replace the
explicit (m x n) matrix multiplication with closed-form operations.


Latent types — mathematical meaning
-------------------------------------

IV_cont: (Tt, Yt)
~~~~~~~~~~~~~~~~~~
DAG:    Z --> T --> Y
              ^     ^
              +--U--+

The instrument Z is exogenous; U is an unmeasured confounder.
Structural equations (after discretising to k bins each):
    T = f_T(Z, U)      Y = f_Y(T, U)

A latent type (Tt, Yt) encodes the full counterfactual schedule of a
unit, i.e. its response to every possible value of (Z, T):

    Tt in {0,1}^k   -- Treatment response function.
                       Tt[z] = T that this unit would choose if Z = z.
                       Encodes how the unit's treatment selection responds
                       to each of the k instrument bins.
                       There are 2^k such functions (all maps {0..k-1}->{0,1}).

    Yt = (y0, y1)   -- Potential outcomes.
                       Yt[t] = Y that this unit would realise under treatment t.
                       There are k^2 such pairs.

Together (Tt, Yt) determines the realised (T, Y) for any value of Z:
    T_obs(z) = Tt[z]
    Y_obs(z) = Yt[Tt[z]]

And the individual causal effect is  Yt[1] - Yt[0].
The LP variable  p[Tt, Yt] >= 0  is the population share of this type.

Constraint matrix A (shape m x n, with m = 2k^2+1, n = 2^k * k^2):
    A[z,t,y][Tt,Yt] = 1{Tt[z]=t} * 1{Yt[t]=y}
    "Does unit-type (Tt,Yt) realise (T=t, Y=y) when Z=z?"

Objective:
    c[Tt,Yt] = Yt[1] - Yt[0]   (individual causal effect)

RHS:
    b[z,t,y] = P_data(T=t, Y=y | Z=z)


Edu_vs_Voting: (x, d, (y0, y1))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DAG:    X --> D --> Y
              ^     ^
              +--U--+   (X also -> Y directly)

No instrument. Latent type includes the covariate x and realised treatment d
because selection into treatment is confounded (d is correlated with U):

    x  in {0,..,kx-1}   -- Observed covariate bin.  Fixed attribute of unit.

    d  in {0,1}          -- Realised treatment for this unit.
                            Including d in the type captures selection:
                            p[x,d,y0,y1] = P(X=x, D=d, Y(0)=y0, Y(1)=y1)
                            allows P(D|X,Y(0),Y(1)) to be non-trivial.

    (y0, y1)             -- Potential outcomes Y(0) and Y(1).

The observed outcome is Y = Yt[d] (the potential outcome for the realised d).

Constraint matrix A (shape m x n, m = kx*2*ky+1, n = kx*2*ky^2):
    A[x',d',y'][x,d,(y0,y1)] = 1{x=x'} * 1{d=d'} * 1{Yt[d]=y'}
    "Does latent type (x,d,y0,y1) realise Y=y' under treatment d'?"

Key observation: each latent type touches EXACTLY ONE observational
constraint (the one with x'=x, d'=d, y'=Yt[d]).  So A^T lam reduces
to a single gather operation.

Objective:
    c[x,d,y0,y1] = y_centers[y1] - y_centers[y0]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import wandb
from itertools import product

from Data.IV_cont.LP_construction import generate_data_IV, empirical_distribution_IV
from Data.IV_cont.utils import plot_dual_heatmap
from Data.Edu_vs_Voting.LP_construction import (
    generate_data_EV, empirical_distribution_EV,
)


# ======================================================================
# Implicit LP operators  (no matrix A ever allocated)
# ======================================================================

class ImplicitLP_IV:
    """
    Matrix-free LP operators for IV_cont.

    Decomposition of  A^T lam:
        (A^T lam)[Tt, y0, y1]
            = sum_z  lam_obs[z, Tt[z], Yt[Tt[z]]]  +  lam_norm
            = sum_{z: Tt[z]=0} lam_obs[z, 0, y0]
            + sum_{z: Tt[z]=1} lam_obs[z, 1, y1]
            + lam_norm
            = S0[Tt, y0]  +  S1[Tt, y1]  +  lam_norm

    where
        S0 = mask0 @ lam_obs[:, 0, :]     (2^k, k) @ (k, k) = (2^k, k)
        S1 = mask1 @ lam_obs[:, 1, :]     mask0[i,z] = 1-Tt[i,z],  mask1[i,z]=Tt[i,z]

    Cost: O(k^2 * 2^k)  vs  naive O(k^4 * 2^k).
    """

    def __init__(self, P, k, device='cpu'):
        self.k      = k
        self.device = device

        T_types = list(product([0, 1], repeat=k))   # 2^k treatment response fns
        Y_types = list(product(range(k), repeat=2)) # k^2  potential outcome pairs

        self.n_Tt = len(T_types)                     # 2^k
        self.n_Yt = len(Y_types)                     # k^2
        self.n    = self.n_Tt * self.n_Yt            # dual constraints
        self.m    = 2 * k * k + 1                    # dual variables  (lam dim)

        # --- masks for structured matmul ---
        # Tt_arr[i, z] = Tt[z] for the i-th treatment response function
        Tt_arr = np.array(T_types, dtype=np.float32)               # (2^k, k)
        self.mask0 = torch.tensor(1.0 - Tt_arr, device=device)     # 1 where Tt[z]=0
        self.mask1 = torch.tensor(Tt_arr,       device=device)     # 1 where Tt[z]=1

        # --- objective c[Tt_idx, y0, y1] = y1 - y0  (independent of Tt) ---
        Yt_arr = np.array(Y_types, dtype=np.int32)                 # (k^2, 2)
        c_Yt   = torch.tensor(
            (Yt_arr[:, 1] - Yt_arr[:, 0]).astype(np.float32), device=device
        ).reshape(k, k)                                            # (k, k)
        # broadcast over Tt dimension: same c for every Tt
        self.c = c_Yt.unsqueeze(0).expand(self.n_Tt, k, k) \
                      .reshape(self.n).contiguous()               # (n,)

        # --- b: empirical distribution, stored as (k, 2, k) + scalar norm ---
        self.b_obs  = torch.tensor(P, dtype=torch.float32, device=device)  # (k,2,k)
        self.b_norm = torch.ones(1, dtype=torch.float32, device=device)

        # --- feature grid for neural network, one feature-vector per lam entry ---
        # lam_obs[z, t, y]  encoded as  (z/(k-1), 2t-1, y/(k-1))
        # lam_norm           encoded as  (-1, -1, -1)
        grid = [
            [z / (k - 1), 2 * t - 1, y / (k - 1)]
            for z in range(k) for t in [0, 1] for y in range(k)
        ]
        grid.append([-1.0, -1.0, -1.0])
        self.feats = torch.tensor(grid, dtype=torch.float32, device=device)  # (m, 3)

    # ------------------------------------------------------------------

    def AtLam(self, lam):
        """
        Compute  A^T @ lam  without forming A.   Cost: O(k^2 * 2^k)

        lam    : (m,)  = (2k^2 + 1,)  — dual variables
        returns: (n,)  = (2^k * k^2,) — one value per latent type
        """
        k        = self.k
        lam_obs  = lam[:-1].reshape(k, 2, k)  # (k, 2, k)
        lam_norm = lam[-1]

        S0 = self.mask0 @ lam_obs[:, 0, :]    # (2^k, k): S0[Tt, y0]
        S1 = self.mask1 @ lam_obs[:, 1, :]    # (2^k, k): S1[Tt, y1]

        # broadcast addition: (2^k,k,1) + (2^k,1,k) + scalar
        return (S0[:, :, None] + S1[:, None, :] + lam_norm).reshape(self.n)

    def b_lam(self, lam):
        """
        Compute  b^T @ lam  (dual objective).

        = sum_{z,t,y} P[z,t,y] * lam_obs[z,t,y]  +  lam_norm
        """
        k        = self.k
        lam_obs  = lam[:-1].reshape(k, 2, k)
        lam_norm = lam[-1]
        return (self.b_obs * lam_obs).sum() + lam_norm


# ----------------------------------------------------------------------

class ImplicitLP_EV:
    """
    Matrix-free LP operators for Edu_vs_Voting.

    Each latent type (x, d, y0, y1) participates in exactly one constraint:
        constraint index  (x, d, y_d)   where  y_d = y1 if d=1 else y0

    So  A^T lam  reduces to a gather + scalar add:
        (A^T lam)[x, d, y0, y1]  =  lam_obs[x, d, y_d]  +  lam_norm

    Cost: O(n) — no multiply, just array indexing.
    """

    def __init__(self, P, kx, ky, y_bins, device='cpu'):
        self.kx     = kx
        self.ky     = ky
        self.device = device

        T_types  = list(product(range(kx), [0, 1],
                                product(range(ky), range(ky))))
        self.n   = len(T_types)    # kx * 2 * ky^2
        self.m   = kx * 2 * ky + 1

        # -- index arrays, precomputed once --
        xs  = np.array([t[0]    for t in T_types], dtype=np.int64)
        ds  = np.array([t[1]    for t in T_types], dtype=np.int64)
        y0s = np.array([t[2][0] for t in T_types], dtype=np.int64)
        y1s = np.array([t[2][1] for t in T_types], dtype=np.int64)
        yd  = np.where(ds == 1, y1s, y0s)  # (n,) — observed potential outcome index

        # flat index into lam_obs.reshape(-1) where lam_obs has shape (kx, 2, ky)
        flat_idx = xs * (2 * ky) + ds * ky + yd
        self.flat_idx = torch.tensor(flat_idx, dtype=torch.long, device=device)

        # -- objective c[x, d, y0, y1] = y_centers[y1] - y_centers[y0] --
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        yc = torch.tensor(y_centers, dtype=torch.float32, device=device)
        self.c = yc[torch.tensor(y1s, dtype=torch.long)] \
               - yc[torch.tensor(y0s, dtype=torch.long)]   # (n,)

        # -- b --
        self.b_obs  = torch.tensor(P, dtype=torch.float32, device=device)   # (kx,2,ky)

        # -- feature grid --
        grid = [
            [x / (kx - 1), 2 * d - 1, float(y_centers[y])]
            for x in range(kx) for d in [0, 1] for y in range(ky)
        ]
        grid.append([-1.0, -1.0, -1.0])
        self.feats = torch.tensor(grid, dtype=torch.float32, device=device)  # (m, 3)

    # ------------------------------------------------------------------

    def AtLam(self, lam):
        """
        Compute  A^T @ lam  without forming A.   Cost: O(n)

        lam    : (m,)  = (kx*2*ky + 1,)
        returns: (n,)  = (kx * 2 * ky^2,)
        """
        lam_obs_flat = lam[:-1]            # (kx*2*ky,)
        lam_norm     = lam[-1]
        return lam_obs_flat[self.flat_idx] + lam_norm   # gather + broadcast

    def b_lam(self, lam):
        """b^T @ lam"""
        lam_obs  = lam[:-1].reshape(self.kx, 2, self.ky)
        lam_norm = lam[-1]
        return (self.b_obs * lam_obs).sum() + lam_norm


# ======================================================================
# Neural network  (identical role to DualModel2 in original solver)
# ======================================================================

class DualNet(nn.Module):
    """
    Maps  feats (m, 3)  ->  (lam_pos (m,), lam_neg (m,))
    so that  lam = lam_pos - lam_neg  is an unconstrained real vector.
    The 3-dim feature encodes the "coordinate" of each dual variable.
    """
    def __init__(self, hidden=32, num_layers=2):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.05)
            nn.init.normal_(m.bias,   0.0, 0.05)

    def forward(self, feats):
        out = self.net(feats)          # (m, 2)
        return out[:, 0], out[:, 1]   # lam_pos, lam_neg


# ======================================================================
# Core solver
# ======================================================================

def solve_dual_implicit(
    lp,                        # ImplicitLP_IV or ImplicitLP_EV
    upper        = False,
    steps_s1     = 30_000,     # Stage 1: aug-Lagrangian feasibility
    steps_s2     = 30_000,     # Stage 2: log-barrier central path
    lr_s1        = 5e-3,
    lr_s2        = 5e-4,
    hidden       = 32,
    num_layers   = 2,
    interior_margin = 1e-3,
    max_backtrack   = 30,
    name         = "run",
    seed         = 0,
):
    """
    Two-stage implicit neural dual solver.

    Stage 1 — Augmented Lagrangian
        Find a strictly feasible interior point  lam0  with
        min_i slack_i > 0  using a penalty / aug-Lagrangian approach.
        The NN is trained to output lam0.

    Stage 2 — Log-barrier central path
        Initialised from Stage 1 weights.  Maximises  b^T lam  subject to
        all dual constraints staying strictly positive via the barrier:
            loss = -b^T lam  +  mu * (-mean log(c - A^T lam))
        mu is annealed from 1 down towards 0.

    In both stages  A^T lam  is evaluated via  lp.AtLam(lam)  without
    ever forming the (m x n) matrix A.
    """
    torch.manual_seed(seed)
    device = lp.device

    sign = -1.0 if upper else 1.0   # upper bound flips objective sign
    c    = sign * lp.c              # (n,) — modified objective for upper bound

    # ----------------------------------------------------------------
    # Stage 1: Augmented Lagrangian — find strictly feasible lam
    # ----------------------------------------------------------------
    print(f"\n=== Stage 1: feasible interior point  ({steps_s1} steps) ===")
    print(f"  |lam| = {lp.m}   |dual constraints| = {lp.n}")

    model = DualNet(hidden, num_layers).to(device)
    opt1  = torch.optim.Adam(model.parameters(), lr=lr_s1)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, steps_s1)

    # Augmented-Lagrangian multipliers (one per dual constraint)
    nu  = torch.zeros(lp.n, device=device)
    rho = 1.0

    for step in range(steps_s1):
        lam_pos, lam_neg = model(lp.feats)
        lam  = lam_pos - lam_neg

        AtL   = lp.AtLam(lam)             # (n,) — matrix-free
        slack = c - AtL                   # (n,) — positive = feasible

        # violation: how much each constraint is violated
        viol  = torch.relu(-slack)        # (n,) >= 0

        dual_obj = sign * lp.b_lam(lam)

        # Augmented Lagrangian loss:  maximise dual_obj  subject to  slack >= 0
        loss = -dual_obj \
             + (nu * viol).sum() \
             + rho * (viol ** 2).sum()

        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()
        sched1.step()

        # Dual variable update (sub-gradient ascent on nu)
        with torch.no_grad():
            nu = torch.clamp(nu + rho * viol, min=0.0)

        # Increase penalty every 1000 steps
        if step % 1000 == 0:
            rho = min(rho * 1.5, 1e5)
            if step % 5000 == 0:
                with torch.no_grad():
                    print(
                        f"  step {step:6d} | "
                        f"dual {dual_obj.item():+.4f} | "
                        f"min slack {slack.min().item():+.6f} | "
                        f"violations {(viol > 0).sum().item()}"
                    )

    # Check interior point quality
    with torch.no_grad():
        lam_pos, lam_neg = model(lp.feats)
        lam_s1 = lam_pos - lam_neg
        slack_s1 = c - lp.AtLam(lam_s1)
        print(f"  Stage 1 done | min slack = {slack_s1.min().item():+.6e}")
        if slack_s1.min().item() <= 0:
            print("  WARNING: not strictly interior after Stage 1.")

    # Save Stage 1 weights
    tag = "upper" if upper else "lower"
    ckpt_path = f"{tag}_stage1_implicit.pt"
    torch.save(model.state_dict(), ckpt_path)

    # ----------------------------------------------------------------
    # Stage 2: Log-barrier central path
    # ----------------------------------------------------------------
    print(f"\n=== Stage 2: central path  ({steps_s2} steps) ===")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    opt2   = torch.optim.Adam(model.parameters(), lr=lr_s2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, steps_s2)

    mu      = 1.0
    mu_floor = 1e-6

    best_obj = -1e9 if not upper else 1e9
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for step in range(steps_s2):
        # --- save state for backtracking ---
        prev_state = {k: v.clone() for k, v in model.state_dict().items()}
        prev_lrs   = [g['lr'] for g in opt2.param_groups]

        # --- forward ---
        lam_pos, lam_neg = model(lp.feats)
        lam   = lam_pos - lam_neg
        AtL   = lp.AtLam(lam)        # matrix-free  A^T lam
        slack = c - AtL              # (n,)

        if (slack <= 0).any():
            slack = torch.clamp(slack, min=interior_margin)

        barrier  = -torch.log(slack).mean()
        dual_obj = lp.b_lam(lam)
        loss     = -sign * dual_obj + mu * barrier

        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        sched2.step()

        # --- feasibility check + backtracking ---
        with torch.no_grad():
            lam_pos_new, lam_neg_new = model(lp.feats)
            lam_new   = lam_pos_new - lam_neg_new
            slack_new = c - lp.AtLam(lam_new)

        if (slack_new <= 0).any():
            success = False
            for _ in range(max_backtrack):
                model.load_state_dict(prev_state)
                for g in opt2.param_groups:
                    g['lr'] *= 0.5

                lam_pos_bt, lam_neg_bt = model(lp.feats)
                lam_bt   = lam_pos_bt - lam_neg_bt
                slack_bt = c - lp.AtLam(lam_bt)
                slack_bt = torch.clamp(slack_bt, min=interior_margin)

                barrier_bt  = -torch.log(slack_bt).mean()
                dual_obj_bt = lp.b_lam(lam_bt)
                loss_bt     = -sign * dual_obj_bt + mu * barrier_bt

                opt2.zero_grad()
                loss_bt.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()

                with torch.no_grad():
                    lam_pos_bt, lam_neg_bt = model(lp.feats)
                    lam_bt   = lam_pos_bt - lam_neg_bt
                    slack_bt = c - lp.AtLam(lam_bt)
                if (slack_bt > 0).all():
                    success = True
                    break

            if not success:
                print(f"  [step {step}] backtracking failed — stopping Stage 2.")
                model.load_state_dict(prev_state)
                for g, lr in zip(opt2.param_groups, prev_lrs):
                    g['lr'] = lr
                break

        # --- anneal barrier parameter ---
        mu = max(10.0 / (step + 1), mu_floor)

        # --- track best feasible solution ---
        with torch.no_grad():
            lam_pos_ev, lam_neg_ev = model(lp.feats)
            lam_ev   = lam_pos_ev - lam_neg_ev
            obj_ev   = lp.b_lam(lam_ev).item()
            if (c - lp.AtLam(lam_ev) > 0).all():
                if (not upper and obj_ev > best_obj) or (upper and obj_ev < best_obj):
                    best_obj   = obj_ev
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if step % 1000 == 0:
            with torch.no_grad():
                lam_pos_l, lam_neg_l = model(lp.feats)
                lam_l   = lam_pos_l - lam_neg_l
                slack_l = c - lp.AtLam(lam_l)
                obj_l   = lp.b_lam(lam_l).item()
                print(
                    f"  step {step:6d} | "
                    f"dual {obj_l:+.4f} | "
                    f"min slack {slack_l.min().item():+.6f} | "
                    f"active {(slack_l < 1e-4).sum().item()} | "
                    f"mu {mu:.2e}"
                )

    # --- final evaluation from best state ---
    model.load_state_dict(best_state)
    with torch.no_grad():
        lam_pos_f, lam_neg_f = model(lp.feats)
        lam_f   = lam_pos_f - lam_neg_f
        dual_val = lp.b_lam(lam_f).item()
        slack_f  = c - lp.AtLam(lam_f)
        nu_f     = slack_f.min().item()

    return (
        lam_pos_f.cpu().numpy(),
        lam_neg_f.cpu().numpy(),
        dual_val,
        nu_f,
    )


# ======================================================================
# Main
# ======================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",              type=str,   default="IV_cont")
    p.add_argument("--distribution_gen",  type=str,   default="generate")
    p.add_argument("--k",                 type=int,   default=8)
    p.add_argument("--hidden",            type=int,   default=32)
    p.add_argument("--layers",            type=int,   default=2)
    p.add_argument("--steps_s1",          type=int,   default=30_000)
    p.add_argument("--steps_s2",          type=int,   default=30_000)
    p.add_argument("--lr_s1",             type=float, default=5e-3)
    p.add_argument("--lr_s2",             type=float, default=5e-4)
    p.add_argument("--n_pts",             type=int,   default=10_000)
    p.add_argument("--seed",              type=int,   default=2022)
    return p.parse_args()


if __name__ == "__main__":
    args   = get_args()
    SEED   = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    k    = args.k
    name = args.name

    # ------------------------------------------------------------------
    # Build implicit LP operator
    # ------------------------------------------------------------------
    if name == "IV_cont":
        if args.distribution_gen == "generate":
            print("Generating IV data...")
            data = generate_data_IV(args.n_pts, lam=0.5)
            P    = empirical_distribution_IV(data, k)
        else:
            P = np.load("./Data/IV_cont/P8.npy")

        lp = ImplicitLP_IV(P, k, device=device)
        print(f"IV_cont  |  k={k}  |  n={lp.n}  |  m={lp.m}")

    elif name == "Edu_vs_Voting":
        from Data.Edu_vs_Voting.LP_construction import empirical_distribution_EV
        kx = ky = k
        print("Generating Edu_vs_Voting data...")
        data, Y0, Y1 = generate_data_EV(args.n_pts, tau=0.5, seed=SEED)
        print(f"True ATE: {np.mean(Y1 - Y0):.4f}")
        P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
        lp = ImplicitLP_EV(P, kx, ky, y_bins, device=device)
        print(f"Edu_vs_Voting  |  k={k}  |  n={lp.n}  |  m={lp.m}")

    else:
        raise ValueError(f"Unknown name: {name}")

    # ------------------------------------------------------------------
    # Benchmark: implicit vs explicit A^T @ lam
    # ------------------------------------------------------------------
    print("\n--- A^T @ lam benchmark ---")
    lam_test = torch.randn(lp.m, device=device)

    N_BENCH = 200
    t0 = time.time()
    for _ in range(N_BENCH):
        _ = lp.AtLam(lam_test)
    t_implicit = (time.time() - t0) / N_BENCH * 1000
    print(f"  implicit:  {t_implicit:.3f} ms / call")

    # ------------------------------------------------------------------
    # Solve lower bound
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    t_start = time.time()

    lam_L_pos, lam_L_neg, dual_L, nu_L = solve_dual_implicit(
        lp,
        upper     = False,
        steps_s1  = args.steps_s1,
        steps_s2  = args.steps_s2,
        lr_s1     = args.lr_s1,
        lr_s2     = args.lr_s2,
        hidden    = args.hidden,
        num_layers= args.layers,
        name      = name,
        seed      = SEED,
    )

    # ------------------------------------------------------------------
    # Solve upper bound
    # ------------------------------------------------------------------
    lam_U_pos, lam_U_neg, dual_U, nu_U = solve_dual_implicit(
        lp,
        upper     = True,
        steps_s1  = args.steps_s1,
        steps_s2  = args.steps_s2,
        lr_s1     = args.lr_s1,
        lr_s2     = args.lr_s2,
        hidden    = args.hidden,
        num_layers= args.layers,
        name      = name,
        seed      = SEED,
    )

    t_end = time.time()

    # ------------------------------------------------------------------
    # Final bounds
    # ------------------------------------------------------------------
    b_np = np.concatenate([P.reshape(-1), [1.0]])   # (m,)
    lower =  (b_np * (lam_L_pos - lam_L_neg)).sum()
    upper = -(b_np * (lam_U_pos - lam_U_neg)).sum()

    print("\n" + "="*60)
    print("BOUNDS")
    print(f"  Lower : {lower:.4f}   (nu = {nu_L:.2e})")
    print(f"  Upper : {upper:.4f}   (nu = {nu_U:.2e})")
    if name == "IV_cont":
        print("  True ATE = 3.0")
        labels_plot = list(product([0,1], repeat=k))
        plot_dual_heatmap(lam_L_pos - lam_L_neg, [], k,
                          f"Lower  {lower:.4f}")
        plot_dual_heatmap(lam_U_pos - lam_U_neg, [], k,
                          f"Upper  {upper:.4f}")
    elif name == "Edu_vs_Voting":
        print("  True ATE = 0.5")
    print(f"  Wall time: {t_end - t_start:.1f}s")