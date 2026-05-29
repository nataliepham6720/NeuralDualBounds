"""
implicit_neural_solver2.py
==========================
Mirrors the algorithm in neural_dual_solver2.py exactly, but replaces the
explicit (m x n) matrix A with the matrix-free ImplicitLP operators from
implicit_neural_solver.py.

Algorithm (unchanged from neural_dual_solver2):
    Warm-start phase  — push all dual-constraint slacks above interior_margin
                        using a max-violation loss (no barrier).
    Central-path loop — log-barrier  loss = -b^T lam + mu * (-log slack).min()
                        with AdamW + CosineAnnealingLR and backtracking.
                        mu is annealed adaptively: drops every `interval` steps
                        where interval shrinks as t grows.

What changed vs neural_dual_solver2:
    - A.t() @ lam   →   lp.AtLam(lam)          (matrix-free, O(k^2·2^k) or O(n))
    - b @ lam        →   lp.b_lam(lam)          (O(m) dot product)
    - c tensor       →   lp.c  (sign-flipped)   (precomputed, never stored as dense A)
    - feature grid   →   lp.feats               (precomputed in ImplicitLP)
    - No build_constraints_IV/EV call needed     (A and b never materialised)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import wandb
from itertools import product

from Data.IV_cont.LP_construction  import generate_data_IV, empirical_distribution_IV
from Data.IV_cont.utils            import plot_dual_heatmap
from Data.Edu_vs_Voting.LP_construction import (
    generate_data_EV, empirical_distribution_EV,
)
# from implicit_neural_solver import ImplicitLP_IV, ImplicitLP_EV

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
# Hyper-parameters (same as neural_dual_solver2)
# ======================================================================

EPS_TOL = 0


# ======================================================================
# Neural network  (identical to neural_dual_solver2.DualNet / DualModel)
# ======================================================================

class DualNet(nn.Module):
    """
    Single-output network: feats (m, 3) → lam_pos (m,).
    lam = lam_pos  (non-negative representation; upper bound flips sign of c).
    """
    def __init__(self, i=3, h=10, num_layers=2):
        super().__init__()
        layers = [nn.Linear(i, h), nn.LayerNorm(h), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(h, h), nn.LayerNorm(h), nn.Tanh()]
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.05)
            nn.init.normal_(m.bias,   0.0, 0.05)

    def forward(self, x):
        return self.net(x)[:, 0]   # (m,)


class DualModel(nn.Module):
    def __init__(self, i=3, h=6, num_layers=2):
        super().__init__()
        self.net = DualNet(i, h, num_layers)

    def forward(self, feats):
        return self.net(feats)     # (m,) — lam_pos


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================
# Core solver  (mirrors neural_dual_solver2.solve_dual_nn)
# ======================================================================

def solve_dual_nn(
    lp,
    upper            = False,
    steps            = 150_000,
    lr               = 1e-5,
    hidden           = 5,
    layers           = 2,
    interior_margin  = 1e-3,
    max_backtrack    = 50,
    warm_start_steps = 2000,
    name             = "run",
):
    """
    Parameters
    ----------
    lp               : ImplicitLP_IV or ImplicitLP_EV
    upper            : if True, solve for the upper bound (flips sign of c)
    steps            : central-path training steps
    lr               : learning rate for central-path phase
    hidden, layers   : DualNet architecture
    interior_margin  : target minimum slack during warm-start
    max_backtrack    : maximum backtracking iterations per step
    warm_start_steps : warm-start steps before central path
    name             : run name for wandb
    """
    if wandb.run is not None:
        wandb.finish()

    bound_tag = "UpperBound" if upper else "LowerBound"
    try:
        wandb.init(
            project = "NeuralDualSolver_Implicit",
            name    = f"{name}_{bound_tag}",
            config  = {"steps": steps, "lr": lr,
                       "hidden": hidden, "layers": layers,
                       "n": lp.n, "m": lp.m},
            reinit  = True,
        )
    except Exception:
        wandb.init(mode="disabled", reinit=True)

    device = lp.device

    # upper bound: flip sign of c so we still maximise b^T lam
    sign = -1 if upper else 1
    c    = sign * lp.c      # (n,) — on device, no copy of A

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = DualModel(i=lp.feats.shape[1], h=hidden, num_layers=layers).to(device)
    print(f"\nDual variables (m): {lp.m}")
    print(f"Dual constraints (n): {lp.n}")
    print(f"NN parameters: {count_params(model)}")
    print(f"Compression ratio: {lp.m / count_params(model):.2f}x")

    # ------------------------------------------------------------------
    # Warm-start: push all slacks above interior_margin
    # ------------------------------------------------------------------
    model.train()
    opt_ws = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for ws_step in range(warm_start_steps):
        lam   = model(lp.feats)               # (m,)
        AtL   = lp.AtLam(lam)                 # (n,) — matrix-free
        slack = c - AtL                       # (n,)

        violation = torch.relu(interior_margin - slack)
        loss      = violation.max()

        opt_ws.zero_grad()
        loss.backward()
        opt_ws.step()

        # stop early once every slack exceeds interior_margin
        if loss.item() == 0.0:
            print(f"[Warm-start] converged at step {ws_step} — all slacks > interior_margin")
            break

    with torch.no_grad():
        lam   = model(lp.feats)
        slack = c - lp.AtLam(lam)
        print(f"[Warm-start] min slack = {slack.min().item():.6e}  "
              f"violations = {(slack < 0).sum().item()}")

    # ------------------------------------------------------------------
    # Central-path loop  (identical logic to neural_dual_solver2)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    max_mu = 1e-3
    t      = 0
    mu     = 1.0 / (t + 1)

    # last confirmed-feasible checkpoint — updated only when slack > 0 before a step
    last_feasible_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_feasible_lr    = [g["lr"] for g in optimizer.param_groups]

    for step in range(steps):
        # ---- forward ----
        lam   = model(lp.feats)               # (m,)
        AtL   = lp.AtLam(lam)                 # (n,) — matrix-free
        slack = c - AtL                       # (n,)

        if (slack <= 0).any():
            print(f"[step {step}] infeasible before step, "
                  f"min_slack={slack.min().item():.6e}")

        # ---- backtracking if infeasible ----
        if (slack <= 0).any():
            success = False
            for _ in range(max_backtrack):
                # restore from last CONFIRMED feasible state, not current infeasible one
                model.load_state_dict(last_feasible_state)
                for g, old_lr in zip(optimizer.param_groups, last_feasible_lr):
                    g["lr"] = old_lr * 0.8

                lam_bt   = model(lp.feats)
                AtL_bt   = lp.AtLam(lam_bt)
                # add interior_margin shift (same as neural_dual_solver2)
                slack_bt = c - AtL_bt + interior_margin

                barrier_bt  = -torch.log(slack_bt).mean()
                dual_obj_bt = lp.b_lam(lam_bt)
                loss_bt     = -dual_obj_bt + mu * barrier_bt

                optimizer.zero_grad()
                loss_bt.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    lam_bt   = model(lp.feats)
                    slack_bt = c - lp.AtLam(lam_bt)

                if (slack_bt > 0).all():
                    success  = True
                    slack    = slack_bt
                    dual_obj = lp.b_lam(lam_bt)
                    # update feasible checkpoint and shrink lr permanently
                    last_feasible_state = {k: v.clone() for k, v in model.state_dict().items()}
                    last_feasible_lr    = [g["lr"] for g in optimizer.param_groups]
                    break
                # shrink the lr used in the next backtrack attempt
                for g in optimizer.param_groups:
                    g["lr"] *= 0.8
                last_feasible_lr = [g["lr"] for g in optimizer.param_groups]

            if not success:
                smallest5 = torch.topk(slack_bt, 5, largest=False).values
                print(
                    f"[step {step}] backtracking failed — stopping.\n"
                    f"  dual {sign * dual_obj_bt.item():.4f} | "
                    f"min slack {slack_bt.min().item():.6f} | "
                    f"top5 {smallest5.detach().cpu().numpy()}"
                )
                model.load_state_dict(last_feasible_state)
                for g, old_lr in zip(optimizer.param_groups, last_feasible_lr):
                    g["lr"] = old_lr
                break
            
            lam   = model(lp.feats)
            slack = c - lp.AtLam(lam)
        else:
            # current state is feasible — checkpoint it before we step
            last_feasible_state = {k: v.clone() for k, v in model.state_dict().items()}
            last_feasible_lr    = [g["lr"] for g in optimizer.param_groups]

        # ---- barrier loss (uses .min() as in neural_dual_solver2) ----
        barrier  = -torch.log(slack).min()
        dual_obj = lp.b_lam(lam)
        loss     = -dual_obj + mu * barrier

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- adaptive mu schedule (identical to neural_dual_solver2) ----
        interval = (max(2500, int(100_000 / (t + 1)))
                    if mu > max_mu * 10 else 2000)
        if step % interval == 0:
            t  += 1
            mu  = max(1.0 / t, max_mu)

        # ---- logging ----
        if step % 1000 == 0:
            with torch.no_grad():
                wandb.log({
                    "step":      step,
                    "dual_obj":  (sign * dual_obj).item(),
                    "min_slack": slack.min().item(),
                    "max_slack": slack.max().item(),
                    "mean_slack":slack.mean().item(),
                    "num_active":(slack < 1e-4).sum().item(),
                    "loss":      loss.item(),
                    "mu":        mu,
                })
            if step % 2000 == 0:
                smallest5 = torch.topk(slack, 5, largest=False).values
                print(
                    f"step {step:7d} | "
                    f"dual {sign * dual_obj.item():+.4f} | "
                    f"min slack {slack.min().item():+.6f} | "
                    f"max slack {slack.max().item():+.6f} | "
                    f"top5 {smallest5.detach().cpu().numpy()} | "
                    f"mu {mu:.3e}"
                )

    # ------------------------------------------------------------------
    # Final readout
    # ------------------------------------------------------------------
    with torch.no_grad():
        lam_final = model(lp.feats)
        slack_f   = c - lp.AtLam(lam_final)
        nu        = slack_f.min().item()

    return lam_final.detach().cpu().numpy(), nu


# ======================================================================
# CLI
# ======================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",             type=str,   default="IV_cont")
    p.add_argument("--distribution_gen", type=str,   default="generate")
    p.add_argument("--k",                type=int,   default=10)
    p.add_argument("--hidden",           type=int,   default=5)
    p.add_argument("--layers",           type=int,   default=2)
    p.add_argument("--steps",            type=int,   default=150_000)
    p.add_argument("--warm_start",       type=int,   default=2000)
    p.add_argument("--lr_lower",         type=float, default=1e-5)
    p.add_argument("--lr_upper",         type=float, default=1e-5)
    p.add_argument("--n_pts",            type=int,   default=10_000)
    p.add_argument("--seed",             type=int,   default=2020)
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    print("Seed:", SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    k    = args.k
    name = args.name
    dist = args.distribution_gen
    n    = args.steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wandb_name = name + f"_k{k}_steps{n}_h{args.hidden}_l{args.layers}"

    # ------------------------------------------------------------------
    # Build implicit LP operator  (no A matrix ever allocated)
    # ------------------------------------------------------------------
    if name == "IV_cont":
        if dist == "generate":
            print("Generating data...")
            data = generate_data_IV(args.n_pts, lam=0.5)
            P    = empirical_distribution_IV(data, k)
        elif dist == "pre-load":
            print("Loading saved distribution...")
            paths = {6: "./Data/IV_cont/P6.npy",
                     8: "./Data/IV_cont/P8.npy",
                     10: "./Data/IV_cont/P10.npy"}
            if k not in paths:
                raise FileNotFoundError(f"No pre-saved P for k={k}")
            P = np.load(paths[k])
        else:
            raise ValueError(f"Unknown distribution_gen: {dist}")

        lp = ImplicitLP_IV(P, k, device=device)
        print(f"\nIV_cont  k={k}  n={lp.n}  m={lp.m}")

    elif name == "Edu_vs_Voting":
        kx = ky = k
        if dist == "generate":
            print("Generating data...")
            data, Y0, Y1 = generate_data_EV(args.n_pts, tau=0.5, seed=SEED)
            print(f"True ATE: {np.mean(Y1 - Y0):.4f}")
            P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
        elif dist == "pre-load":
            print("Loading saved distribution...")
            paths = {12: "./Data/Edu_vs_Voting/P12_EV.npz",
                     16: "./Data/Edu_vs_Voting/P16_EV.npz"}
            if k not in paths:
                raise FileNotFoundError(f"No pre-saved P for k={k}")
            data_np = np.load(paths[k])
            P, x_bins, y_bins = data_np["P"], data_np["x_bins"], data_np["y_bins"]
        else:
            raise ValueError(f"Unknown distribution_gen: {dist}")

        lp = ImplicitLP_EV(P, kx, ky, y_bins, device=device)
        print(f"\nEdu_vs_Voting  k={k}  n={lp.n}  m={lp.m}")

    else:
        raise ValueError(f"Unknown name: {name}")

    # ------------------------------------------------------------------
    # A^T @ lam benchmark
    # ------------------------------------------------------------------
    print("\n--- A^T @ lam benchmark (200 calls) ---")
    _lam = torch.randn(lp.m, device=device)
    _t0  = time.time()
    for _ in range(200):
        _ = lp.AtLam(_lam)
    print(f"  {(time.time() - _t0) / 200 * 1000:.3f} ms / call")
    del _lam

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    t_start = time.time()

    print("Solving LOWER bound...")
    lam_L, nu_L = solve_dual_nn(
        lp,
        upper            = False,
        steps            = n,
        lr               = args.lr_lower,
        hidden           = args.hidden,
        layers           = args.layers,
        interior_margin  = 1e-3,
        max_backtrack    = 50,
        warm_start_steps = args.warm_start,
        name             = wandb_name,
    )

    print("\nSolving UPPER bound...")
    lam_U, nu_U = solve_dual_nn(
        lp,
        upper            = True,
        steps            = n,
        lr               = args.lr_upper,
        hidden           = args.hidden,
        layers           = args.layers,
        interior_margin  = 1e-3,
        max_backtrack    = 50,
        warm_start_steps = args.warm_start,
        name             = wandb_name,
    )

    t_end = time.time()

    # ------------------------------------------------------------------
    # Compute bounds from the returned lam vectors
    # ------------------------------------------------------------------
    # b is stored inside lp as lp.b_obs (shape (k,2,k) or (kx,2,ky)) + norm=1
    # b^T lam = (P * lam_obs).sum() + lam_norm  — same as lp.b_lam but on numpy
    b_flat = np.concatenate([P.reshape(-1), [1.0]])   # (m,)

    lower =  (b_flat * lam_L).sum()
    upper = -(b_flat * lam_U).sum()

    print("\n" + "="*60)
    print("BOUNDS")
    print(f"  Lower : {lower:.4f}   (nu = {nu_L:.2e})")
    print(f"  Upper : {upper:.4f}   (nu = {nu_U:.2e})")
    if name == "IV_cont":
        print("  True ATE = 3.0")
        plot_dual_heatmap(lam_L, [], k, f"Lower {lower:.4f}")
        plot_dual_heatmap(lam_U, [], k, f"Upper {upper:.4f}")
    elif name == "Edu_vs_Voting":
        print("  True ATE = 0.5")
    print(f"  Wall time: {t_end - t_start:.1f}s")
