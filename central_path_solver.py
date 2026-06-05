"""
direct_dual_solver.py
=====================
Central-path algorithm on the original LP dual problem with explicit A, b, c
matrices and a direct dual variable vector — no neural network.

Dual LP:
    max  b^T lam
    s.t. A^T lam <= c

Algorithm mirrors neural_dual_solver2.py exactly:
    Warm-start phase  — push all slacks above interior_margin using a
                        max-violation loss (no barrier). Stops early once
                        every slack exceeds interior_margin.
    Central-path loop — log-barrier loss = -b^T lam + mu * (-log slack).min()
                        with AdamW + CosineAnnealingLR and backtracking.
                        mu is annealed adaptively.
"""

import argparse
import numpy as np
import torch
import time
import wandb

from Data.IV_cont.LP_construction import (
    generate_data_IV, empirical_distribution_IV, build_constraints_IV,
)
from Data.IV_cont.utils import plot_dual_heatmap
from Data.Edu_vs_Voting.LP_construction import (
    generate_data_EV, empirical_distribution_EV, build_constraints_EV,
)


def solve_dual(
    A,
    b,
    c,
    upper            = False,
    steps            = 150_000,
    lr               = 1e-3,
    interior_margin  = 1e-3,
    max_backtrack    = 50,
    warm_start_steps = 2000,
    name             = "run",
):
    """
    Parameters
    ----------
    A                : (m, n) numpy array  — LP constraint matrix
    b                : (m,)  numpy array   — dual objective / primal RHS
    c                : (n,)  numpy array   — primal objective (slack = c - A^T lam)
    upper            : if True, flip sign of c to solve for upper bound
    steps            : central-path optimisation steps
    lr               : AdamW learning rate
    interior_margin  : target minimum slack during warm-start
    max_backtrack    : maximum backtracking iterations per step
    warm_start_steps : warm-start budget (may exit early)
    name             : run name for wandb
    """
    if wandb.run is not None:
        wandb.finish()

    bound_tag = "UpperBound" if upper else "LowerBound"
    try:
        wandb.init(
            project = "DirectDualSolver",
            name    = f"{name}_{bound_tag}",
            config  = {
                "steps": steps, "lr": lr,
                "m": int(A.shape[0]), "n": int(A.shape[1]),
            },
            reinit = True,
        )
    except Exception:
        wandb.init(mode="disabled", reinit=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sign = -1 if upper else 1

    A_t = torch.tensor(A,          dtype=torch.float32, device=device)  # (m, n)
    b_t = torch.tensor(b,          dtype=torch.float32, device=device)  # (m,)
    c_t = torch.tensor(sign * c,   dtype=torch.float32, device=device)  # (n,)

    m, n = A_t.shape
    print(f"\nDual variables (m): {m}")
    print(f"Dual constraints (n): {n}")

    # lam is the direct dual variable — a single optimisable parameter
    lam = torch.nn.Parameter(torch.zeros(m, device=device))

    # ------------------------------------------------------------------
    # Warm-start: push all slacks above interior_margin
    # ------------------------------------------------------------------
    opt_ws = torch.optim.AdamW([lam], lr=1e-2)

    for ws_step in range(warm_start_steps):
        slack     = c_t - A_t.t() @ lam          # (n,)
        violation = torch.relu(interior_margin - slack)
        loss      = violation.max()

        opt_ws.zero_grad()
        loss.backward()
        opt_ws.step()

        if loss.item() == 0.0:
            print(f"[Warm-start] converged at step {ws_step} — all slacks > interior_margin")
            break

    with torch.no_grad():
        slack_ws = c_t - A_t.t() @ lam
        print(f"[Warm-start] min slack = {slack_ws.min().item():.6e}  "
              f"violations = {(slack_ws < 0).sum().item()}")

    # ------------------------------------------------------------------
    # Central-path loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW([lam], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    max_mu = 1e-3
    t      = 0
    mu     = 1.0 / (t + 1)

    # placeholders so the final-readout block is always valid
    slack    = c_t - A_t.t() @ lam.detach()
    dual_obj = b_t @ lam.detach()
    loss     = -dual_obj

    for step in range(steps):
        # save state for backtracking
        lam_prev = lam.data.clone()
        lr_prev  = [g["lr"] for g in optimizer.param_groups]

        # ---- forward ----
        slack    = c_t - A_t.t() @ lam           # (n,)
        dual_obj = b_t @ lam

        if (slack <= 0).any():
            print(f"[step {step}] infeasible before step, "
                  f"min_slack={slack.min().item():.6e}")

        # ---- backtracking if infeasible ----
        if (slack <= 0).any():
            success = False
            for _ in range(max_backtrack):
                lam.data.copy_(lam_prev)
                for g in optimizer.param_groups:
                    g["lr"] *= 0.8

                slack_bt    = c_t - A_t.t() @ lam + interior_margin
                barrier_bt  = -torch.log(slack_bt).mean()
                dual_obj_bt = b_t @ lam
                loss_bt     = -dual_obj_bt + mu * barrier_bt

                optimizer.zero_grad()
                loss_bt.backward()
                torch.nn.utils.clip_grad_norm_([lam], 1.0)
                optimizer.step()

                with torch.no_grad():
                    slack_bt = c_t - A_t.t() @ lam

                if (slack_bt > 0).all():
                    success  = True
                    slack    = slack_bt
                    dual_obj = b_t @ lam
                    break

            if not success:
                smallest5 = torch.topk(slack_bt, 5, largest=False).values
                print(
                    f"[step {step}] backtracking failed — stopping.\n"
                    f"  dual {sign * dual_obj_bt.item():.4f} | "
                    f"min slack {slack_bt.min().item():.6f} | "
                    f"top5 {smallest5.detach().cpu().numpy()}"
                )
                lam.data.copy_(lam_prev)
                for g, old_lr in zip(optimizer.param_groups, lr_prev):
                    g["lr"] = old_lr
                break

        # ---- barrier loss ----
        barrier  = -torch.log(slack).min()
        dual_obj = b_t @ lam
        loss     = -dual_obj + mu * barrier

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([lam], 1.0)
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
                    "step":       step,
                    "dual_obj":   (sign * dual_obj).item(),
                    "min_slack":  slack.min().item(),
                    "max_slack":  slack.max().item(),
                    "mean_slack": slack.mean().item(),
                    "num_active": (slack < 1e-4).sum().item(),
                    "loss":       loss.item(),
                    "mu":         mu,
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
        slack_f = c_t - A_t.t() @ lam
        nu      = slack_f.min().item()

    return lam.detach().cpu().numpy(), nu


# ======================================================================
# CLI
# ======================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",             type=str,   default="IV_cont")
    p.add_argument("--distribution_gen", type=str,   default="generate")
    p.add_argument("--k",                type=int,   default=10)
    p.add_argument("--steps",            type=int,   default=150_000)
    p.add_argument("--warm_start",       type=int,   default=2000)
    p.add_argument("--lr_lower",         type=float, default=1e-3)
    p.add_argument("--lr_upper",         type=float, default=1e-3)
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

    wandb_name = name + f"_k{k}_steps{n}"

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------
    if name == "IV_cont":
        if dist == "generate":
            print("Generating data...")
            data = generate_data_IV(args.n_pts, lam=0.5)
            P    = empirical_distribution_IV(data, k)
        elif dist == "pre-load":
            paths = {6:  "./Data/IV_cont/P6.npy",
                     8:  "./Data/IV_cont/P8.npy",
                     10: "./Data/IV_cont/P10.npy"}
            if k not in paths:
                raise FileNotFoundError(f"No pre-saved P for k={k}")
            P = np.load(paths[k])
        else:
            raise ValueError(f"Unknown distribution_gen: {dist}")

        print("Building LP system...")
        A, b, c, labels = build_constraints_IV(P, k)
        y_centers = None

    elif name == "Edu_vs_Voting":
        kx = ky = k
        if dist == "generate":
            print("Generating data...")
            data, Y0, Y1 = generate_data_EV(args.n_pts, tau=0.5, seed=SEED)
            print(f"True ATE: {np.mean(Y1 - Y0):.4f}")
            P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
        elif dist == "pre-load":
            paths = {12: "./Data/Edu_vs_Voting/P12_EV.npz",
                     16: "./Data/Edu_vs_Voting/P16_EV.npz"}
            if k not in paths:
                raise FileNotFoundError(f"No pre-saved P for k={k}")
            data_np = np.load(paths[k])
            P, x_bins, y_bins = data_np["P"], data_np["x_bins"], data_np["y_bins"]
        else:
            raise ValueError(f"Unknown distribution_gen: {dist}")

        print("Building LP system...")
        A, b, c, labels = build_constraints_EV(P, kx, ky, y_bins)

    else:
        raise ValueError(f"Unknown name: {name}")

    print(f"A shape: {A.shape}  b shape: {b.shape}  c shape: {c.shape}")

    # ------------------------------------------------------------------
    # Solve lower and upper bounds
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    t_start = time.time()

    print("Solving LOWER bound...")
    lam_L, nu_L = solve_dual(
        A, b, c,
        upper            = False,
        steps            = n,
        lr               = args.lr_lower,
        interior_margin  = 1e-3,
        max_backtrack    = 50,
        warm_start_steps = args.warm_start,
        name             = wandb_name,
    )

    print("\nSolving UPPER bound...")
    lam_U, nu_U = solve_dual(
        A, b, c,
        upper            = True,
        steps            = n,
        lr               = args.lr_upper,
        interior_margin  = 1e-3,
        max_backtrack    = 50,
        warm_start_steps = args.warm_start,
        name             = wandb_name,
    )

    t_end = time.time()

    # ------------------------------------------------------------------
    # Compute and report bounds
    # ------------------------------------------------------------------
    lower =  (b * lam_L).sum()
    upper = -(b * lam_U).sum()

    print("\n" + "=" * 60)
    print("BOUNDS")
    print(f"  Lower : {lower:.4f}   (nu = {nu_L:.2e})")
    print(f"  Upper : {upper:.4f}   (nu = {nu_U:.2e})")
    if name == "IV_cont":
        print("  True ATE = 3.0")
        plot_dual_heatmap(lam_L, labels[:-1], k, f"Lower {lower:.4f}")
        plot_dual_heatmap(lam_U, labels[:-1], k, f"Upper {upper:.4f}")
    elif name == "Edu_vs_Voting":
        print("  True ATE = 0.5")
    print(f"  Wall time: {t_end - t_start:.1f}s")