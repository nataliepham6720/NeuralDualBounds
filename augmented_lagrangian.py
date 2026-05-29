import argparse
import numpy as np
from itertools import product
from scipy.stats import truncnorm
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import wandb
import osqp
import scipy.sparse as sp

from Data.IV_cont.LP_construction import * 
from Data.IV_cont.utils import *

from Data.Edu_vs_Voting.LP_construction import * 
# from Data.Edu_vs_Voting.utils import *

EPS_TOL = 0 #1e-6
K_active = 10

SLACK_TOL = 1e-3
NU_TOL = 1e-9


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="experiment") # IV_cont or Edu_vs_Voting
    parser.add_argument("--distribution_gen", type=str, default="generate")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--lr_lower", type=float, default=5e-3)
    parser.add_argument("--lr_upper", type=float, default=5e-4)

    parser.add_argument("--n_pts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


SEED = 2022
print("Seed:", SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)

def project_lambda_qp(lam_np, A_np, c_np, nu):
    """
    Solve:
        min 1/2 ||λ - λ0||^2
        s.t. A^T λ <= c - nu
    """

    n = lam_np.shape[0]

    # P = I
    P = sp.eye(n, format='csc')

    # q = -λ0
    q = -lam_np

    # Constraints: A^T λ <= c - nu
    G = A_np.T  # shape (m, n)

    l = -np.inf * np.ones(G.shape[0])
    u = c_np - nu

    G = sp.csc_matrix(G)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=G, l=l, u=u, verbose=False)

    res = prob.solve()

    if res.info.status != 'solved':
        print("⚠️ QP projection did not fully solve:", res.info.status)

    return res.x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def solve_dual_cp_gd(A, b, c, labels, k, y_centers=None, upper=False, 
                  steps=3000, lr=1e-5, name="Run1"):
    if wandb.run is not None:
        wandb.finish()

    if not upper:
        name = name + "_LowerBound"
    else:
        name = name + "_UpperBound"

    wandb.init(
        project="NeuralDualSolver",
        name=name,
        config={"steps": steps, "lr": lr, "k": k},
        reinit=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sign = -1 if upper else 1
    c = sign * c

    A = torch.tensor(A, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)
    c = torch.tensor(c, dtype=torch.float32, device=device)

    print("\nDual variables:", len(b))

    lam = torch.zeros_like(b, requires_grad=True)
    optimizer = torch.optim.Adam([lam], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # mu = 1
    nu = torch.zeros_like(c)   # size = number of constraints
    rho = 1.0

    for step in range(steps):
        optimizer.zero_grad()

        slack = c - (A.t() @ lam)

        violation = torch.relu(-(c - (A.t() @ lam )))#+ nu)))
        penalty = 1 * violation.max()

        dual_obj = (b+EPS_TOL)@lam 
        nu = torch.clamp(nu + rho * violation.detach(), min=0)
        loss = -dual_obj \
              + (nu * violation).sum() \
              + rho * (violation**2).mean()
        # nu = nu + rho * violation.detach()

        loss.backward()
        optimizer.step()
        scheduler.step()

        # mu = min(mu * 1.002, 100)

        if step % 10000 == 0:
            with torch.no_grad():
                wandb.log({
                    "step": step,
                    "dual_obj": (sign * dual_obj).item(),
                    "min_slack": slack.min().item(),
                    "max_slack": slack.max().item(),
                    "mean_slack": slack.mean().item(),
                    "num_active": (slack < 1e-4).sum().item(),
                    "loss": loss.item(),
                    # "mu": mu,
                })

                smallest5 = torch.topk(slack, 5, largest=False).values
                print(
                    f"step {step} | dual {sign * dual_obj.item():.4f} | "
                    f"min slack {slack.min().item():.6f} | "
                    f"max slack {slack.max().item():.6f} | "
                    f"top5 smallest {smallest5.detach().cpu().numpy()} | "
                    # f"mu {mu:.3e}"
                )

    dual_value = dual_obj.item()
    if upper:
        dual_value = -dual_value

    with torch.no_grad():
        slack = c - (A.t() @ lam)
        nu = slack.min().item()

    # ---------- Final projection ----------
    with torch.no_grad():
        lam_np = lam.detach().cpu().numpy()
        A_np = A.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()

        # project to A^T λ <= c - NU_TOL
        lam_proj = project_lambda_qp(lam_np, A_np, c_np, nu=0)

        slack_proj = c_np - A_np.T @ lam_proj
        print("\nProjection:"
            f" min slack={slack_proj.min():.3e},"
            f" max violation={max(0,-slack_proj.min()):.3e}")
        
        # lam_final = torch.tensor(lam_proj,dtype=torch.float32)

    return lam_proj

# Main

if __name__ == "__main__":
    args = get_args()
    
    n = args.steps
    n_pts = args.n_pts
    k = args.k
    name = args.name
    dist = args.distribution_gen

    if name == "IV_cont":
        wandb_name = name + f"_k{k}_steps{n}_hidden{args.hidden}_layers{args.layers}"
        
        if dist == "generate":
            print("Generating data...")
            data = generate_data_IV(n_pts, lam=0.5)

            print("Estimating distribution...")
            P = empirical_distribution_IV(data, k)
        elif dist == "pre-load":
            print("Loading saved distribution...")
            P = np.load("./Data/IV_cont/P8.npy")

        print("Building LP system...")
        A, b, c, labels = build_constraints_IV(P, k)
        y_centers = None

    elif name == "Edu_vs_Voting":
        wandb_name = name + f"_k{k}_steps{n}_hidden{args.hidden}_layers{args.layers}"

        kx = ky = args.k

        print("Generating data...")
        data, Y0, Y1 = generate_data_EV(n_pts, tau=0.5, seed=SEED)
        ATE_true = np.mean(Y1 - Y0)
        print("True ATE:", ATE_true)

        print("Estimating distribution...")
        P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2 

        print("Building LP system...")
        A, b, c, labels = build_constraints_EV(P, kx, ky, y_bins)

    print("\n===== SIZE CHECK =====")
    print("A shape:", A.shape)
    print("b shape:", b.shape)
    print("c shape:", c.shape)

    print("\nTraining neural dual...")
    start = time.time()

    lamL_pos = solve_dual_cp_gd(A, b, c, labels, k, y_centers=None, upper=False, 
                  steps=n, lr=args.lr_lower, name=wandb_name) 
    lamU_pos = solve_dual_cp_gd(A, b, c, labels, k, y_centers=None, upper=True, 
                  steps=n, lr=args.lr_upper, name=wandb_name) 

    b_obs = b #[:-1]
    # print(b.type(), lamL_pos.type())
    lower = ((b_obs+EPS_TOL)@lamL_pos) #- (b_obs-EPS_TOL)@lamL_neg) #+ nuL)
    upper = -((b_obs+EPS_TOL)@lamU_pos) #- (b_obs-EPS_TOL)@lamU_neg) # + nuU)

    end = time.time()

    print("\n===== BOUNDS =====")
    if name == "Edu_vs_Voting":
        print(f"NN lower bound : {lower:.4f}")
        print(f"NN upper bound : {upper:.4f}")
        # print("True ATE = 0.5")
    elif name == "IV_cont":
        print(f"NN lower bound : {lower:.4f}")
        print(f"NN upper bound : {upper:.4f}")
        print("True ATE = 3")

        plot_dual_heatmap(lamL_pos, labels[:-1], k, f"Lower Bound Dual - {lower:.4f}")
        plot_dual_heatmap(lamU_pos, labels[:-1], k, f"Upper Bound Dual - {upper:.4f}")
    
    print("Time taken: ", end-start)