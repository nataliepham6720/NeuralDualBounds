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

EPS_TOL = 1e-6
K_active = 10


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


SEED = 2020
print("Seed:", SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)

class DualNet(nn.Module):
    def __init__(self, h=10, num_layers=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(3, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(h, 2))

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        out = self.net(x)
        lam_pos = out[:, 0]
        lam_neg = out[:, 1]
        return lam_pos, lam_neg


class DualNet2(nn.Module):
    def __init__(self, h=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, h),
            nn.Tanh(),
        )

        self.gate = nn.Linear(h, 1)
        self.mag = nn.Linear(h, 1)

        self.apply(init_weights)

    def forward(self, x):
        h = self.net(x)

        gate = torch.sigmoid(self.gate(h))
        mag = torch.nn.functional.softplus(self.mag(h))

        lam = gate * mag

        lam_pos = torch.clamp(lam, min=0)
        lam_neg = torch.clamp(-lam, min=0)

        return lam_pos.squeeze(), lam_neg.squeeze()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        nn.init.normal_(m.bias, mean=0.0, std=0.05)


class DualModel(nn.Module):
    def __init__(self, h=6, num_layers=2):
        super().__init__()

        self.shared_net = DualNet(h, num_layers)
        self.log_scale = nn.Parameter(torch.tensor(-0.5))
        self.nu_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, feats):
        lam_pos, lam_neg = self.shared_net(feats)
        return lam_pos, lam_neg, self.nu_raw


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


def solve_dual_nn(A, b, c, labels, k, y_centers=None, upper=False, steps=3000, lr=1e-5, name="Run1", hidden=5, layers=2):
    if wandb.run is not None:
        wandb.finish()

    if upper == False:
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

    sign = 1 if upper else -1
    c = sign * c

    if "IV_cont" in name:
        feats = torch.tensor(
            [[z/(k-1), 2*t-1, y/(k-1)] for z, t, y in labels[:-1]],
            dtype=torch.float32,
            device=device
        )
    elif "Edu_vs_Voting" in name:
      obs_tuples = [(x, d, y) for x in range(kx) for d in [0,1] for y in range(ky)]
      feats = torch.tensor([[x / (kx - 1), # normalize X
                              2 * d - 1,   # map {0,1} → {-1,1}
                              y_centers[y] # observed Y
                            ]
                    for (x, d, y) in obs_tuples],
                    dtype=torch.float32,
                    device=device)

    A = torch.tensor(A, dtype=torch.float32, device=device)
    A_obs = A[:-1]
    b = torch.tensor(b, dtype=torch.float32, device=device)
    b_obs = b[:-1]
    c = torch.tensor(c, dtype=torch.float32, device=device)

    model = DualModel(h=hidden, num_layers=layers).to(device)

    print("\nNeural network parameters:", count_params(model))
    print("Dual variables:", len(b))
    print("Compression ratio: {:.2f}x".format(len(b)/count_params(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    model.nu_raw.data.fill_(0.0)

    mu = 1

    for step in range(steps):
        optimizer.zero_grad()

        lam_pos, lam_neg, nu_raw = model(feats)
        lam = lam_pos - lam_neg

        nu_max = torch.min(c - (A_obs.t() @ lam))
        nu = nu_max - torch.nn.functional.softplus(nu_raw)

        slack = c - (A_obs.t() @ lam + nu)
        # if slack.min().item() < EPS_TOL**2:
        #     print(slack.min().item())
        #     break

        violation = torch.relu(-(c - (A_obs.t() @ lam + nu)))
        penalty = 1 * violation.mean()

        dual_obj = (b_obs+EPS_TOL)@lam_pos - (b_obs-EPS_TOL)@lam_neg + nu

        loss = -dual_obj + mu * penalty
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        violation_mask = slack < 0

        if violation_mask.any():
            print("\n--- Exact QP Projection ---")

            # BEFORE
            min_slack_before = slack.min().item()
            print(f"Min slack BEFORE: {min_slack_before:.6e}")

            lam_np = lam.detach().cpu().numpy()
            A_np = A_obs.detach().cpu().numpy()
            c_np = c.detach().cpu().numpy()
            nu_val = nu.item()

            # Solve projection
            lam_proj_np = project_lambda_qp(lam_np, A_np, c_np, nu_val)

            # back to torch
            lam_proj = torch.tensor(lam_proj_np, dtype=torch.float32, device=device)

            # update lam_pos / lam_neg
            lam_pos.copy_(torch.clamp(lam_proj, min=0))
            lam_neg.copy_(torch.clamp(-lam_proj, min=0))

            # AFTER → recompute slack
            lam_new = lam_pos - lam_neg

            nu_max = torch.min(c - (A_obs.t() @ lam_new))
            nu = nu_max - torch.nn.functional.softplus(nu_raw)

            slack_after = c - (A_obs.t() @ lam_new + nu)

            print(f"Min slack AFTER:  {slack_after.min().item():.6e}")
            print(f"#violations AFTER: {(slack_after < -1e-8).sum().item()}")

            if (slack_after < -1e-8).any():
                print("⚠️ Still slight violation (numerical tolerance)")
            else:
                print("✅ Fully feasible after projection")

            print("--- End QP Projection ---\n")

        mu = min(mu * 1.002, 100)

        if step % 1000 == 0:
            with torch.no_grad():
                wandb.log({
                    "step": step,
                    "dual_obj": (sign * dual_obj).item(),
                    "min_slack": slack.min().item(),
                    "max_slack": slack.max().item(),
                    "mean_slack": slack.mean().item(),
                    "num_active": (slack < 1e-2).sum().item(),
                    "loss": loss.item(),
                })

            smallest5 = torch.topk(slack, 5, largest=False).values

            print(
                f"step {step} | dual {sign*dual_obj.item():.4f} | "
                f"min slack {slack.min().item():.6f} | "
                f"max slack {slack.max().item():.6f} | "
                f"top5 smallest {smallest5.detach().cpu().numpy()}"
            )

    dual_value = dual_obj.item()

    if upper:
        dual_value = -dual_value

    return lam_pos.detach().cpu().numpy(), lam_neg.detach().cpu().numpy(), nu.item()


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

    lamL_pos, lamL_neg, nuL = solve_dual_nn(
        A, b, c, labels, k,
        y_centers=y_centers,
        upper=False,
        lr=args.lr_lower,
        steps=n,
        name=wandb_name,
        hidden=args.hidden,
        layers=args.layers
    )

    lamU_pos, lamU_neg, nuU = solve_dual_nn(
        A, b, c, labels, k,
        y_centers=y_centers,
        upper=True,
        lr=args.lr_upper,
        steps=n,
        name=wandb_name,
        hidden=args.hidden,
        layers=args.layers
    )

    b_obs = b[:-1]
    lower = -((b_obs+EPS_TOL)@lamL_pos - (b_obs-EPS_TOL)@lamL_neg + nuL)
    upper = ((b_obs+EPS_TOL)@lamU_pos - (b_obs-EPS_TOL)@lamU_neg + nuU)

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

        plot_dual_heatmap(lamL_pos-lamL_neg, labels[:-1], k, "Lower Bound Dual")
        plot_dual_heatmap(lamU_pos-lamU_neg, labels[:-1], k, "Upper Bound Dual")
    
    print("Time taken: ", end-start)