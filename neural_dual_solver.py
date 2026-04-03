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

from Data.IV_cont.LP_construction import * 
from Data.IV_cont.utils import *

SEED = 2020

np.random.seed(SEED)
torch.manual_seed(SEED)

EPS_TOL = 1e-6
K_active = 10


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="experiment") # IV_cont or Edu_vs_Voting
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--lr_lower", type=float, default=5e-3)
    parser.add_argument("--lr_upper", type=float, default=5e-4)

    parser.add_argument("--n_pts", type=int, default=10000)

    return parser.parse_args()


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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def solve_dual_nn(A, b, c, labels, k, upper=False, steps=3000, lr=1e-5, name="Run1", hidden=5, layers=2):
    if wandb.run is not None:
        wandb.finish()

    if upper == False:
        name = name + "_LowerBound"
    else:
        name = name + "_UpperBound"

    wandb.init(
        project="neural-dual-lp",
        name=name,
        config={"steps": steps, "lr": lr, "k": k},
        reinit=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sign = 1 if upper else -1
    c = sign * c

    A = torch.tensor(A, dtype=torch.float32, device=device)
    A_obs = A[:-1]
    b = torch.tensor(b, dtype=torch.float32, device=device)
    b_obs = b[:-1]
    c = torch.tensor(c, dtype=torch.float32, device=device)

    feats = torch.tensor(
        [[z/(k-1), 2*t-1, y/(k-1)] for z, t, y in labels[:-1]],
        dtype=torch.float32,
        device=device
    )

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
        if slack.min().item() < EPS_TOL**2:
            print(slack.min().item())
            break

        violation = torch.relu(-(c - (A_obs.t() @ lam + nu)))
        penalty = 1 * violation.mean()

        dual_obj = (b_obs+EPS_TOL)@lam_pos - (b_obs-EPS_TOL)@lam_neg + nu

        loss = -dual_obj + mu * penalty
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

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

    if name == "IV_cont":
        name = name + f"_k{k}_steps{n}_lrL{args.lr_lower}_lrU{args.lr_upper}_hidden{args.hidden}_layers{args.layers}"
        print("Generating data...")
        data = generate_data_IV(n_pts, lam=0.5)

        print("Estimating distribution...")
        P = empirical_distribution_IV(data, k)

        print("Building LP system...")
        A, b, c, labels = build_constraints_IV(P, k)

    print("\n===== SIZE CHECK =====")
    print("A shape:", A.shape)
    print("b shape:", b.shape)
    print("c shape:", c.shape)

    print("\nTraining neural dual...")
    start = time.time()

    lamL_pos, lamL_neg, nuL = solve_dual_nn(
        A, b, c, labels, k,
        upper=False,
        lr=args.lr_lower,
        steps=n,
        name=name,
        hidden=args.hidden,
        layers=args.layers
    )

    lamU_pos, lamU_neg, nuU = solve_dual_nn(
        A, b, c, labels, k,
        upper=True,
        lr=args.lr_upper,
        steps=n,
        name=name,
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
        # print("True ATE = 0.25")
    elif name == "IV_cont":
        print(f"NN lower bound : {lower:.4f}")
        print(f"NN upper bound : {upper:.4f}")
        print("True ATE = 3")

        plot_dual_heatmap(lamL_pos-lamL_neg, labels[:-1], k, "Lower Bound Dual")
        plot_dual_heatmap(lamU_pos-lamU_neg, labels[:-1], k, "Upper Bound Dual")
    
    print("Time taken: ", end-start)