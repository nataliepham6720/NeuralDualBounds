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

# SLACK_TOL = 1e-4
# NU_TOL = 1e-9


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="experiment") # IV_cont or Edu_vs_Voting
    parser.add_argument("--distribution_gen", type=str, default="generate")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--warm_start", type=int, default=2000)
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
    def __init__(self,i=3, h=10, num_layers=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(i, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(h, 1))

        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        out = self.net(x)
        lam_pos = out[:, 0]
        # lam_neg = out[:, 1]
        return lam_pos #, lam_neg


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
    def __init__(self, i=3, h=6, num_layers=2):
        super().__init__()

        self.shared_net = DualNet(i, h, num_layers)
        # self.log_scale = nn.Parameter(torch.tensor(-0.5))
        # self.nu_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, feats):
        lam_pos = self.shared_net(feats)
        return lam_pos #, lam_neg # , self.nu_raw

class DualModel2(nn.Module):
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

def solve_dual_nn(A, b, c, labels, k, y_centers=None, upper=False, 
                  steps=3000, lr=1e-5, name="Run1", hidden=5, layers=2,
                  interior_margin=1e-3, max_backtrack=100,
                  warm_start_steps=2000):
    if wandb.run is not None:
        wandb.finish()

    if not upper:
        name = name + "_LowerBound"
    else:
        name = name + "_UpperBound"

    wandb.init(
        project="NeuralDualSolver_Edu+Voting",
        name=name,
        config={"steps": steps, "lr": lr, "k": k},
        reinit=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sign = -1 if upper else 1
    c = sign * c

    # ----- Build feature grid -----
    if "IV_cont" in name:
        grid = [[z/(k-1), 2*t-1, y/(k-1)] for z, t, y in labels[:-1]]
        grid.append(labels[-1])  # normalization constraint in primal
        feats = torch.tensor(grid, dtype=torch.float32, device=device)
    elif "Edu_vs_Voting" in name:
        kx = k
        ky = k
        # observational dual constraints only
        obs_tuples = [(x, d, y) for x in range(kx)
                                for d in [0, 1]
                                for y in range(ky)]
        grid = [[x / (kx - 1), 2 * d - 1, y_centers[y]] for (x, d, y) in obs_tuples]
        # normalization constraint
        grid.append([-1, -1, -1])
        feats = torch.tensor(grid, dtype=torch.float32, device=device)
        print("feats shape:", feats.shape)
    else:
        raise ValueError("Need a feature grid for neural lam approximation")

    A = torch.tensor(A, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)
    c = torch.tensor(c, dtype=torch.float32, device=device)

    print("\nDual variables:", len(b))
    input_size = feats.shape[1]

    # ---------- Build neural model for lam ----------
    model = DualModel(i=input_size, h=hidden, num_layers=layers).to(device)
    print("Neural network parameters:", count_params(model))
    print("Compression ratio: {:.2f}x".format(len(b) / count_params(model)))

    model.train()
    optimizer_phase1 = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(warm_start_steps):
        lam  = model(feats)
        slack = c - (A.t() @ lam)
        violation = torch.relu(interior_margin - slack)
        mse = (interior_margin - slack) ** 2
        loss = violation.max() #+ mse.mean()
        optimizer_phase1.zero_grad()
        loss.backward()
        optimizer_phase1.step()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # mu = 1.0
    
    max_mu = 1e-3
    t = 0
    mu = 1 / (t + 1)

    for step in range(steps):
        # Save previous state for backtracking
        prev_state = {k: v.clone() for k, v in model.state_dict().items()}
        lr_prev = [g["lr"] for g in optimizer.param_groups]

        # Forward: lam from NN
        lam_pos = model(feats)
        lam = lam_pos # - lam_neg

        slack = c - (A.t() @ lam)
        if (slack <= 0).any():
            print(f"[Step {step}] Infeasible before step, min_slack={slack.min().item():.6e}")

        if (slack <= 0).any():
            print("violation")
            success = False
            for _ in range(max_backtrack):
                # Restore model params
                model.load_state_dict(prev_state)

                # Halve learning rate
                for g in optimizer.param_groups:
                    g["lr"] *= 0.8

                # Recompute from restored params
                lam_pos_bt = model(feats)
                lam_bt = lam_pos_bt #- lam_neg_bt
                slack_bt = c - (A.t() @ lam_bt) + interior_margin
                # slack_bt = torch.clamp(slack_bt, min=interior_margin)

                barrier_bt = -torch.log(slack_bt).mean()
                dual_obj_bt = b @ lam_bt
                loss_bt = -dual_obj_bt + mu * barrier_bt

                optimizer.zero_grad()
                loss_bt.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                lam_pos_bt = model(feats)
                lam_bt = lam_pos_bt  #- lam_neg_bt
                slack_bt = c - (A.t() @ lam_bt)

                if (slack_bt > 0).all():
                    success = True
                    slack = slack_bt
                    dual_obj = dual_obj_bt
                    loss = loss_bt
                    break

            if not success:
                print(f"[Step {step}] Could not restore feasibility; stopping.")
                smallest5 = torch.topk(slack_bt, 5, largest=False).values
                print(
                    f"step {step} | dual {sign * dual_obj_bt.item():.4f} | "
                    f"min slack {slack_bt.min().item():.6f} | "
                    f"max slack {slack_bt.max().item():.6f} | "
                    f"top5 smallest {smallest5.detach().cpu().numpy()} | "
                    # f"mu {mu:.3e}"
                )
                model.load_state_dict(prev_state)
                for g, old_lr in zip(optimizer.param_groups, lr_prev):
                    g["lr"] = old_lr
                break

        # Central path parameter update
        # mu = 10.0 / (step + 1)
        # mu = max(mu, max_mu)
        # if step % 100 == 0 and step > 10000:
        #     max_mu = max_mu * 0.99
        barrier = -torch.log(slack).min()
        dual_obj = b @ lam
        loss = -dual_obj + mu * barrier

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # t = 0 #step / 100000
        interval = max(2500, int(100000 / (t + 1))) if mu > max_mu*10 else 2000

        if step % interval == 0:
            mu = 1 / (t + 1)
            t = t+1
            mu = max(mu, max_mu)

        if step % 1000 == 0:
        #     if step % 100000 == 0:
        #         t = step / 100000
        #         mu = 1/(t+1)
        #         mu = max(mu, max_mu)

            with torch.no_grad():
                wandb.log({
                    "step": step,
                    "dual_obj": (sign * dual_obj).item(),
                    "min_slack": slack.min().item(),
                    "max_slack": slack.max().item(),
                    "mean_slack": slack.mean().item(),
                    "num_active": (slack < 1e-4).sum().item(),
                    "loss": loss.item(),
                    "mu": mu,
                })

                smallest5 = torch.topk(slack, 5, largest=False).values
                if step % 2000 == 0:
                    print(
                        f"step {step} | dual {sign * dual_obj.item():.4f} | "
                        f"min slack {slack.min().item():.6f} | "
                        f"max slack {slack.max().item():.6f} | "
                        f"top5 smallest {smallest5.detach().cpu().numpy()} | "
                        f"mu {mu:.3e}"
                    )

    # Final dual value
    dual_value = dual_obj.item()
    if upper:
        dual_value = -dual_value

    with torch.no_grad():
        lam_pos = model(feats)
        lam = lam_pos #- lam_neg
        slack = c - (A.t() @ lam)
        nu = slack.min().item()

    lam_pos_np = lam_pos.detach().cpu().numpy()
    

    return lam_pos_np 

# Main

if __name__ == "__main__":
    args = get_args()
    
    n = args.steps
    n_pts = args.n_pts
    k = args.k
    name = args.name
    dist = args.distribution_gen
    # warm_start = args.warm_start

    if name == "IV_cont":
        wandb_name = name + f"_k{k}_steps{n}_hidden{args.hidden}_layers{args.layers}"
        
        if dist == "generate":
            print("Generating data...")
            data = generate_data_IV(n_pts, lam=0.5)

            print("Estimating distribution...")
            P = empirical_distribution_IV(data, k)
            print(P.shape)
        elif dist == "pre-load":
            print("Loading saved distribution...")
            if k == 6:
               P = np.load("./Data/IV_cont/P6.npy")
            elif k == 8:
                P = np.load("./Data/IV_cont/P8.npy")
            elif k == 10:
                P = np.load("./Data/IV_cont/P10.npy")
            else:
                print("No prior data found!!!")

        print("Building LP system...")
        A, b, c, labels = build_constraints_IV(P, k)
        y_centers = None

    elif name == "Edu_vs_Voting":
        wandb_name = name + f"_k{k}_steps{n}_hidden{args.hidden}_layers{args.layers}"

        kx = ky = args.k

        if dist == "generate":
            print("Generating data...")
            data, Y0, Y1 = generate_data_EV(n_pts, tau=0.5, seed=SEED)
            ATE_true = np.mean(Y1 - Y0)
            print("True ATE:", ATE_true)
            print("Estimating distribution...")
            P, x_bins, y_bins = empirical_distribution_EV(data, kx, ky)
            print(P.shape)
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2 
        elif dist == "pre-load":
            print("Loading saved distribution...")
            if k == 12:
                data = np.load("./Data/Edu_vs_Voting/P12_EV.npz")
                # P = np.load("./Data/Edu_vs_Voting/P12_EV.npy")
            elif k == 16:
                data = np.load("./Data/Edu_vs_Voting/P16_EV.npz")
                # P = np.load("./Data/Edu_vs_Voting/P16_EV.npy")
            else:
                print("No prior data found!!!")

            # Y, y_bins = discretize(P[:, 0], ky)
            P = data["P"]
            print(P.shape)
            x_bins = data["x_bins"]
            y_bins = data["y_bins"]
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        
        print("Building LP system...")
        A, b, c, labels = build_constraints_EV(P, kx, ky, y_bins)

    print("\n===== SIZE CHECK =====")
    print("A shape:", A.shape)
    print("b shape:", b.shape)
    print("c shape:", c.shape)

    print("\nTraining neural dual...")
    start = time.time()

    lamL_pos = solve_dual_nn(A, b, c, labels, k, y_centers=y_centers, upper=False, 
                  steps=n, lr=1e-5, name=wandb_name, hidden=args.hidden,
                  layers=args.layers, interior_margin=1e-3, max_backtrack=50,
                  warm_start_steps=args.warm_start)
    lamU_pos = solve_dual_nn(A, b, c, labels, k, y_centers=y_centers, upper=True, 
                  steps=n, lr=1e-5, name=wandb_name, hidden=args.hidden,
                  layers=args.layers, interior_margin=1e-3, max_backtrack=50,
                  warm_start_steps=args.warm_start)

    b_obs = b #[:-1]
    lower = ((b_obs+EPS_TOL)@lamL_pos)
    upper = -(b_obs+EPS_TOL)@lamU_pos
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

        plot_dual_heatmap(lamL_pos, labels[:-1], k, "Lower Bound Dual - {lower:.4f}")
        plot_dual_heatmap(lamU_pos, labels[:-1], k, "Upper Bound Dual - {upper:.4f}")
    
    print("Time taken: ", end-start)