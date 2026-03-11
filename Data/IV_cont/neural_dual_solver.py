import numpy as np
from itertools import product
from scipy.stats import truncnorm
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from LP_construction import * 
from utils import *

SEED = 2026

np.random.seed(SEED)
torch.manual_seed(SEED)


EPS_TOL = 1e-6

class DualNet(nn.Module):
    def __init__(self,h=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,h),
            nn.Tanh(),
            nn.Linear(h,2),
        )
        self.apply(init_weights)

    def forward(self,x):
        out = self.net(x)        # (N,2)

        lam_pos = torch.nn.functional.softplus(out[:,0])
        lam_neg = torch.nn.functional.softplus(out[:,1])

        return lam_pos, lam_neg

class DualNet2(nn.Module):

    def __init__(self,h=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3,h),
            nn.Tanh(),
        )

        self.gate = nn.Linear(h,1)
        self.mag  = nn.Linear(h,1)

        self.apply(init_weights)

    def forward(self,x):

        h = self.net(x)

        gate = torch.sigmoid(self.gate(h))     # which constraints active
        mag  = torch.nn.functional.softplus(self.mag(h))  # magnitude

        lam = gate * mag

        lam_pos = torch.clamp(lam,min=0)
        lam_neg = torch.clamp(-lam,min=0)

        return lam_pos.squeeze(), lam_neg.squeeze()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        nn.init.normal_(m.bias, mean=0.0, std=0.05)

class DualModel(nn.Module):
    def __init__(self,h=6):
        super().__init__()

        self.shared_net = DualNet2(h)
        self.log_scale = nn.Parameter(torch.tensor(-0.5))

        self.nu_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self,feats):
        lam_pos, lam_neg = self.shared_net(feats)

        return lam_pos, lam_neg, self.nu_raw

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 6. Neural dual solver
# ============================================================

def solve_dual_nn(A,b,c,labels,k,upper=False,steps=3000,lr=1e-5):

    device="cuda" if torch.cuda.is_available() else "cpu"

    sign=-1 if upper else 1
    c=sign*c

    A=torch.tensor(A,dtype=torch.float32,device=device)
    A_obs = A[:-1]
    b=torch.tensor(b,dtype=torch.float32,device=device)
    b_obs = b[:-1]
    c=torch.tensor(c,dtype=torch.float32,device=device)

    feats=torch.tensor(
        [[z/(k-1),2*t-1,y/(k-1)] for z,t,y in labels[:-1]],
        dtype=torch.float32,
        device=device
    )

    model=DualModel(h=10).to(device)
    # if upper:
    #     model= copy.deepcopy(upper_pretrain_model)
    # else:
    #     model = copy.deepcopy(pretrain_model)
    print("\nNeural network parameters:", count_params(model))
    print("Dual variables:", len(b))
    print("Compression ratio: {:.2f}x".format(len(b)/count_params(model)))

    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    model.nu_raw.data.fill_(0.0)

    mu=0.1


    for step in range(steps):
      optimizer.zero_grad()

      lam_pos,lam_neg,nu_raw=model(feats)
      lam=lam_pos-lam_neg
      lam = torch.where(torch.abs(lam) < 1e-4, torch.zeros_like(lam), lam)

      nu_max = torch.min(c - (A_obs.t() @ lam))
      nu = nu_max - torch.nn.functional.softplus(nu_raw)

      slack = c-(A_obs.t()@lam+nu)
      if slack.min().item() < EPS_TOL*1e-2:
        break

      barrier=-mu*torch.mean(torch.log(slack))
      slack_reg = 0.01 * torch.mean(slack**2)
      lam_abs = lam_pos + lam_neg
      sparse_reg = 0.01 * torch.mean(lam_abs)
      dual_slack = A_obs @ slack

      cs_reg = 0.01 * torch.mean(torch.abs(lam * dual_slack))
      # barrier=-mu*torch.sum(1/slack)

      dual_obj=(b_obs+EPS_TOL)@lam_pos-(b_obs-EPS_TOL)@lam_neg+nu
      # dual_obj = b_obs @ (lam_pos - lam_neg) + nu

      loss= -dual_obj  + sparse_reg #+ barrier + slack_reg + cs_reg
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(),1.5)
      optimizer.step()

      mu *= 0.998

      if step%1000==0:
          smallest5 = torch.topk(slack, 5, largest=False).values

          print(
              f"step {step} | dual {sign*dual_obj.item():.4f} | "
              f"min slack {slack.min().item():.6f} | "
              f"max slack {slack.max().item():.6f} | "
              f"top5 smallest {smallest5.detach().cpu().numpy()}"
          )

    tol = 1e-2
    num_zero = torch.sum(slack < tol).item()

    print(f'no. of near-zero slack constraints: {num_zero}/{slack.numel()}')
    dual_value=dual_obj.item()

    if upper:
        dual_value=-dual_value

    return lam_pos.detach().cpu().numpy(), lam_neg.detach().cpu().numpy(), nu.item()


if __name__=="__main__":
    n=40000
    n_pts = 10000
    k=8

    print("Generating data...")
    data = generate_data(n_pts,0.5)

    print("Estimating distribution...")
    P = empirical_distribution(data,k)

    print("Building LP system...")
    A,b,c,labels=build_constraints(P,k)


    print("\n===== SIZE CHECK =====")

    print("A shape:",A.shape)
    print("b shape:",b.shape)
    print("c shape:",c.shape)
    print("Dual variables:",A.shape[0])
    print("Latent types:",A.shape[1])


    print("\nTraining neural dual...")

    lamL_pos, lamL_neg, nuL=solve_dual_nn(A,b,c,labels,k,upper=False,lr=5e-4,steps=n)
    print("nu ", nuL)
    lamU_pos, lamU_neg,nuU=solve_dual_nn(A,b,c,labels,k,upper=True,lr=5e-4,steps=n)
    print("nu ", nuU)

    b_obs = b[:-1]
    lower=(b_obs+EPS_TOL)@lamL_pos-(b_obs-EPS_TOL)@lamL_neg+nuL
    upper=-((b_obs+EPS_TOL)@lamU_pos-(b_obs-EPS_TOL)@lamU_neg+nuU)

    # lower,upper=min(lower,upper),max(lower,upper)


    print("\n===== BOUNDS =====")

    print(f"NN lower bound : {lower:.4f}")
    print(f"NN upper bound : {upper:.4f}")
    print("True ATE = 3")


    plot_dual_heatmap(lamL_pos-lamL_neg,labels[:-1],k,"Lower Bound Dual")
    plot_dual_heatmap(lamU_pos-lamU_neg,labels[:-1],k,"Upper Bound Dual")