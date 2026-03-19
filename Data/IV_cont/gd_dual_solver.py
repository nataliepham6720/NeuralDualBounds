import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import truncnorm
import torch

from LP_construction import * 
from utils import *

SEED = 2026

np.random.seed(SEED)
torch.manual_seed(SEED)

def solve_dual_band(A, b, c, eps=1e-8, upper=False,
                    steps=6000, lr=5e-4,
                    mu_init=1.0, mu_decay=0.95):

    if upper:
        c = -c

    m, n = A.shape
    At = A.T

    lam_pos = np.zeros(m)
    lam_neg = np.zeros(m)
    nu = np.min(c) - 1.0   # strictly feasible start

    mu = mu_init

    for step in range(steps):

        lam = lam_pos - lam_neg
        slack = c - (At @ lam + nu)

        # keep slack positive (barrier domain)
        # slack = np.maximum(slack, 1e-8)
        if slack.min() < 1e-8:
          mu = mu_init * (mu_decay**steps)
        else:
          mu = 0
          # break
        inv_slack = 1.0 / slack

        # correct gradients
        # grad_pos = (b + eps) + mu * (A @ inv_slack)
        # grad_neg = -(b - eps) - mu * (A @ inv_slack)
        # grad_nu  = 1 + mu * np.sum(inv_slack)
        grad_pos = (b + eps) - mu * (A @ inv_slack)
        grad_neg = -(b - eps) + mu * (A @ inv_slack)
        grad_nu  = 1 - mu * np.sum(inv_slack)

        # normalize gradient (critical for stability)
        scale = (
            np.linalg.norm(grad_pos)
            + np.linalg.norm(grad_neg)
            + abs(grad_nu)
            + 1e-12
        )
        # scale = norm(grad_pos)+norm(grad_neg)+abs(grad_nu)
        lam_pos += lr * grad_pos / scale
        lam_neg += lr * grad_neg / scale
        nu      += lr * grad_nu  / scale

        lam_pos = np.maximum(lam_pos, 0)
        lam_neg = np.maximum(lam_neg, 0)

       

        if step % 1000 == 0:
            dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
            print(f"step {step} | dual {dual_val:.6f} | min slack {slack.min():.3e} | max slack {slack.max():.3e}")

    dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu

    if upper:
        dual_val = -dual_val

    lam_final = lam_pos - lam_neg
    print('no. of zero slack constraints:', np.sum(slack < 1e-4))
    print('nu', nu)

    return dual_val, lam_final, nu

import torch

def solve_dual_torch(A, b, c, eps=1e-8, upper=False,
                     steps=10000,
                     lr=1e-4,
                     mu=1, mu_decay=0.995,
                     device="cpu"):

    # use double precision for stability near the boundary
    A = torch.tensor(A, dtype=torch.float64, device=device)
    b = torch.tensor(b, dtype=torch.float64, device=device)
    c = torch.tensor(c, dtype=torch.float64, device=device)

    sign = -1 if upper else 1
    c = sign * c

    m, n = A.shape
    print(m,n)

    lam_pos = torch.zeros(m, device=device, dtype=torch.float64, requires_grad=True)
    lam_neg = torch.zeros(m, device=device, dtype=torch.float64, requires_grad=True)

    # start strictly feasible: slack = c - A^T lam - nu > 0
    nu = torch.tensor(c.min().item() - 1.0,
                      device=device, dtype=torch.float64,
                      requires_grad=True)

    optimizer = torch.optim.SGD([lam_pos, lam_neg, nu], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        lam = lam_pos - lam_neg
        slack = c - A.T @ lam - nu  # must stay > 0

        # dual objective
        dual_obj = (
            (b + eps) @ lam_pos
            - (b - eps) @ lam_neg
            + nu
        )

        # log-barrier: maximize dual_obj - mu * sum(-log(slack))
        # => loss = -dual_obj + mu * barrier, with barrier = -sum(log(slack))
        # if (slack <= 0).any():
        #     # should not happen if fraction-to-boundary is working,
        #     # but keep a big penalty as a safety net
        #     infeas = -slack[slack <= 0]
        #     barrier = 1e4 * torch.mean(infeas**2)
        # else:
        #     barrier = -torch.mean(torch.log(slack))

        slack_safe = torch.clamp(slack, min=1e-12)
        barrier = -torch.mean(torch.log(slack_safe))
        
        loss = -dual_obj - mu * barrier
        loss.backward()

        # save old values
        lam_pos_old = lam_pos.data.clone()
        lam_neg_old = lam_neg.data.clone()
        nu_old      = nu.data.clone()

        # let Adam propose a step
        optimizer.step()

        lam_pos_new = lam_pos.data.clone()
        lam_neg_new = lam_neg.data.clone()
        nu_new      = nu.data.clone()

        # fraction-to-boundary step
        lam_pos.data.copy_(lam_pos_old)
        lam_neg.data.copy_(lam_neg_old)
        nu.data.copy_(nu_old)

        d_lam_pos = lam_pos_new - lam_pos_old
        d_lam_neg = lam_neg_new - lam_neg_old
        d_nu      = nu_new - nu_old

        lam_old = lam_pos_old - lam_neg_old
        d_lam   = d_lam_pos - d_lam_neg

        slack_old = c - A.T @ lam_old - nu_old
        d_slack   = - (A.T @ d_lam + d_nu)

        # compute maximum alpha to keep slack > 0
        tau = 0.9  # fraction-to-boundary
        alpha_max = 1.0
        mask = d_slack < 0
        if mask.any():
            alpha_max = torch.min(-tau * slack_old[mask] / d_slack[mask]).item()
        alpha = max(0.0, min(1.0, alpha_max))

        lam_pos.data.copy_(lam_pos_old + alpha * d_lam_pos)
        lam_neg.data.copy_(lam_neg_old + alpha * d_lam_neg)
        nu.data.copy_(nu_old + alpha * d_nu)

        # enforce lam_pos, lam_neg >= 0
        with torch.no_grad():
            lam_pos.clamp_(min=0.0)
            lam_neg.clamp_(min=0.0)

        if step % 2000 == 0:
            lam = lam_pos - lam_neg
            slack = c - A.T @ lam - nu
            print(f"step {step} | dual {dual_obj.item():.6f} | "
                  f"min slack {slack.min().item():.3e} | "
                  f"max slack {slack.max().item():.3e}")
            mu *= mu_decay

    lam_final = (lam_pos - lam_neg).detach().cpu().numpy()

    dual_val = (
        (b + eps) @ lam_pos.detach()
        - (b - eps) @ lam_neg.detach()
        + nu.detach()
    ).item()

    if upper:
        dual_val = -dual_val

    lam = lam_pos - lam_neg
    slack = c - A.T @ lam - nu
    print("active constraints:", torch.sum(slack < 1e-4).item())

    return dual_val, lam_final, nu.item()

if __name__=="__main__":
    n = 200000
    k = 8

    data = generate_data(n, lam=0.5) #np.random.rand())
    P = empirical_distribution(data,k)

    A,b,c,labels = build_constraints(P,k)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Dual variables:", len(b))
    print("Latent types:", len(c))

    # lower, lamL, nuL = solve_dual_band(A[:-1],b[:-1],c,eps=1e-8,upper=False,lr=5e-4,steps=n)
    # upper, lamU, nuU = solve_dual_band(A[:-1],b[:-1],c,eps=1e-8,upper=True,lr=5e-4,steps=n)

    lower, lamL, nuL = solve_dual_torch(A[:-1],b[:-1],c,eps=1e-8,
                                        upper=False,
                                        lr=5e-6,
                                        mu=0.1,
                                        steps=n,
                                        device=device)
    upper, lamU, nuU = solve_dual_torch(A[:-1],b[:-1],c,eps=1e-8,
                                        upper=True,
                                        lr=5e-6,
                                        mu=0.1,
                                        steps=n,
                                        device=device)


    # lower, lamL, nuL = solve_dual_ipm(A, b, c, eps=1e-8, max_iter=80, tol=1e-8)

    print("\n==============================")
    print("ATE lower bound:", lower)
    print("ATE upper bound:", upper)
    print("True ATE = 3")
    print("==============================")

    plot_dual_heatmap(lamL, labels[:-1], k, "Lower Bound Dual")
    plot_dual_heatmap(lamU, labels[:-1], k, "Upper Bound Dual")