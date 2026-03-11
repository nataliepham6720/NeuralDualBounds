import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import truncnorm

from LP_construction import * 
from utils import *

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
          break
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

        mu *= mu_decay

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
# def solve_dual_band(A, b, c, eps=1e-8, upper=False,
#                     steps=6000, lr=5e-4,
#                     mu_init=1.0, mu_decay=0.95):

#     if upper:
#         c = -c

#     m, n = A.shape
#     At = A.T

#     lam_pos = np.zeros(m)
#     lam_neg = np.zeros(m)
#     nu = np.min(c) - 1.0   # strictly feasible start

#     mu = mu_init

#     for step in range(steps):

#         lam = lam_pos - lam_neg
#         # slack = c - (At @ lam + nu)

#         # # keep slack positive (barrier domain)
#         # slack = np.maximum(slack, 1e-8)
#         # # if np.min(slack) < 
#         # inv_slack = 1.0 / slack

#         # # correct gradients
#         # grad_pos = (b + eps) + mu * (A @ inv_slack)
#         # grad_neg = -(b - eps) - mu * (A @ inv_slack)
#         # grad_nu  = 1 + mu * np.sum(inv_slack)

#         # # normalize gradient (critical for stability)
#         # scale = (
#         #     np.linalg.norm(grad_pos)
#         #     + np.linalg.norm(grad_neg)
#         #     + abs(grad_nu)
#         #     + 1e-12
#         # )

#         # lam_pos += lr * grad_pos / scale
#         # lam_neg += lr * grad_neg / scale
#         # nu      += lr * grad_nu  / scale

#         # lam_pos = np.maximum(lam_pos, 0)
#         # lam_neg = np.maximum(lam_neg, 0)

#         # mu *= mu_decay
#         slack = c - (At @ lam + nu)
#         if np.min(slack) < 1e-12:
#           break
#         inv_slack = 1.0 / slack

#         grad_pos = (b + eps) + mu * (A @ inv_slack)
#         grad_neg = -(b - eps) - mu * (A @ inv_slack)
#         grad_nu  = 1 - mu * np.sum(inv_slack)

#         scale = np.linalg.norm(
#             np.concatenate([grad_pos, grad_neg, [grad_nu]])
#         ) + 1e-12

#         d_lam_pos = grad_pos / scale
#         d_lam_neg = grad_neg / scale
#         d_nu      = grad_nu  / scale

#         d_lam = d_lam_pos - d_lam_neg
#         Ad = At @ d_lam + d_nu

#         mask = Ad > 0
#         if np.any(mask):
#             alpha_max = np.min(slack[mask] / Ad[mask])
#             alpha = min(lr, 0.99 * alpha_max)
#         else:
#             alpha = lr

#         lam_pos += alpha * d_lam_pos
#         lam_neg += alpha * d_lam_neg
#         nu      += alpha * d_nu

#         if step % 1000 == 0:
#             dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
#             print(f"step {step} | dual {dual_val:.6f} | min slack {slack.min():.3e} | max slack {slack.max():.3e}")

#     dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu

#     if upper:
#         dual_val = -dual_val

#     lam_final = lam_pos - lam_neg
#     print('no. of zero slack constraints:', np.sum(slack < 1e-4))

#     return dual_val, lam_final, nu

import numpy as np

def solve_dual_band3(A, b, c, eps=1e-8, upper=False,
                    steps=6000, lr=5e-4,
                    mu_init=1.0, mu_decay=0.995):
    if upper:
        c = -c

    m, n = A.shape
    At = A.T

    lam_pos = np.zeros(m)
    lam_neg = np.zeros(m)

    # dual variable for normalization constraint
    nu = np.min(c) - 1.0

    mu = mu_init

    for step in range(steps):

        lam = lam_pos - lam_neg

        slack = (c - At @ lam) - nu

        if np.any(slack < 1e-12):
            raise RuntimeError("Barrier violated")

        safe_slack = np.maximum(slack, 1e-8) # to prevent gradient explode
        inv_slack = 1.0 / safe_slack

        grad_pos = (b + eps) + mu * (A @ inv_slack)
        grad_neg = -(b - eps) - mu * (A @ inv_slack)
        grad_nu  = 1 - mu * np.sum(inv_slack)

        scale = np.linalg.norm(
            np.concatenate([grad_pos, grad_neg, [grad_nu]])
        ) + 1e-12

        d_lam_pos = grad_pos / scale
        d_lam_neg = grad_neg / scale
        d_nu = grad_nu / scale

        d_lam = d_lam_pos - d_lam_neg
        Ad = At @ d_lam + d_nu

        mask = Ad > 0

        if np.any(mask):
            alpha_max = np.min(slack[mask] / Ad[mask])
            # alpha = min(lr, 0.99 * alpha_max)
            alpha = min(lr, 0.9 * alpha_max)
        else:
            alpha = lr

        lam_pos += alpha * d_lam_pos
        lam_neg += alpha * d_lam_neg
        nu      += alpha * d_nu

        # if step % 1000 == 0:

        #     dual_val = (
        #         (b + eps) @ lam_pos
        #         - (b - eps) @ lam_neg
        #         + nu
        #     )

            
        if step % 1000 == 0:
            dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
            print(f"step {step} | dual {dual_val:.6f} | min slack {slack.min():.3e} |"
                  f"max slack {slack.max():.3e}")

        mu *= mu_decay

    lam_final = lam_pos - lam_neg

    dual_val = ((b + eps) @ lam_pos - (b - eps) @ lam_neg + nu)
    if upper:
        dual_val = -dual_val

    print("active constraints:", np.sum(slack < 1e-4))

    return dual_val, lam_final, nu

# def solve_dual_band(A, b, c, eps=1e-8, upper=False,
#                     steps=6000, lr=2e-3,
#                     mu_init=1.0, mu_decay=0.995):

#     if upper:
#         c = -c

#     m, n = A.shape
#     At = A.T

#     lam_pos = np.zeros(m)
#     lam_neg = np.zeros(m)
#     nu = np.min(c) - 1.0

#     mu = mu_init

#     for step in range(steps):

#         lam = lam_pos - lam_neg
#         slack = c - (At @ lam + nu)

#         if np.min(slack) <= 0:
#             raise RuntimeError("Lost feasibility")

#         inv_slack = 1.0 / slack

#         grad_common = -mu * (A @ inv_slack)

#         grad_pos = (b + eps) - grad_common
#         grad_neg = -(b - eps) + grad_common
#         grad_nu  = 1 - mu * np.sum(inv_slack)

#         alpha = 1.0

#         while True:

#             lam_pos_new = np.maximum(lam_pos + alpha*lr*grad_pos, 0)
#             lam_neg_new = np.maximum(lam_neg + alpha*lr*grad_neg, 0)
#             nu_new = nu + alpha*lr*grad_nu

#             lam_new = lam_pos_new - lam_neg_new
#             slack_new = c - (At @ lam_new + nu_new)

#             if np.min(slack_new) > 0:
#                 break

#             alpha *= 0.5

#         lam_pos = lam_pos_new
#         lam_neg = lam_neg_new
#         nu = nu_new
#         slack = slack_new

#         if step % 200 == 0:
#             mu *= mu_decay

#         if step % 1000 == 0:
#             dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
#             print(f"step {step} | dual {dual_val:.6f} | min slack {slack.min():.3e}")

#     dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu

#     if upper:
#         dual_val = -dual_val

#     lam_final = lam_pos - lam_neg

#     print("active constraints:", np.sum(slack < 1e-6))
#     print("nu:", nu)

#     return dual_val, lam_final, nu
def solve_dual_band2(A, b, c, eps=1e-8, upper=False,
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
    lr_current = lr

    # safe floor for computing inv_slack (prevents inf). keep not too large.
    inv_slack_floor = 1e-6

    for step in range(steps):

        lam = lam_pos - lam_neg
        true_slack = c - (At @ lam + nu)      # don't modify this; use for checks/alpha
        # diagnostic before any operation
        min_true = np.min(true_slack)
        max_true = np.max(true_slack)

        # If any slack nonpositive, we must NOT compute 1/slack directly.
        if min_true <= 0:
            # revert/repair strategy: reduce step-size and try a conservative "restore" step
            lr_current *= 0.5
            # project lam_pos/lam_neg/nu slightly back into interior (small damping)
            # (this is conservative: it avoids inf and gives the algorithm a chance)
            nu += 1e-3
            lam_pos *= 0.99
            lam_neg *= 0.99
            if step % 100 == 0:
                print(f"step {step} | encountered non-positive slack (min {min_true:.3e}), "
                      f"reducing lr to {lr_current:.3e} and nudging interior.")
            continue

        # compute inv_slack with a safe floor to avoid extreme 1/s values,
        # but use true_slack for step-size limiting and diagnostics.
        safe_slack = np.maximum(true_slack, inv_slack_floor)
        inv_slack = 1.0 / safe_slack

        # corrected gradients (note grad_nu sign)
        grad_pos = (b + eps) + mu * (A @ inv_slack)
        grad_neg = -(b - eps) - mu * (A @ inv_slack)
        grad_nu  = 1.0 - mu * np.sum(inv_slack)

        # normalize gradient vector (keep consistent norm)
        scale = np.linalg.norm(
            np.concatenate([grad_pos, grad_neg, np.array([grad_nu])])
        ) + 1e-12

        d_lam_pos = grad_pos / scale
        d_lam_neg = grad_neg / scale
        d_nu      = grad_nu  / scale

        # search direction and the effect on slack: Ad = A^T d_lam + d_nu
        d_lam = d_lam_pos - d_lam_neg
        Ad = At @ d_lam + d_nu

        # compute maximum alpha that keeps true_slack - alpha * Ad > 0 for all i with Ad>0
        mask = Ad > 1e-14
        if np.any(mask):
            alpha_max = np.min(true_slack[mask] / Ad[mask])
            # if alpha_max is extremely small or non-positive, shrink step instead of taking it.
            if not np.isfinite(alpha_max) or alpha_max <= 0:
                lr_current *= 0.5
                if step % 100 == 0:
                    print(f"step {step} | tiny/nonfinite alpha_max ({alpha_max}), reducing lr to {lr_current:.3e}")
                continue
            alpha = min(lr_current, 0.99 * alpha_max)
        else:
            alpha = lr_current

        # take a tentative update
        lam_pos_new = lam_pos + alpha * d_lam_pos
        lam_neg_new = lam_neg + alpha * d_lam_neg
        nu_new      = nu      + alpha * d_nu

        # project positivity for multipliers
        lam_pos_new = np.maximum(lam_pos_new, 0.0)
        lam_neg_new = np.maximum(lam_neg_new, 0.0)

        # check resulting slack (exact) — if anything nonpositive, reject and reduce lr
        lam_new = lam_pos_new - lam_neg_new
        true_slack_new = c - (At @ lam_new + nu_new)
        if np.min(true_slack_new) <= 0 or not np.all(np.isfinite(true_slack_new)):
            lr_current *= 0.5
            if step % 100 == 0:
                print(f"step {step} | rejected step (min new slack {np.min(true_slack_new):.3e}), lr->{lr_current:.3e}")
            continue

        # accept update
        lam_pos = lam_pos_new
        lam_neg = lam_neg_new
        nu      = nu_new

        # optionally decay mu only when the iterate is well interior
        if step % 200 == 0 and np.min(true_slack_new) > 1e-2:
            mu *= mu_decay

        if step % 1000 == 0:
            lam = lam_pos - lam_neg
            true_slack = c - (At @ lam + nu)
            dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
            print(f"step {step} | dual {dual_val:.6f} | min slack {true_slack.min():.3e} | max slack {true_slack.max():.3e}")

    lam_final = lam_pos - lam_neg
    dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu
    if upper:
        dual_val = -dual_val

    print('no. of near-zero slack constraints:', np.sum((c - (At @ lam_final + nu)) < 1e-4))
    return dual_val, lam_final, nu

import numpy as np

def solve_dual_ipm(A, b, c, eps=1e-8, max_iter=80, tol=1e-8):

    m, n = A.shape
    At = A.T

    # variables
    lam_pos = np.ones(m)
    lam_neg = np.ones(m)
    nu = 0.0
    s = np.ones(n)

    z = np.ones(n)  # dual variables for inequality

    for it in range(max_iter):

        lam = lam_pos - lam_neg

        # residuals
        r_p = At @ lam + nu + s - c
        r_d = (b+eps) - (b-eps)
        r_c = s * z

        mu = np.mean(r_c)

        if np.linalg.norm(r_p) < tol and mu < tol:
            break

        # KKT matrix pieces
        S = np.diag(s)
        Z = np.diag(z)

        # Schur complement
        M = A @ np.linalg.solve(S @ Z, A.T)

        rhs = -(b+eps - (b-eps))

        d_lam = np.linalg.solve(M + 1e-8*np.eye(m), rhs)

        d_nu = -np.mean(r_p)

        d_s = -r_p - At @ d_lam - d_nu
        d_z = (-r_c - Z @ d_s) / s

        # step length
        alpha = 1.0

        idx = d_s < 0
        if np.any(idx):
            alpha = min(alpha, 0.99*np.min(-s[idx]/d_s[idx]))

        idx = d_z < 0
        if np.any(idx):
            alpha = min(alpha, 0.99*np.min(-z[idx]/d_z[idx]))

        lam_pos += alpha*np.maximum(d_lam,0)
        lam_neg += alpha*np.maximum(-d_lam,0)
        nu += alpha*d_nu
        s += alpha*d_s
        z += alpha*d_z

        dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu

        print(f"iter {it} | dual {dual_val:.6f} | mu {mu:.3e}")

    lam_final = lam_pos - lam_neg
    dual_val = (b+eps)@lam_pos - (b-eps)@lam_neg + nu

    return dual_val, lam_final, nu

import torch

import torch

def solve_dual_torch(A, b, c, eps=1e-8, upper=False,
                     steps=10000,
                     lr=1e-4,
                     mu=1, mu_decay=0.99,
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

    optimizer = torch.optim.AdamW([lam_pos, lam_neg, nu], lr=lr)

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
        if (slack <= 0).any():
            # should not happen if fraction-to-boundary is working,
            # but keep a big penalty as a safety net
            infeas = -slack[slack <= 0]
            barrier = 1e6 * torch.mean(infeas**2)
        else:
            barrier = -torch.mean(torch.log(slack))

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
        tau = 0.99  # fraction-to-boundary
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

# def solve_dual_torch(A, b, c, eps=1e-8, upper=False,
#                      steps=10000,
#                      lr=1e-3,
#                      mu=1.0, mu_decay=0.99,
#                      device="cpu"):

#     A = torch.tensor(A, dtype=torch.float32, device=device)
#     b = torch.tensor(b, dtype=torch.float32, device=device)
#     c = torch.tensor(c, dtype=torch.float32, device=device)

#     sign = -1 if upper else 1
#     c = sign * c

#     m, n = A.shape

#     lam_pos = torch.zeros(m, device=device, requires_grad=True)
#     lam_neg = torch.zeros(m, device=device, requires_grad=True)
#     nu = torch.tensor(c.min().item() - 1.0,
#                       device=device,
#                       requires_grad=True)

#     optimizer = torch.optim.Adam([lam_pos, lam_neg, nu], lr=lr)

#     for step in range(steps):

#         optimizer.zero_grad()

#         lam = lam_pos - lam_neg
#         slack = c - A.T @ lam - nu

#         # dual objective
#         dual_obj = (
#             (b + eps) @ lam_pos
#             - (b - eps) @ lam_neg
#             + nu
#         )

#         # log-barrier (only valid when slack > 0)
#         if (slack <= 0).any():
#             infeas = -slack[slack <= 0]
#             barrier = 1e3 * torch.sum(infeas**2)
#         else:
#             barrier = -torch.sum(torch.log(slack))

#         loss = -dual_obj - mu * barrier
#         loss.backward()

#         # -------------------------------------------------------
#         #  LINE SEARCH
#         # -------------------------------------------------------

#         # Save old values
#         lam_pos_old = lam_pos.data.clone()
#         lam_neg_old = lam_neg.data.clone()
#         nu_old      = nu.data.clone()

#         # Let Adam compute its proposed step
#         optimizer.step()

#         # Proposed new values
#         lam_pos_new = lam_pos.data.clone()
#         lam_neg_new = lam_neg.data.clone()
#         nu_new      = nu.data.clone()

#         # Restore old values before line search
#         lam_pos.data.copy_(lam_pos_old)
#         lam_neg.data.copy_(lam_neg_old)
#         nu.data.copy_(nu_old)

#         alpha = 1.0
#         beta  = 0.5
#         min_slack_threshold = 1e-8

#         for _ in range(20):
#             lam_trial = (lam_pos_old + alpha * (lam_pos_new - lam_pos_old)) \
#                       - (lam_neg_old + alpha * (lam_neg_new - lam_neg_old))
#             nu_trial  = nu_old + alpha * (nu_new - nu_old)

#             slack_trial = c - A.T @ lam_trial - nu_trial

#             if slack_trial.min() > min_slack_threshold:
#                 break
#             alpha *= beta

#         # If still infeasible after 20 reductions, reject step
#         if slack_trial.min() <= min_slack_threshold:
#             alpha = 0.0

#         # Apply accepted step
#         lam_pos.data.copy_(lam_pos_old + alpha * (lam_pos_new - lam_pos_old))
#         lam_neg.data.copy_(lam_neg_old + alpha * (lam_neg_new - lam_neg_old))
#         nu.data.copy_(nu_old + alpha * (nu_new - nu_old))

#         # Enforce nonnegativity
#         with torch.no_grad():
#             lam_pos.clamp_(min=0.0)
#             lam_neg.clamp_(min=0.0)

#         if step % 2000 == 0:
#             lam = lam_pos - lam_neg
#             slack = c - A.T @ lam - nu
#             print(f"step {step} | dual {dual_obj.item():.6f} | "
#                   f"min slack {slack.min().item():.3e} | "
#                   f"max slack {slack.max().item():.3e}")
#             mu *= mu_decay

#     lam_final = (lam_pos - lam_neg).detach().cpu().numpy()

#     dual_val = (
#         (b + eps) @ lam_pos.detach()
#         - (b - eps) @ lam_neg.detach()
#         + nu.detach()
#     ).item()

#     if upper:
#         dual_val = -dual_val

#     lam = lam_pos - lam_neg
#     slack = c - A.T @ lam - nu
#     print("active constraints:", torch.sum(slack < 1e-4).item())

#     return dual_val, lam_final, nu.item()


# def solve_dual_torch(A, b, c, eps=1e-8, upper=False,
#                      steps=10000,
#                      lr=1e-3,
#                      mu=1.0, mu_decay=0.99,
#                      device="cpu"):

#     A = torch.tensor(A, dtype=torch.float32, device=device)
#     b = torch.tensor(b, dtype=torch.float32, device=device)
#     c = torch.tensor(c, dtype=torch.float32, device=device)

#     sign = -1 if upper else 1
#     c = sign*c

#     m, n = A.shape

#     lam_pos = torch.zeros(m, device=device, requires_grad=True)
#     lam_neg = torch.zeros(m, device=device, requires_grad=True)

#     nu = torch.tensor(c.min().item() - 1.0,
#                       device=device,
#                       requires_grad=True)

#     optimizer = torch.optim.AdamW(
#         [lam_pos, lam_neg, nu],
#         lr=lr
#     )

#     for step in range(steps):

#         optimizer.zero_grad()

#         lam = lam_pos - lam_neg

#         slack = c - A.T @ lam - nu

#         slack_safe = torch.clamp(slack, min=1e-8)
#         barrier = torch.sum(torch.log(slack_safe))

#         # if torch.any(slack < 0):
#         #     print("Barrier violated")
#         #     break

#         # barrier = torch.sum(torch.log(slack))

#         dual_obj = (
#             (b + eps) @ lam_pos
#             - (b - eps) @ lam_neg
#             + nu
#         )

#         loss = - (dual_obj  - mu * barrier)

#         loss.backward()

#         optimizer.step()
        
#         # enforce lam_pos, lam_neg >= 0
#         with torch.no_grad():
#             lam_pos.clamp_(min=0.0)
#             lam_neg.clamp_(min=0.0)

#         if step % 2000 == 0:
#             # print(
#             #     f"step {step} | "
#             #     f"dual {dual_obj.item():.6f} | "
#             #     f"min slack {slack.min().item():.3e}"
#             # )
#             print(f"step {step} | dual {dual_obj.item():.6f} |"
#                   f"min slack {slack.min().item():.3e} |"
#                   f"max slack {slack.max().item():.3e}")
#             mu *= mu_decay

#     lam_final = (lam_pos - lam_neg).detach().cpu().numpy()

#     dual_val = (
#         (b+eps) @ lam_pos.detach()
#         - (b-eps) @ lam_neg.detach()
#         + nu.detach()
#     ).item()

#     if upper:
#         dual_val = -dual_val

#     print("active constraints:", torch.sum(slack < 1e-4).item())

#     return dual_val, lam_final, nu.item()

if __name__=="__main__":
    n = 200000
    k = 8

    data = generate_data(n, lam=0.5) #np.random.rand())
    P = empirical_distribution(data,k)

    A,b,c,labels = build_constraints(P,k)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Dual variables:", len(b))
    print("Latent types:", len(c))

    lower, lamL, nuL = solve_dual_band(A[:-1],b[:-1],c,eps=1e-8,upper=False,lr=5e-4,steps=n)
    upper, lamU, nuU = solve_dual_band(A[:-1],b[:-1],c,eps=1e-8,upper=True,lr=5e-4,steps=n)

    # lower, lamL, nuL = solve_dual_torch(A[:-1],b[:-1],c,eps=1e-8,upper=False,lr=5e-6,steps=n,device=device)
    # upper, lamU, nuU = solve_dual_torch(A[:-1],b[:-1],c,eps=1e-8,upper=True,lr=5e-6,steps=n,device=device)


    # lower, lamL, nuL = solve_dual_ipm(A, b, c, eps=1e-8, max_iter=80, tol=1e-8)

    print("\n==============================")
    print("ATE lower bound:", lower)
    print("ATE upper bound:", upper)
    print("True ATE = 3")
    print("==============================")

    plot_dual_heatmap(lamL, labels[:-1], k, "Lower Bound Dual")
    plot_dual_heatmap(lamU, labels[:-1], k, "Upper Bound Dual")