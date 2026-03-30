import torch
import numpy as np

def compute_pl_torch(S: torch.Tensor, deltas: torch.Tensor, Z: torch.Tensor, p0: float, lam: float) -> torch.Tensor:
    dS = S[:, 1:] - S[:, :-1]
    gains = (deltas * dS).sum(dim=1)
    delta_prev = torch.cat([torch.zeros((S.shape[0], 1), device=S.device), deltas[:, :-1]], dim=1)
    trade = deltas - delta_prev
    costs = (lam * S[:, :-1] * trade.abs()).sum(dim=1)
    close_cost = lam * S[:, -1] * deltas[:, -1].abs()
    costs = costs + close_cost
    pl = -Z + p0 + gains - costs
    return pl

def rollout_strategy(model, feats_base: torch.Tensor) -> torch.Tensor:
    N, n, _ = feats_base.shape
    deltas = []
    delta_prev = torch.zeros((N, 1), device=feats_base.device)
    for k in range(n):
        x = feats_base[:, k, :].clone()
        x[:, 3:4] = delta_prev
        delta_k = model(x)
        delta_k = torch.tanh(delta_k)
        deltas.append(delta_k)
        delta_prev = delta_k
    return torch.cat(deltas, dim=1)
