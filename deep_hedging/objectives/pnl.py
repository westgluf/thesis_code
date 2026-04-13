"""
Hedging P&L computation (Definition 5.5).

All functions are fully differentiable through PyTorch autograd and
operate in batch-vectorised form — no Python loops over paths or
time steps.
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_payoff(
    S: Tensor, K: float, option_type: str = "call",
) -> Tensor:
    """Terminal payoff of a European option.

    Args:
        S: (batch, n_steps + 1) price paths.
        K: Strike price.
        option_type: ``"call"`` for (S_T - K)^+ or ``"put"`` for (K - S_T)^+.

    Returns:
        payoff: (batch,)
    """
    S_T = S[:, -1]
    if option_type == "call":
        return torch.relu(S_T - K)
    elif option_type == "put":
        return torch.relu(K - S_T)
    else:
        raise ValueError(f"Unknown option_type {option_type!r}; use 'call' or 'put'")


def compute_trading_gains(S: Tensor, deltas: Tensor) -> Tensor:
    """Discrete trading gains: sum_k delta_k (S_{k+1} - S_k).

    Args:
        S: (batch, n_steps + 1)
        deltas: (batch, n_steps)

    Returns:
        gains: (batch,)
    """
    dS = S[:, 1:] - S[:, :-1]          # (batch, n_steps)
    return (deltas * dS).sum(dim=1)


def compute_transaction_costs(
    S: Tensor, deltas: Tensor, cost_lambda: float,
) -> Tensor:
    """Proportional transaction costs (Definition 4.3).

    C_T = lambda * sum_{k=0}^{n-1} |delta_k - delta_{k-1}| * S_k

    with delta_{-1} = 0 (zero initial position).

    Args:
        S: (batch, n_steps + 1)
        deltas: (batch, n_steps)
        cost_lambda: proportional cost coefficient lambda >= 0.

    Returns:
        costs: (batch,)
    """
    if cost_lambda == 0.0:
        return torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)

    # Prepend zero column for delta_{-1} = 0
    delta_prev = torch.cat([
        torch.zeros(deltas.shape[0], 1, dtype=deltas.dtype, device=deltas.device),
        deltas[:, :-1],
    ], dim=1)                                       # (batch, n_steps)

    trades = torch.abs(deltas - delta_prev)         # (batch, n_steps)
    S_rebal = S[:, :-1]                             # (batch, n_steps)
    return cost_lambda * (trades * S_rebal).sum(dim=1)


def compute_hedging_pnl(
    S: Tensor,
    deltas: Tensor,
    payoff: Tensor,
    p0: float,
    cost_lambda: float = 0.0,
) -> Tensor:
    """Pathwise terminal P&L (Definition 5.5).

    PL_T = -Z + p_0 + sum delta_k (S_{k+1} - S_k) - C_T(delta)

    Args:
        S: (batch, n_steps + 1) price paths.
        deltas: (batch, n_steps) hedge ratios.
        payoff: (batch,) terminal payoff Z.
        p0: initial capital (option premium).
        cost_lambda: proportional cost coefficient.

    Returns:
        pnl: (batch,)
    """
    gains = compute_trading_gains(S, deltas)
    costs = compute_transaction_costs(S, deltas, cost_lambda)
    return -payoff + p0 + gains - costs
