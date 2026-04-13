"""
Convex risk functionals (Definition 4.17, Remark 4.17).

All risk measures accept a (batch,) P&L tensor and return a scalar.
Expected Shortfall and entropic risk are differentiable and suitable
for training; Value-at-Risk is for evaluation only.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def expected_shortfall(pnl: Tensor, alpha: float = 0.95) -> Tensor:
    """Differentiable Expected Shortfall (AVaR) at level *alpha*.

    ES_alpha = mean of the worst ceil(n*(1-alpha)) losses,
    where loss = -PnL.

    ``torch.sort`` is differentiable so gradients flow through the
    permutation.

    Args:
        pnl: (batch,) P&L values.
        alpha: confidence level in (0, 1).

    Returns:
        Scalar tensor (differentiable).
    """
    n = pnl.shape[0]
    k = max(1, math.ceil(n * (1.0 - alpha)))
    # Sort losses in descending order
    losses_sorted, _ = torch.sort(-pnl, descending=True)
    return losses_sorted[:k].mean()


def entropic_risk(pnl: Tensor, lam: float = 1.0) -> Tensor:
    """Entropic risk measure (Definition 4.17).

    rho_ent = (1/lambda) log E[exp(-lambda PnL)]

    Uses ``torch.logsumexp`` for numerical stability.

    Args:
        pnl: (batch,) P&L values.
        lam: risk-aversion parameter lambda > 0.

    Returns:
        Scalar tensor (differentiable).
    """
    n = pnl.shape[0]
    return (torch.logsumexp(-lam * pnl, dim=0) - math.log(n)) / lam


def value_at_risk(pnl: Tensor, alpha: float = 0.95) -> Tensor:
    """Value-at-Risk at level *alpha* (evaluation only, NOT differentiable).

    VaR_alpha = -quantile(PnL, 1 - alpha)

    Args:
        pnl: (batch,) P&L values.
        alpha: confidence level in (0, 1).

    Returns:
        Scalar tensor.
    """
    return -torch.quantile(pnl, 1.0 - alpha)


def mixed_risk(
    pnl: Tensor,
    alpha: float = 0.95,
    lam: float = 1.0,
    w: float = 0.5,
) -> Tensor:
    """Convex combination: w * ES_alpha + (1 - w) * rho_ent_lambda."""
    return w * expected_shortfall(pnl, alpha) + (1.0 - w) * entropic_risk(pnl, lam)


def compute_all_metrics(
    pnl: Tensor, alpha: float = 0.95, lam: float = 1.0,
) -> dict[str, float]:
    """Full evaluation vector M(delta) from Definition 6.5.

    Returns a dict with:
        mean_pnl, std_pnl, var_95, es_95, es_99,
        entropic_1, max_loss, min_pnl, skewness, kurtosis
    """
    pnl_f = pnl.detach().float()
    mu = pnl_f.mean()
    std = pnl_f.std()
    centered = pnl_f - mu

    return {
        "mean_pnl":   float(mu),
        "std_pnl":    float(std),
        "var_95":     float(value_at_risk(pnl_f, 0.95)),
        "es_95":      float(expected_shortfall(pnl_f, 0.95)),
        "es_99":      float(expected_shortfall(pnl_f, 0.99)),
        "entropic_1": float(entropic_risk(pnl_f, lam)),
        "max_loss":   float(-pnl_f.min()),
        "min_pnl":    float(pnl_f.min()),
        "skewness":   float((centered ** 3).mean() / (std ** 3 + 1e-12)),
        "kurtosis":   float((centered ** 4).mean() / (std ** 4 + 1e-12)),
    }
