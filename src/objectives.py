from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


ObjectiveName = Literal["cvar", "entropic", "mean_variance"]


def cvar_loss_from_pl(pl: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    loss = -pl
    a = float(alpha)
    return w + torch.relu(loss - w).mean() / (1.0 - a)


def entropic_risk_loss(pl: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    gamma_value = float(gamma)
    if gamma_value <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    scaled = -gamma_value * pl
    return (torch.logsumexp(scaled, dim=0) - math.log(pl.numel())) / gamma_value


def mean_variance_loss(pl: torch.Tensor, lambda_mv: float = 1.0) -> torch.Tensor:
    lambda_value = float(lambda_mv)
    if lambda_value < 0.0:
        raise ValueError(f"lambda_mv must be non-negative, got {lambda_mv}")
    mean_pl = pl.mean()
    var_pl = pl.var(unbiased=False)
    return -mean_pl + 0.5 * lambda_value * var_pl


class PnLObjective(nn.Module):
    objective_name: str = "unknown"

    def monitor_value(self) -> float:
        return float("nan")


class CVaRObjective(PnLObjective):
    objective_name = "cvar"

    def __init__(self, alpha: float = 0.95, w0: float = 0.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.w = nn.Parameter(torch.tensor(float(w0)))

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return cvar_loss_from_pl(pl, self.w, alpha=self.alpha)

    def monitor_value(self) -> float:
        return float(self.w.detach().cpu().item())


class EntropicRiskObjective(PnLObjective):
    objective_name = "entropic"

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return entropic_risk_loss(pl, gamma=self.gamma)


class MeanVarianceObjective(PnLObjective):
    objective_name = "mean_variance"

    def __init__(self, lambda_mv: float = 1.0) -> None:
        super().__init__()
        self.lambda_mv = float(lambda_mv)

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return mean_variance_loss(pl, lambda_mv=self.lambda_mv)


def canonical_objective_name(name: str | None) -> ObjectiveName:
    raw = "cvar" if name is None else str(name).strip().lower().replace("-", "_")
    if raw in {"es", "cvar", "expected_shortfall"}:
        return "cvar"
    if raw in {"entropic", "entropic_risk", "exponential_utility"}:
        return "entropic"
    if raw in {"mean_variance", "mv"}:
        return "mean_variance"
    valid = "cvar, entropic, mean_variance"
    raise ValueError(f"unknown objective name {name!r}; expected one of: {valid}")


def build_objective(
    *,
    name: str | None = None,
    alpha: float = 0.95,
    gamma: float = 1.0,
    lambda_mv: float = 1.0,
    w0: float = 0.0,
) -> PnLObjective:
    objective_name = canonical_objective_name(name)
    if objective_name == "cvar":
        return CVaRObjective(alpha=alpha, w0=w0)
    if objective_name == "entropic":
        return EntropicRiskObjective(gamma=gamma)
    return MeanVarianceObjective(lambda_mv=lambda_mv)
