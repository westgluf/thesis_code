"""
Shared training convenience wrappers for Prompt 9 (Pareto front).

Provides:
  - make_objective(name, **kwargs)
      Returns a differentiable loss fn for one of 'es', 'entropic', 'mse', 'mean'.
  - make_objective_tag(name, kwargs)
      Short filename-safe tag for an (objective, kwargs) pair.
  - train_deep_hedger_with_objective(...)
      Trains one DeepHedgerFNN under the specified risk objective and
      returns {'model', 'history', 'train_time_s'}.

These wrap the existing ``train_deep_hedger`` / ``DeepHedgerFNN`` from
Prompt 3 (refactored in Prompt 7 for the generic hedger interface).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor

from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger
from deep_hedging.objectives.risk_measures import (
    expected_shortfall,
    entropic_risk,
)


def make_objective(name: str, **kwargs) -> Callable[[Tensor], Tensor]:
    """Return a differentiable loss function for the given risk measure.

    Parameters
    ----------
    name : str
        One of ``'es'``, ``'entropic'``, ``'mse'``, ``'mean'``.
    kwargs : dict
        For ``'es'``: ``alpha`` (default 0.95).
        For ``'entropic'``: ``lam`` (default 1.0).
    """
    key = name.lower()
    if key == "es":
        alpha = float(kwargs.get("alpha", 0.95))
        return lambda pnl: expected_shortfall(pnl, alpha)
    if key == "entropic":
        lam = float(kwargs.get("lam", 1.0))
        return lambda pnl: entropic_risk(pnl, lam)
    if key == "mse":
        return lambda pnl: (pnl ** 2).mean()
    if key == "mean":
        return lambda pnl: (-pnl).mean()
    raise ValueError(f"Unknown objective: {name!r}")


def make_objective_tag(name: str, kwargs: dict) -> str:
    """Return a short filename-safe tag for (name, kwargs)."""
    key = name.lower()
    if key == "es":
        return f"es_a{float(kwargs.get('alpha', 0.95)):.2f}"
    if key == "entropic":
        return f"entropic_l{float(kwargs.get('lam', 1.0)):.1f}"
    return key  # mse, mean


def train_deep_hedger_with_objective(
    S_train: Tensor,
    S_val: Tensor,
    objective_name: str,
    objective_kwargs: dict,
    cost_lambda: float,
    p0: float,
    *,
    K: float = 100.0,
    T: float = 1.0,
    S0: float = 100.0,
    hidden_dim: int = 128,
    n_res_blocks: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 2048,
    patience: int = 30,
    seed: int = 2024,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Train one deep hedger with a given risk objective.

    Uses the existing ``train_deep_hedger`` (the generic-hedger interface
    introduced in Prompt 7). The only thing that changes between calls is
    the ``risk_fn`` produced by :func:`make_objective`.

    Returns
    -------
    dict with keys ``'model'``, ``'history'``, ``'train_time_s'``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepHedgerFNN(
        input_dim=4, hidden_dim=hidden_dim, n_res_blocks=n_res_blocks,
    )
    risk_fn = make_objective(objective_name, **(objective_kwargs or {}))

    t0 = time.time()
    history = train_deep_hedger(
        model, S_train, S_val,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=cost_lambda,
        risk_fn=risk_fn,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        device=device,
        verbose=False,
    )
    train_time = time.time() - t0

    return {"model": model, "history": history, "train_time_s": train_time}
