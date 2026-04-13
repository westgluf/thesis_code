"""
Deep hedging via residual feedforward network (Definition 4.16).

The deep hedger learns a stationary policy F^theta that maps the
information set I_k = (t_k/T, log(S_k/S_0), tau_k/T, delta_{k-1})
to a hedge ratio delta_k in [0, 1].  The same network is applied at
every rebalancing time step (weight sharing).

Training minimises a convex risk functional of terminal P&L
(Definition 4.17 / Remark 4.18).
"""
from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall
from deep_hedging.hedging.delta_hedger import BlackScholesDelta


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Single residual block with skip connection.

    Structure: Linear → LeakyReLU → Linear → LeakyReLU,
    with a skip connection x → x + block(x).
    Uses LayerNorm (batch-size invariant) instead of BatchNorm
    for stable training with varying batch compositions.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class DeepHedgerFNN(nn.Module):
    """Deep hedging strategy via residual feedforward network (Definition 4.16).

    Architecture::

        input_proj:  Linear(input_dim -> hidden_dim) + LeakyReLU
        res_blocks:  N x ResidualBlock(hidden_dim)
        output_head: Linear(hidden_dim -> 1) + Sigmoid

    Sigmoid enforces delta in [0, 1] for European calls.

    Configurations:
        "baseline": n_res_blocks=1, hidden_dim=64   (~12k params)
        "medium":   n_res_blocks=2, hidden_dim=128  (~66k params)  [DEFAULT]
        "deep":     n_res_blocks=3, hidden_dim=128  (~115k params)

    Parameters
    ----------
    input_dim : int
        Dimension of the information set I_k (default 4).
    hidden_dim : int
        Width of hidden layers.
    n_res_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout probability (applied after input projection if > 0).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Map information set I_k to hedge ratio.

        Args:
            features: (batch, input_dim)

        Returns:
            delta: (batch, 1) in [0, 1]
        """
        return self.head(self.backbone(features))

    def hedge_paths(self, S: Tensor, T: float = 1.0, S0: float = 100.0) -> Tensor:
        """Compute hedge ratios for full price paths (uniform interface).

        Delegates to :func:`hedge_paths_deep`.
        """
        return hedge_paths_deep(self, S, T, S0)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(
    S: Tensor,
    t_grid: Tensor,
    T: float,
    deltas_prev: Tensor,
    k: int,
) -> Tensor:
    """Construct normalised information set I_k (Definition 4.16).

    Features:
        - t_k / T             (scaled time, in [0, 1])
        - log(S_k / S_0)      (log-moneyness, centred near 0)
        - (T - t_k) / T       (time-to-maturity, in [0, 1])
        - delta_{k-1}         (previous hedge ratio, in [0, 1])

    Casts to float32 for the network (simulator outputs float64).

    Args:
        S: (batch, n_steps+1) price paths.
        t_grid: (n_steps+1,) time grid.
        T: maturity.
        deltas_prev: (batch,) previous hedge ratio.
        k: current time step index.

    Returns:
        features: (batch, 4) in float32.
    """
    batch = S.shape[0]
    t_k = float(t_grid[k])
    S_k = S[:, k]
    S_0 = S[:, 0]

    feat = torch.stack([
        torch.full((batch,), t_k / T, dtype=S.dtype, device=S.device),
        torch.log(S_k / S_0),
        torch.full((batch,), (T - t_k) / T, dtype=S.dtype, device=S.device),
        deltas_prev,
    ], dim=1)   # (batch, 4)

    return feat.float()


# ---------------------------------------------------------------------------
# Hedging loop
# ---------------------------------------------------------------------------

def hedge_paths_deep(
    model: DeepHedgerFNN,
    S: Tensor,
    T: float,
    S0: float = 100.0,
) -> Tensor:
    """Apply the deep hedger across all time steps (Definition 4.16).

    Loops over k = 0 .. n-1 because delta_k depends on delta_{k-1}.
    Within each step the computation is fully batched over paths.

    Args:
        model: trained DeepHedgerFNN.
        S: (batch, n_steps+1) price paths.
        T: maturity.
        S0: initial spot (for log-moneyness; unused if S[:,0] available).

    Returns:
        deltas: (batch, n_steps) hedge ratios in [0, 1].
    """
    batch, n_plus_1 = S.shape
    n = n_plus_1 - 1
    device = S.device
    t_grid = torch.linspace(0.0, T, n_plus_1, device=device)

    deltas_list: list[Tensor] = []
    delta_prev = torch.zeros(batch, dtype=S.dtype, device=device)

    for k in range(n):
        feat = build_features(S, t_grid, T, delta_prev, k)
        delta_k = model(feat).squeeze(-1)                  # (batch,)
        deltas_list.append(delta_k)
        delta_prev = delta_k.detach()                       # no grad through recurrence

    return torch.stack(deltas_list, dim=1)                  # (batch, n)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deep_hedger(
    model: DeepHedgerFNN,
    S_train: Tensor,
    S_val: Tensor,
    K: float = 100.0,
    T: float = 1.0,
    S0: float = 100.0,
    p0: float | None = None,
    cost_lambda: float = 0.0,
    risk_fn: Callable[[Tensor], Tensor] | None = None,
    alpha: float = 0.95,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 200,
    patience: int = 20,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train the deep hedger by minimising a convex risk functional.

    Args:
        model: DeepHedgerFNN to train (modified in-place).
        S_train: (n_train, n_steps+1) training price paths.
        S_val: (n_val, n_steps+1) validation price paths.
        K: strike price.
        T: maturity.
        S0: initial spot price.
        p0: initial capital.  If None, uses BS call price with sigma=0.235.
        cost_lambda: proportional transaction cost coefficient.
        risk_fn: callable pnl -> scalar loss.  Default: ES_alpha.
        alpha: ES confidence level (used when risk_fn is None).
        lr: learning rate for Adam.
        batch_size: mini-batch size.
        epochs: maximum training epochs.
        patience: early stopping patience.
        device: torch device (default: cpu).
        verbose: print progress every 10 epochs.

    Returns:
        history dict with keys: train_risk, val_risk, best_epoch, best_val_risk.
    """
    if device is None:
        device = torch.device("cpu")

    if p0 is None:
        p0 = BlackScholesDelta.bs_call_price(S0, K, T, sigma=0.235)

    # Default: Rockafellar-Uryasev smooth CVaR with learnable quantile w.
    # This is much more stable for SGD than the sort-based ES used in
    # evaluation.  The jointly-optimised w converges to VaR_alpha.
    _w_param: torch.Tensor | None = None
    if risk_fn is None:
        _w_param = torch.nn.Parameter(torch.tensor(0.0, device=device))

        def _cvar_loss(pnl: Tensor) -> Tensor:
            loss = -pnl
            return _w_param + torch.relu(loss - _w_param).mean() / (1.0 - alpha)

        risk_fn = _cvar_loss

    model = model.to(device)
    S_train = S_train.to(device)
    S_val = S_val.to(device)

    params = list(model.parameters())
    if _w_param is not None:
        params.append(_w_param)
    optimiser = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    n_train = S_train.shape[0]

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_risks: list[float] = []
    val_risks: list[float] = []

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses: list[float] = []

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            S_batch = S_train[idx]

            optimiser.zero_grad(set_to_none=True)

            deltas = model.hedge_paths(S_batch, T, S0)
            # Cast deltas to match S dtype for PnL computation
            deltas_pnl = deltas.to(S_batch.dtype)
            payoff = compute_payoff(S_batch, K, "call")
            pnl = compute_hedging_pnl(S_batch, deltas_pnl, payoff, p0, cost_lambda)
            loss = risk_fn(pnl)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_losses.append(float(loss.detach()))

        train_risk = sum(epoch_losses) / len(epoch_losses)
        train_risks.append(train_risk)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            deltas_val = model.hedge_paths(S_val, T, S0)
            deltas_val = deltas_val.to(S_val.dtype)
            payoff_val = compute_payoff(S_val, K, "call")
            pnl_val = compute_hedging_pnl(S_val, deltas_val, payoff_val, p0, cost_lambda)
            val_risk = float(risk_fn(pnl_val))
        val_risks.append(val_risk)

        # ---- early stopping ----
        if val_risk < best_val - 1e-6:
            best_val = val_risk
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}  train_risk={train_risk:.4f}  val_risk={val_risk:.4f}  best={best_val:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    return {
        "train_risk": train_risks,
        "val_risk": val_risks,
        "best_epoch": best_epoch,
        "best_val_risk": best_val,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_deep_hedger(
    model: DeepHedgerFNN,
    S_test: Tensor,
    K: float = 100.0,
    T: float = 1.0,
    S0: float = 100.0,
    p0: float | None = None,
    cost_lambda: float = 0.0,
) -> Tensor:
    """Evaluate the trained deep hedger on test paths.

    Args:
        model: trained DeepHedgerFNN.
        S_test: (n_test, n_steps+1) price paths.
        K, T, S0, p0, cost_lambda: same as train_deep_hedger.

    Returns:
        pnl: (n_test,) terminal P&L.
    """
    if p0 is None:
        p0 = BlackScholesDelta.bs_call_price(S0, K, T, sigma=0.235)

    model.eval()
    with torch.no_grad():
        deltas = model.hedge_paths(S_test, T, S0)
        deltas = deltas.to(S_test.dtype)
        payoff = compute_payoff(S_test, K, "call")
        return compute_hedging_pnl(S_test, deltas, payoff, p0, cost_lambda)
