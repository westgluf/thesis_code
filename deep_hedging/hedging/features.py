"""
Path-dependent features for rough-vol-aware deep hedging (Definition 6.3).

All features are computed from the price path alone (no observation of
latent variance), ensuring a fair comparison with BS delta.

Feature sets:
  - "flat":     4 dims  (t/T, log-moneyness, tau/T, delta_{k-1})
  - "sig-3":    7 dims  (flat + multi-scale realised volatility)
  - "sig-full": 12 dims (sig-3 + roughness ratio, vol-of-vol proxy,
                         running max/min, normalised QV)

The module precomputes features for the entire trajectory in a single
batched pass; the hedging loop indexes into the tensor at each step.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─── Constants ───────────────────────────────────────────────
DEFAULT_WINDOWS: Tuple[int, ...] = (5, 15, 50)
DEFAULT_VOV_WINDOW: int = 15
ROUGHNESS_CLAMP: Tuple[float, float] = (0.2, 5.0)
VOV_CLAMP: Tuple[float, float] = (0.0, 10.0)
EPS: float = 1e-10


class PathFeatureExtractor(nn.Module):
    """Compute path-dependent features from a batch of price paths.

    Parameters
    ----------
    feature_set : str
        One of ``"flat"``, ``"sig-3"``, ``"sig-full"``.
    windows : tuple of int
        Realised-volatility window sizes (in time steps).
    vov_window : int
        Window for the vol-of-vol proxy Q_k.
    xi0 : float
        Initial forward variance (for normalisation).
    eta_ref : float
        Reference vol-of-vol for Q_k normalisation.
    T : float
        Maturity.
    """

    # Index of the delta_{k-1} feature in the output tensor.
    # Constant across all feature sets (always the 4th flat feature).
    DELTA_PREV_INDEX: int = 3

    def __init__(
        self,
        feature_set: str = "sig-full",
        windows: Tuple[int, ...] = DEFAULT_WINDOWS,
        vov_window: int = DEFAULT_VOV_WINDOW,
        xi0: float = 0.235 ** 2,
        eta_ref: float = 1.9,
        T: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_set not in ("flat", "sig-3", "sig-full"):
            raise ValueError(f"Unknown feature_set: {feature_set!r}")

        self.feature_set = feature_set
        self.windows = tuple(windows)
        self.vov_window = vov_window
        self.xi0 = xi0
        self.sqrt_xi0 = math.sqrt(xi0)
        self.eta_ref_sq = eta_ref ** 2
        self.T = T

        if feature_set == "flat":
            self.n_features = 4
        elif feature_set == "sig-3":
            self.n_features = 4 + len(self.windows)
        else:
            self.n_features = 4 + len(self.windows) + 5

    # ─── Main entry point ────────────────────────────────────
    def compute_all_features(
        self,
        S: Tensor,
        deltas_prev: Tensor,
        S0: Optional[float] = None,
    ) -> Tensor:
        """Precompute features for every rebalancing step.

        Parameters
        ----------
        S : (batch, n_steps + 1)
        deltas_prev : (batch, n_steps)
            delta_{k-1} for k = 0 … n-1.  deltas_prev[:, 0] should be 0.
        S0 : float, optional

        Returns
        -------
        (batch, n_steps, n_features)
        """
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        dt = self.T / n
        device = S.device
        dtype = S.dtype

        S0_t = torch.full((batch, 1), S0, dtype=dtype, device=device) if S0 is not None else S[:, 0:1]

        # ── Flat features (batch, n, 4) ──
        t_k = torch.arange(n, dtype=dtype, device=device) * dt          # (n,)
        f1 = (t_k / self.T).unsqueeze(0).expand(batch, -1)
        f3 = 1.0 - f1
        log_S = torch.log(S / S0_t)                                      # (batch, n+1)
        f2 = log_S[:, :n]
        f4 = deltas_prev
        flat = torch.stack([f1, f2, f3, f4], dim=-1)                     # (batch, n, 4)

        if self.feature_set == "flat":
            return flat

        # ── Shared: log returns + cumulative squared returns ──
        log_ret = torch.diff(torch.log(S.clamp(min=EPS)), dim=1)         # (batch, n)
        sq_ret = log_ret ** 2
        cum_pad = F.pad(torch.cumsum(sq_ret, dim=1), (1, 0))             # (batch, n+1)

        # ── Multi-scale realised vol ──
        rv = self._windowed_rv(cum_pad, dt, n, device, dtype, batch)     # (batch, n, W)
        rv_norm = rv / self.sqrt_xi0

        if self.feature_set == "sig-3":
            return torch.cat([flat, rv_norm], dim=-1)

        # ── sig-full extras ──
        rv_short = rv[..., 0]                                             # (batch, n)
        rv_long = rv[..., -1]
        R_k = torch.clamp(rv_short / (rv_long + EPS), *ROUGHNESS_CLAMP)

        Q_k = self._vol_of_vol(rv_short, dt)                             # (batch, n)
        Q_k = torch.clamp(Q_k / self.eta_ref_sq, *VOV_CLAMP)

        run_max, _ = torch.cummax(log_S[:, :n], dim=1)
        run_min, _ = torch.cummin(log_S[:, :n], dim=1)

        QV = cum_pad[:, :n]
        t_safe = torch.clamp(t_k, min=dt).unsqueeze(0)
        QV_norm = torch.clamp(QV / (self.xi0 * t_safe), 0.0, 10.0)

        extra = torch.stack([R_k, Q_k, run_max, run_min, QV_norm], dim=-1)
        return torch.cat([flat, rv_norm, extra], dim=-1)

    # ─── Helpers ─────────────────────────────────────────────
    def _windowed_rv(self, cum_pad, dt, n, device, dtype, batch):
        """Realised vol for each window at every step."""
        k = torch.arange(n, dtype=torch.long, device=device)
        rv_list = []
        for w in self.windows:
            start = torch.clamp(k - w, min=0)
            win_sz = torch.clamp((k - start).to(dtype), min=1.0)
            end_exp = k.unsqueeze(0).expand(batch, -1)
            start_exp = start.unsqueeze(0).expand(batch, -1)
            s = torch.gather(cum_pad, 1, end_exp) - torch.gather(cum_pad, 1, start_exp)
            sigma = torch.sqrt(torch.clamp(s / (win_sz * dt + EPS), min=EPS))
            # Sentinel at k=0
            sigma[:, 0] = self.sqrt_xi0
            rv_list.append(sigma)
        return torch.stack(rv_list, dim=-1)

    def _vol_of_vol(self, rv_short, dt):
        """Q_k: windowed mean squared log-increment of short RV."""
        batch, n = rv_short.shape
        device, dtype = rv_short.device, rv_short.dtype
        log_rv = torch.log(torch.clamp(rv_short, min=EPS))
        d_log = torch.diff(log_rv, dim=1)                               # (batch, n-1)
        sq_d = d_log ** 2
        cum = F.pad(torch.cumsum(sq_d, dim=1), (1, 0))                  # (batch, n)

        k = torch.arange(n, dtype=torch.long, device=device)
        end_raw = torch.clamp(k - 1, min=0)
        start_raw = torch.clamp(k - 1 - self.vov_window, min=0)
        valid = (k >= 2).to(dtype).unsqueeze(0)

        end_exp = end_raw.unsqueeze(0).expand(batch, -1)
        start_exp = start_raw.unsqueeze(0).expand(batch, -1)
        s = torch.gather(cum, 1, end_exp) - torch.gather(cum, 1, start_exp)
        win = torch.clamp((end_raw - start_raw).to(dtype), min=1.0).unsqueeze(0)
        return s / (win * dt + EPS) * valid

    def forward(self, S, deltas_prev, S0=None):
        return self.compute_all_features(S, deltas_prev, S0)
