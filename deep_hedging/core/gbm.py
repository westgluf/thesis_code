"""
Geometric Brownian Motion simulator (nn.Module).

Provides the same forward API as :class:`DifferentiableRoughBergomi` so
that all hedging code can be model-agnostic.  Under the risk-neutral
(discounted) measure the dynamics are simply dS = σ S dW.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GBM(nn.Module):
    """Geometric Brownian Motion with constant volatility.

    Forward signature matches the rough Bergomi module for seamless
    substitution in the hedging pipeline.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    sigma : float
        Constant volatility.
    """

    def __init__(self, n_steps: int = 100, T: float = 1.0, sigma: float = 0.235) -> None:
        super().__init__()
        self.register_buffer("_sigma", torch.tensor(sigma, dtype=torch.float64))
        self.register_buffer("_n_steps", torch.tensor(n_steps, dtype=torch.long))
        self.register_buffer("_T", torch.tensor(T, dtype=torch.float64))
        self.register_buffer("_dt", torch.tensor(T / n_steps, dtype=torch.float64))

        t_grid = torch.linspace(0.0, T, n_steps + 1, dtype=torch.float64)
        self.register_buffer("t_grid", t_grid)

    # ---- properties -------------------------------------------------------

    @property
    def sigma(self) -> Tensor:
        return self._sigma

    @property
    def dt(self) -> Tensor:
        return self._dt

    @property
    def n_steps(self) -> int:
        return int(self._n_steps.item())

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        Z_vol: Tensor,         # [batch, n_steps, 2]  — ignored
        Z_price: Tensor,       # [batch, n_steps]
        S0: Tensor | float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """Simulate GBM paths.

        Parameters
        ----------
        Z_vol : Tensor [batch, n_steps, 2]
            **Ignored** — present only for API compatibility.
        Z_price : Tensor [batch, n_steps]
            i.i.d. N(0,1) driving the asset price.
        S0 : float or Tensor
            Initial spot price.

        Returns
        -------
        S : Tensor [batch, n_steps + 1]
        V : Tensor [batch, n_steps + 1]  — constant σ² everywhere.
        """
        if not isinstance(S0, Tensor):
            S0 = torch.tensor(S0, dtype=Z_price.dtype, device=Z_price.device)

        sigma = self._sigma
        dt = self._dt
        batch, n = Z_price.shape

        # Log-Euler (exact for GBM): log(S_{k+1}/S_k) = -½σ²dt + σ√dt Z_k
        log_inc = -0.5 * sigma ** 2 * dt + sigma * torch.sqrt(dt) * Z_price
        log_S = torch.cumsum(log_inc, dim=-1)                    # [batch, n]
        log_S = torch.cat([
            torch.zeros(batch, 1, dtype=Z_price.dtype, device=Z_price.device),
            log_S,
        ], dim=-1)                                                # [batch, n+1]

        S = S0 * torch.exp(log_S)

        # Constant variance
        V = (sigma ** 2) * torch.ones(
            batch, n + 1, dtype=Z_price.dtype, device=Z_price.device,
        )
        return S, V

    # ---- convenience ------------------------------------------------------

    def simulate(
        self,
        n_paths: int,
        S0: float = 100.0,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate paths with internally created noise.

        Returns (S, V, t_grid).
        """
        dev = device or self._dt.device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev).manual_seed(seed)
        n = self.n_steps
        Z_vol = torch.randn(n_paths, n, 2, dtype=torch.float64, device=dev, generator=gen)
        Z_price = torch.randn(n_paths, n, dtype=torch.float64, device=dev, generator=gen)
        S, V = self.forward(Z_vol, Z_price, S0)
        return S, V, self.t_grid
