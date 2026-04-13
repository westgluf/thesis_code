"""
Heston stochastic volatility simulator (nn.Module).

Uses the full-truncation Euler scheme (Lord, Koekkoek & Van Dijk 2010)
for variance positivity.  A loop over time steps is used because the
variance recursion is inherently sequential; the batch dimension is
still fully vectorised.

Provides the same forward API as :class:`DifferentiableRoughBergomi`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Heston(nn.Module):
    """Heston model with full-truncation Euler discretisation.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    v0 : float
        Initial instantaneous variance.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance level.
    sigma_v : float
        Vol-of-vol.
    rho : float
        Spot-vol correlation.
    """

    def __init__(
        self,
        n_steps: int = 100,
        T: float = 1.0,
        v0: float = 0.235 ** 2,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma_v: float = 2.0,
        rho: float = -0.7,
    ) -> None:
        super().__init__()
        for name, val in [
            ("_v0", v0), ("_kappa", kappa), ("_theta", theta),
            ("_sigma_v", sigma_v), ("_rho", rho),
        ]:
            self.register_buffer(name, torch.tensor(val, dtype=torch.float64))

        self.register_buffer("_n_steps", torch.tensor(n_steps, dtype=torch.long))
        self.register_buffer("_T", torch.tensor(T, dtype=torch.float64))
        self.register_buffer("_dt", torch.tensor(T / n_steps, dtype=torch.float64))

        t_grid = torch.linspace(0.0, T, n_steps + 1, dtype=torch.float64)
        self.register_buffer("t_grid", t_grid)

    # ---- properties -------------------------------------------------------

    @property
    def dt(self) -> Tensor:
        return self._dt

    @property
    def n_steps(self) -> int:
        return int(self._n_steps.item())

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        Z_vol: Tensor,         # [batch, n_steps, 2]
        Z_price: Tensor,       # [batch, n_steps]
        S0: Tensor | float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """Simulate Heston paths via full-truncation Euler.

        Parameters
        ----------
        Z_vol : Tensor [batch, n_steps, 2]
            Uses ``Z_vol[:, :, 0]`` for the variance diffusion.
        Z_price : Tensor [batch, n_steps]
            Independent normal for the price orthogonal component.
        S0 : float or Tensor
            Initial spot price.

        Returns
        -------
        S : Tensor [batch, n_steps + 1]
        V : Tensor [batch, n_steps + 1]
        """
        if not isinstance(S0, Tensor):
            S0 = torch.tensor(S0, dtype=Z_vol.dtype, device=Z_vol.device)

        batch, n = Z_price.shape
        dt = self._dt
        sqrt_dt = torch.sqrt(dt)
        rho = self._rho
        rho_perp = torch.sqrt(1.0 - rho * rho)

        V_list: list[Tensor] = [self._v0.expand(batch)]
        log_inc_list: list[Tensor] = []

        for k in range(n):
            V_k = V_list[-1]
            V_pos = torch.clamp(V_k, min=0.0)      # full truncation

            # Variance update
            V_next = (
                V_k
                + self._kappa * (self._theta - V_pos) * dt
                + self._sigma_v * torch.sqrt(V_pos * dt) * Z_vol[:, k, 0]
            )
            V_next = torch.clamp(V_next, min=0.0)   # truncate again
            V_list.append(V_next)

            # Correlated price Brownian increment
            dB_k = rho * sqrt_dt * Z_vol[:, k, 0] + rho_perp * sqrt_dt * Z_price[:, k]

            # Log-price increment
            log_inc_list.append(-0.5 * V_pos * dt + torch.sqrt(V_pos) * dB_k)

        V = torch.stack(V_list, dim=-1)              # [batch, n + 1]
        log_inc = torch.stack(log_inc_list, dim=-1)   # [batch, n]

        log_S = torch.cumsum(log_inc, dim=-1)
        log_S = torch.cat([
            torch.zeros(batch, 1, dtype=Z_vol.dtype, device=Z_vol.device),
            log_S,
        ], dim=-1)

        S = S0 * torch.exp(log_S)
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
