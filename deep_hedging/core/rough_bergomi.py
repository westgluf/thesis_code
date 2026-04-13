"""
Differentiable rough Bergomi simulator (nn.Module).

Implements the rBergomi model of Bayer, Friz & Gatheral (2016) with a
fully differentiable forward pass.  The variance process is driven by
a fractional Brownian motion produced by the :class:`HybridVolterraDriver`,
and the asset price follows a log-Euler scheme with correlated innovations.

All computations use float64 for numerical stability with small H.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .volterra import HybridVolterraDriver


class DifferentiableRoughBergomi(nn.Module):
    """Rough Bergomi model with pathwise-differentiable simulation.

    The dynamics (Definition 5.9) are:

        v(t_k) = ξ₀ · exp(η · W^H(t_k) − ½ η² t_k^{2H})
        dB_k   = ρ dW_k + √(1 − ρ²) √dt Z^price_k
        S_{k+1}= S_k · exp(−½ v_k dt + √v_k · dB_k)

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    H : float
        Hurst exponent.
    eta : float
        Vol-of-vol parameter η.
    rho : float
        Spot-vol correlation ρ.
    xi0 : float
        Flat forward variance curve ξ₀.
    """

    def __init__(
        self,
        n_steps: int = 100,
        T: float = 1.0,
        H: float = 0.07,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.235 ** 2,
    ) -> None:
        super().__init__()
        self.volterra = HybridVolterraDriver(n_steps, T, H)

        self.register_buffer("_eta", torch.tensor(eta, dtype=torch.float64))
        self.register_buffer("_rho", torch.tensor(rho, dtype=torch.float64))
        self.register_buffer("_xi0", torch.tensor(xi0, dtype=torch.float64))

        t_grid = torch.linspace(0.0, T, n_steps + 1, dtype=torch.float64)
        self.register_buffer("t_grid", t_grid)

    # ---- properties -------------------------------------------------------

    @property
    def H(self) -> Tensor:
        return self.volterra.H

    @property
    def eta(self) -> Tensor:
        return self._eta

    @property
    def rho(self) -> Tensor:
        return self._rho

    @property
    def xi0(self) -> Tensor:
        return self._xi0

    @property
    def dt(self) -> Tensor:
        return self.volterra.dt

    @property
    def n_steps(self) -> int:
        return self.volterra.n_steps

    # ---- differentiability ------------------------------------------------

    def make_params_differentiable(self) -> None:
        """Promote η and ρ from buffers to ``nn.Parameter`` (Proposition 6.1).

        After calling this method, ``loss.backward()`` will populate
        ``.grad`` on ``self._eta`` and ``self._rho``.
        """
        eta_val = self._eta.detach().clone()
        rho_val = self._rho.detach().clone()
        del self._eta, self._rho
        self._eta = nn.Parameter(eta_val)
        self._rho = nn.Parameter(rho_val)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        Z_vol: Tensor,         # [batch, n_steps, 2]
        Z_price: Tensor,       # [batch, n_steps]
        S0: Tensor | float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """Simulate rBergomi paths.

        Parameters
        ----------
        Z_vol : Tensor [batch, n_steps, 2]
            Driving noise for the Volterra fBm.
        Z_price : Tensor [batch, n_steps]
            Independent N(0,1) for the price orthogonal component.
        S0 : float or Tensor
            Initial spot price.

        Returns
        -------
        S : Tensor [batch, n_steps + 1]
            Asset price paths.
        V : Tensor [batch, n_steps + 1]
            Instantaneous variance paths.
        """
        if not isinstance(S0, Tensor):
            S0 = torch.tensor(S0, dtype=Z_vol.dtype, device=Z_vol.device)

        H = self.volterra.H
        eta = self._eta
        rho = self._rho
        xi0 = self._xi0
        dt = self.volterra.dt
        batch = Z_vol.shape[0]

        # --- Step 1: fractional Brownian motion ---
        WH, dW = self.volterra(Z_vol)   # WH [batch, n+1], dW [batch, n]

        # --- Step 2: variance process (Definition 5.9) ---
        # v_k = ξ₀ · exp(η · WH_k − ½ η² t_k^{2H})
        t_2H = self.t_grid ** (2.0 * H)                      # [n + 1]
        V = xi0 * torch.exp(eta * WH - 0.5 * eta ** 2 * t_2H.unsqueeze(0))
        #                                                     # [batch, n+1]

        # Variance at hedge-rebalancing times (k = 0 … n−1)
        V_hedge = torch.clamp(V[:, :-1], min=1e-12)          # [batch, n]

        # --- Step 3: correlated price innovations ---
        rho_perp = torch.sqrt(1.0 - rho ** 2)
        dB = rho * dW + rho_perp * torch.sqrt(dt) * Z_price  # [batch, n]

        # --- Step 4: log-Euler price path (vectorised, no loop) ---
        sqrt_V = torch.sqrt(V_hedge)
        log_inc = -0.5 * V_hedge * dt + sqrt_V * dB          # [batch, n]

        log_S = torch.cumsum(log_inc, dim=-1)                 # [batch, n]
        log_S = torch.cat([
            torch.zeros(batch, 1, dtype=Z_vol.dtype, device=Z_vol.device),
            log_S,
        ], dim=-1)                                             # [batch, n+1]

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
        dev = device or self.volterra.dt.device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev).manual_seed(seed)
        n = self.n_steps
        Z_vol = torch.randn(n_paths, n, 2, dtype=torch.float64, device=dev, generator=gen)
        Z_price = torch.randn(n_paths, n, dtype=torch.float64, device=dev, generator=gen)
        S, V = self.forward(Z_vol, Z_price, S0)
        return S, V, self.t_grid
