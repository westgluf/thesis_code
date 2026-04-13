"""
Hybrid Volterra driver for fractional Brownian motion (Algorithm 6).

Implements the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017) for
discretising the Volterra integral representation of fBm.  The scheme
splits each grid-point value into a near-field (exact covariance) and a
far-field (FFT convolution), giving O(n log n) complexity per path.

All computations use float64 for numerical stability with small H.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _b_star(k: Tensor, a: Tensor) -> Tensor:
    """Optimal discretisation point for the TBSS kernel (Definition 5.8).

    b*(k, a) = ((k^{a+1} - (k-1)^{a+1}) / (a+1))^{1/a}

    Parameters
    ----------
    k : Tensor  — integer indices >= 2 (float64 for exponentiation)
    a : Tensor  — scalar, a = H - 0.5
    """
    a1 = a + 1.0
    return ((k ** a1 - (k - 1.0) ** a1) / a1) ** (1.0 / a)


def _build_cholesky_and_kernel(
    H: Tensor, n_steps: int, dt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Pre-compute the 2×2 Cholesky factor and far-field kernel G.

    Returns
    -------
    L : Tensor [2, 2]  — lower-triangular Cholesky of the near-field
        covariance matrix.
    G : Tensor [n_steps + 1]  — far-field kernel with G[0] = G[1] = 0.
    """
    a = H - 0.5

    # --- 2×2 near-field covariance (κ = 1) ---
    # C[0,0] = dt
    # C[0,1] = dt^{a+1} / (a+1)
    # C[1,1] = dt^{2a+1} / (2a+1)
    C00 = dt
    C01 = dt ** (a + 1.0) / (a + 1.0)
    C11 = dt ** (2.0 * a + 1.0) / (2.0 * a + 1.0)

    # --- Analytical Cholesky L such that C = L @ L^T ---
    L00 = torch.sqrt(C00)
    L10 = C01 / L00
    # Guard against floating-point noise making the argument negative
    L11 = torch.sqrt(torch.clamp(C11 - L10 * L10, min=1e-30))

    L = torch.zeros(2, 2, dtype=dt.dtype, device=dt.device)
    L[0, 0] = L00
    L[1, 0] = L10
    L[1, 1] = L11

    # --- Far-field kernel G[k] for k = 0 … n_steps ---
    # G[0] = G[1] = 0
    # For k >= 2:  G[k] = (b*(k, a) · dt)^a
    G = torch.zeros(n_steps + 1, dtype=dt.dtype, device=dt.device)
    if n_steps >= 2:
        k = torch.arange(2, n_steps + 1, dtype=dt.dtype, device=dt.device)
        G[2:] = (_b_star(k, a) * dt) ** a

    return L, G


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class HybridVolterraDriver(nn.Module):
    """Hybrid-scheme discretisation of Volterra fBm (Algorithm 6).

    Given i.i.d. standard-normal innovations Z of shape ``(batch, n_steps, 2)``,
    the driver produces the fractional Brownian motion values at each grid
    point and the underlying Brownian increments.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    H : float
        Hurst exponent (0 < H < 0.5 for rough volatility).

    Forward
    -------
    Z : Tensor [batch, n_steps, 2]
        i.i.d. N(0, 1) driving noise.
    Returns (WH, dW) with
        WH : Tensor [batch, n_steps + 1]  — fBm values at the grid.
        dW : Tensor [batch, n_steps]       — Brownian increments (Var = dt).
    """

    def __init__(self, n_steps: int, T: float, H: float) -> None:
        super().__init__()
        self.register_buffer("_H", torch.tensor(H, dtype=torch.float64))
        self.register_buffer("_n_steps", torch.tensor(n_steps, dtype=torch.long))
        self.register_buffer("_T", torch.tensor(T, dtype=torch.float64))
        self.register_buffer("_dt", torch.tensor(T / n_steps, dtype=torch.float64))

        # Pre-compute Cholesky and kernel (H is a buffer, not a Parameter)
        L, G = _build_cholesky_and_kernel(self._H, n_steps, self._dt)
        self.register_buffer("_L", L)
        self.register_buffer("_G", G)

    # ---- properties -------------------------------------------------------

    @property
    def H(self) -> Tensor:
        return self._H

    @property
    def dt(self) -> Tensor:
        return self._dt

    @property
    def n_steps(self) -> int:
        return int(self._n_steps.item())

    @property
    def T(self) -> float:
        return float(self._T.item())

    # ---- differentiability ------------------------------------------------

    def make_H_parameter(self) -> None:
        """Promote H from buffer to ``nn.Parameter`` for sensitivity analysis."""
        val = self._H.detach().clone()
        del self._H
        self._H = nn.Parameter(val)

    # ---- forward ----------------------------------------------------------

    def forward(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """Map i.i.d. normals to (WH, dW).

        Implements the hybrid scheme (Algorithm 6) with near-field exact
        integrals and far-field FFT convolution.

        Parameters
        ----------
        Z : Tensor [batch, n_steps, 2]

        Returns
        -------
        WH : Tensor [batch, n_steps + 1]
        dW : Tensor [batch, n_steps]
        """
        batch, n, two = Z.shape
        assert two == 2, f"Expected Z.shape[-1] == 2, got {two}"

        # If H is a Parameter, recompute L and G so gradients flow
        if isinstance(self._H, nn.Parameter):
            L, G = _build_cholesky_and_kernel(self._H, n, self._dt)
        else:
            L, G = self._L, self._G

        H = self._H
        a = H - 0.5

        # ------------------------------------------------------------------
        # Near-field: correlated pair via Cholesky reparametrisation
        # dW_k = L[0,0] · Z[:,k,0]                              (Var = dt)
        # Y1_near_k = L[1,0] · Z[:,k,0] + L[1,1] · Z[:,k,1]
        # ------------------------------------------------------------------
        dW = L[0, 0] * Z[:, :, 0]                          # [batch, n]

        Y1_inc = L[1, 0] * Z[:, :, 0] + L[1, 1] * Z[:, :, 1]  # [batch, n]

        # Y1 at grid points: Y1[:,0] = 0, Y1[:,k] = Y1_inc[:,k-1] for k>=1
        Y1 = torch.zeros(batch, n + 1, dtype=Z.dtype, device=Z.device)
        Y1[:, 1:] = Y1_inc  # Y1[:,k] for k = 1 … n

        # ------------------------------------------------------------------
        # Far-field: FFT convolution of kernel G with dW
        # Y2[k] = Σ_{j=2}^{k} G[j] · dW[k-j]   (causal, linear conv)
        # ------------------------------------------------------------------
        n_fft = _next_power_of_2(n + n + 1)

        # Pad G (length n+1) and dW (length n) to n_fft
        G_pad = torch.zeros(n_fft, dtype=Z.dtype, device=Z.device)
        G_pad[: n + 1] = G

        dW_pad = torch.zeros(batch, n_fft, dtype=Z.dtype, device=Z.device)
        dW_pad[:, :n] = dW

        G_hat = torch.fft.rfft(G_pad)              # [n_fft//2 + 1]
        dW_hat = torch.fft.rfft(dW_pad, dim=-1)    # [batch, n_fft//2 + 1]

        Y2_full = torch.fft.irfft(G_hat.unsqueeze(0) * dW_hat, n=n_fft, dim=-1)
        Y2 = Y2_full[:, : n + 1]                   # [batch, n + 1]

        # ------------------------------------------------------------------
        # Total: WH[k] = √(2a+1) · (Y1[k] + Y2[k]),  WH[0] = 0
        # ------------------------------------------------------------------
        scale = torch.sqrt(2.0 * a + 1.0)
        WH = scale * (Y1 + Y2)
        # Enforce WH[:,0] = 0 exactly (should already be from construction)
        WH[:, 0] = 0.0

        return WH, dW


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class FractionalBM(nn.Module):
    """Standalone fractional Brownian motion generator.

    Wraps :class:`HybridVolterraDriver`, generating the required i.i.d.
    normal innovations internally.

    Parameters
    ----------
    n_steps, T, H : same as :class:`HybridVolterraDriver`.
    """

    def __init__(self, n_steps: int, T: float, H: float) -> None:
        super().__init__()
        self.driver = HybridVolterraDriver(n_steps, T, H)

    def forward(
        self,
        n_paths: int,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Generate fBm paths.

        Returns
        -------
        WH : Tensor [n_paths, n_steps + 1]
        """
        dev = device or self.driver._dt.device
        Z = torch.randn(
            n_paths, self.driver.n_steps, 2,
            dtype=torch.float64, device=dev, generator=generator,
        )
        WH, _ = self.driver(Z)
        return WH
