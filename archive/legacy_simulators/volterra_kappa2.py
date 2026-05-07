"""
Hybrid Volterra driver with κ=2 (3×3 near-field Cholesky).

Extends the κ=1 hybrid scheme (Algorithm 6 of Bennedsen-Lunde-Pakkanen 2017)
by including TWO near-field neighbours in the exact covariance step,
yielding a 3×3 Cholesky factor and starting the far-field FFT at lag k=3.

This reduces near-field truncation error at the cost of a slightly larger
innovation vector (3 per step instead of 2).

Drop-in API-compatible with ``HybridVolterraDriver`` except that internal
noise Z has shape [batch, n_steps, 3].

All computations use float64 for numerical stability with small H.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .volterra import _next_power_of_2, _b_star


def _build_cholesky_and_kernel_kappa2(
    H: Tensor, n_steps: int, dt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Pre-compute the 3×3 Cholesky factor and far-field kernel G (κ=2).

    Near-field covariance matrix C (3×3) covers the increments at
    lags 0, 1, 2 relative to the current step.

    The entries are:

        C[i, j] = integral_0^1 integral_0^1 |s + i - t - j|^{2H-1} / Gamma(H+0.5)^2
                  ... simplified to power-function forms.

    For the BLP hybrid scheme, the 3 components per step are:
        Z0 → dW (Brownian increment, variance dt)
        Z1 → Y1 at lag 1 (near-field fBm contribution)
        Z2 → Y1 at lag 2 (near-field fBm contribution)

    The covariance structure uses a = H - 0.5:

        C[0,0] = dt                               (dW variance)
        C[0,1] = dt^{a+1} / (a+1)                (dW-Y1 cross-cov at lag 0-1)
        C[0,2] = ((2*dt)^{a+1} - dt^{a+1}) / (a+1)    (dW-Y1 at lag 0-2)
        C[1,1] = dt^{2a+1} / (2a+1)              (Y1 variance at lag 1)
        C[1,2] = ((2*dt)^{2a+1} - 2*dt^{2a+1}) / (2*(2a+1))  (Y1 cross lag 1-2)
        C[2,2] = dt^{2a+1} / (2a+1)              (Y1 variance at lag 2)

    Returns
    -------
    L : Tensor [3, 3]
    G : Tensor [n_steps + 1]  — far-field kernel with G[0] = G[1] = G[2] = 0.
    """
    a = H - 0.5
    a1 = a + 1.0
    two_a1 = 2.0 * a + 1.0

    # --- 3×3 near-field covariance ---
    C = torch.zeros(3, 3, dtype=dt.dtype, device=dt.device)

    # Diagonal entries
    C[0, 0] = dt
    C[1, 1] = dt ** two_a1 / two_a1
    C[2, 2] = dt ** two_a1 / two_a1

    # Off-diagonal: C[0,1] and C[0,2]
    C[0, 1] = dt ** a1 / a1
    C[1, 0] = C[0, 1]

    # C[0,2]: integral of kernel at lag 2
    # = integral_0^dt (s + dt)^a ds = ((2dt)^{a+1} - dt^{a+1}) / (a+1)
    C[0, 2] = ((2.0 * dt) ** a1 - dt ** a1) / a1
    C[2, 0] = C[0, 2]

    # C[1,2]: cross-covariance of fBm kernel integrals at lags 1 and 2
    # This is the integral: int_0^dt int_0^dt |s - t + dt|^{2a-1} * (2a+1) ds dt
    # After evaluation: ((2dt)^{2a+1} - 2*dt^{2a+1}) / (2*(2a+1))
    C[1, 2] = ((2.0 * dt) ** two_a1 - 2.0 * dt ** two_a1) / (2.0 * two_a1)
    C[2, 1] = C[1, 2]

    # --- Cholesky factorisation ---
    # Add small jitter for numerical safety
    jitter = 1e-30 * torch.eye(3, dtype=dt.dtype, device=dt.device)
    L = torch.linalg.cholesky(C + jitter)

    # --- Far-field kernel G[k] for k = 0 … n_steps ---
    # G[0] = G[1] = G[2] = 0  (these lags are handled by near-field)
    # For k >= 3: G[k] = (b*(k, a) · dt)^a
    G = torch.zeros(n_steps + 1, dtype=dt.dtype, device=dt.device)
    if n_steps >= 3:
        k = torch.arange(3, n_steps + 1, dtype=dt.dtype, device=dt.device)
        G[3:] = (_b_star(k, a) * dt) ** a

    return L, G


class HybridVolterraDriverKappa2(nn.Module):
    """Hybrid-scheme Volterra fBm with κ=2 (3×3 near-field).

    API-compatible with ``HybridVolterraDriver`` except that the
    internal noise vector has 3 components per step.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    H : float
        Hurst exponent.
    """

    def __init__(self, n_steps: int, T: float, H: float) -> None:
        super().__init__()
        self.register_buffer("_H", torch.tensor(H, dtype=torch.float64))
        self.register_buffer("_n_steps", torch.tensor(n_steps, dtype=torch.long))
        self.register_buffer("_T", torch.tensor(T, dtype=torch.float64))
        self.register_buffer("_dt", torch.tensor(T / n_steps, dtype=torch.float64))

        L, G = _build_cholesky_and_kernel_kappa2(self._H, n_steps, self._dt)
        self.register_buffer("_L", L)
        self.register_buffer("_G", G)

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

    def forward(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """Map i.i.d. normals to (WH, dW).

        Parameters
        ----------
        Z : Tensor [batch, n_steps, 3]

        Returns
        -------
        WH : Tensor [batch, n_steps + 1]
        dW : Tensor [batch, n_steps]
        """
        batch, n, three = Z.shape
        assert three == 3, f"κ=2 requires Z.shape[-1]==3, got {three}"

        L = self._L
        G = self._G
        H = self._H
        a = H - 0.5

        # --- Near-field: 3 correlated components via Cholesky ---
        # Component 0 → dW (Brownian increments)
        # Components 1,2 → Y1 at lags 1 and 2
        corr = torch.matmul(Z, L.T)  # [batch, n, 3]

        dW = corr[:, :, 0]          # [batch, n], Var = dt
        Y1_lag1 = corr[:, :, 1]     # [batch, n], near-field fBm at lag 1
        Y1_lag2 = corr[:, :, 2]     # [batch, n], near-field fBm at lag 2

        # Y1 at grid points:
        # Y1[k] = Y1_lag1[k-1] + Y1_lag2[k-2]
        # (each step contributes to the current and next grid point)
        Y1 = torch.zeros(batch, n + 1, dtype=Z.dtype, device=Z.device)
        Y1[:, 1:] += Y1_lag1        # lag-1 contribution to k=1..n
        if n >= 2:
            Y1[:, 2:] += Y1_lag2[:, :-1]  # lag-2 contribution to k=2..n

        # --- Far-field: FFT convolution of kernel G with dW ---
        n_fft = _next_power_of_2(n + n + 1)

        G_pad = torch.zeros(n_fft, dtype=Z.dtype, device=Z.device)
        G_pad[:n + 1] = G

        dW_pad = torch.zeros(batch, n_fft, dtype=Z.dtype, device=Z.device)
        dW_pad[:, :n] = dW

        G_hat = torch.fft.rfft(G_pad)
        dW_hat = torch.fft.rfft(dW_pad, dim=-1)
        Y2_full = torch.fft.irfft(G_hat.unsqueeze(0) * dW_hat, n=n_fft, dim=-1)
        Y2 = Y2_full[:, :n + 1]

        # --- Total ---
        scale = torch.sqrt(2.0 * a + 1.0)
        WH = scale * (Y1 + Y2)
        WH[:, 0] = 0.0

        return WH, dW
