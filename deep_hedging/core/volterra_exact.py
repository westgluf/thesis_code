"""
Exact Cholesky simulator of fractional Brownian motion.

Constructs the full (n+1)×(n+1) fBm covariance matrix and factorises
it via Cholesky.  Complexity is O(n³) for the one-off factorisation
and O(n²) per batch of paths — prohibitive for training but adequate
for validation against the hybrid scheme at n ≤ 400.

All computations use float64 for numerical stability with small H.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

def _build_fbm_covariance(n_steps: int, T: float, H: float) -> Tensor:
    """Build the fBm covariance matrix on an equispaced grid.

    Σ_{i,j} = 0.5 * (t_i^{2H} + t_j^{2H} − |t_i − t_j|^{2H})

    where t_i = i·T/n, i = 0, 1, …, n.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    T : float
        Time horizon.
    H : float
        Hurst exponent.

    Returns
    -------
    Tensor [n_steps+1, n_steps+1]
        Symmetric positive semi-definite covariance matrix.
    """
    n = n_steps
    t = torch.linspace(0.0, T, n + 1, dtype=torch.float64)
    twoH = 2.0 * H

    # t_i^{2H} and t_j^{2H} via outer sum; |t_i - t_j|^{2H} via abs diff
    ti_2H = t ** twoH                                  # [n+1]
    diff = torch.abs(t.unsqueeze(1) - t.unsqueeze(0))  # [n+1, n+1]
    diff_2H = diff ** twoH                              # |t_i - t_j|^{2H}

    Sigma = 0.5 * (ti_2H.unsqueeze(1) + ti_2H.unsqueeze(0) - diff_2H)
    # Enforce Σ[0,:] = Σ[:,0] = 0 (fBM starts at zero)
    Sigma[0, :] = 0.0
    Sigma[:, 0] = 0.0

    return Sigma

class ExactCholeskyVolterra(nn.Module):
    """O(n³) exact Cholesky simulator of Volterra fBm.

    For validation use only; not intended for training because of
    O(n³) scaling. At n=100 this is tractable (~1 ms per batch of 1000).

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
        self.register_buffer("_n_steps_t", torch.tensor(n_steps, dtype=torch.long))
        self.register_buffer("_T", torch.tensor(T, dtype=torch.float64))
        self.register_buffer("_dt", torch.tensor(T / n_steps, dtype=torch.float64))

        Sigma = _build_fbm_covariance(n_steps, T, H)

        # Cholesky on the (n)×(n) sub-matrix (rows/cols 1..n),
        # since row/col 0 is identically zero.
        Sigma_inner = Sigma[1:, 1:]

        # Add small diagonal jitter for numerical safety
        jitter = 1e-12 * torch.eye(n_steps, dtype=torch.float64)
        L_inner = torch.linalg.cholesky(Sigma_inner + jitter)

        self.register_buffer("_L_inner", L_inner)   # [n, n]

    @property
    def H(self) -> Tensor:
        return self._H

    @property
    def dt(self) -> Tensor:
        return self._dt

    @property
    def n_steps(self) -> int:
        return int(self._n_steps_t.item())

    def forward(
        self,
        Z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Generate exact fBm paths from i.i.d. normals.

        Parameters
        ----------
        Z : Tensor [batch, n_steps]
            i.i.d. N(0,1) driving noise.

        Returns
        -------
        WH : Tensor [batch, n_steps + 1]
            fBm values (WH[:,0] = 0).
        dWH : Tensor [batch, n_steps]
            fBm increments WH[:,k+1] - WH[:,k].
        """
        batch, n = Z.shape
        assert n == self.n_steps, f"Expected Z.shape[1] == {self.n_steps}, got {n}"

        # WH[1:] = L_inner @ Z^T   → [batch, n]
        WH_inner = torch.matmul(Z, self._L_inner.T)  # [batch, n]

        # Prepend zero
        WH = torch.zeros(batch, n + 1, dtype=Z.dtype, device=Z.device)
        WH[:, 1:] = WH_inner

        # Increments
        dWH = WH[:, 1:] - WH[:, :-1]  # [batch, n]

        return WH, dWH

    def simulate(
        self,
        n_paths: int,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate paths with internally created noise.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        seed : int, optional
            Random seed.
        device : torch.device, optional
            Computation device.

        Returns
        -------
        WH : Tensor [n_paths, n_steps + 1]
        dWH : Tensor [n_paths, n_steps]
        """
        dev = device or self._dt.device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev).manual_seed(seed)
        Z = torch.randn(n_paths, self.n_steps, dtype=torch.float64,
                         device=dev, generator=gen)
        return self.forward(Z)
