"""
Analytical delta hedging strategies (Section 4.1).

Provides Black-Scholes and Heston "plug-in" delta hedgers with a
fully vectorised, differentiable implementation so that gradients
can flow for adversarial experiments (Prompt 12).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF via torch.erf (differentiable)."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _bs_d1(S: Tensor, K: float, tau: Tensor, sigma: Tensor, r: float) -> Tensor:
    """Black-Scholes d1 with broadcasting.

    Args:
        S: spot prices (any shape).
        K: strike (scalar).
        tau: time-to-maturity (same shape as S or broadcastable).
        sigma: volatility (same shape as S or broadcastable).
        r: risk-free rate.
    """
    tau_safe = torch.clamp(tau, min=1e-8)
    sigma_sqrt_tau = sigma * torch.sqrt(tau_safe)
    return (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau_safe) / sigma_sqrt_tau


# ---------------------------------------------------------------------------
# Black-Scholes delta hedger
# ---------------------------------------------------------------------------

class BlackScholesDelta:
    """Analytical BS delta hedger (Section 4.1).

    Computes delta at each time step using a FIXED assumed volatility
    sigma_assumed.  When sigma_assumed differs from the true dynamics
    this is the model-misspecification channel central to Hypothesis H1.

    Parameters
    ----------
    sigma : float
        Assumed (constant) volatility for the BS formula.
    K : float
        Strike price.
    T : float
        Option maturity.
    r : float
        Risk-free rate (default 0).
    """

    def __init__(self, sigma: float, K: float, T: float, r: float = 0.0) -> None:
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r

    def compute_delta(self, t_k: Tensor, S_k: Tensor) -> Tensor:
        """BS delta at time t_k for spot S_k.

        Args:
            t_k: scalar or (batch,) current time.
            S_k: (batch,) current spot price.

        Returns:
            delta: (batch,) in [0, 1].
        """
        tau = self.T - t_k
        sigma_t = torch.tensor(self.sigma, dtype=S_k.dtype, device=S_k.device)
        d1 = _bs_d1(S_k, self.K, tau, sigma_t, self.r)
        delta = _normal_cdf(d1)
        return torch.clamp(delta, 0.0, 1.0)

    def hedge_paths(self, S: Tensor) -> Tensor:
        """Compute BS delta at every rebalancing time for full price paths.

        Fully vectorised — no loops over paths or time steps.

        Args:
            S: (batch, n_steps + 1) price paths.

        Returns:
            deltas: (batch, n_steps) hedge ratios.
        """
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        dt = self.T / n

        # Time grid at rebalancing points: t_0, t_1, ..., t_{n-1}
        t_k = torch.arange(n, dtype=S.dtype, device=S.device) * dt   # (n,)
        S_k = S[:, :-1]                                               # (batch, n)

        tau = self.T - t_k.unsqueeze(0)                                # (1, n) -> broadcast
        sigma_t = torch.tensor(self.sigma, dtype=S.dtype, device=S.device)
        d1 = _bs_d1(S_k, self.K, tau, sigma_t, self.r)
        delta = _normal_cdf(d1)
        return torch.clamp(delta, 0.0, 1.0)

    @staticmethod
    def bs_call_price(
        S0: float, K: float, T: float, sigma: float, r: float = 0.0,
    ) -> float:
        """Analytical BS European call price for setting initial capital p0.

        C = S0 Phi(d1) - K e^{-rT} Phi(d2)
        """
        if T <= 0:
            return max(S0 - K, 0.0)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        from math import erf
        phi = lambda x: 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
        return S0 * phi(d1) - K * math.exp(-r * T) * phi(d2)


# ---------------------------------------------------------------------------
# Heston plug-in delta hedger
# ---------------------------------------------------------------------------

class HestonDelta:
    """Heston "plug-in" delta hedger (Remark 4.5).

    Uses the BS delta formula with sigma replaced by sqrt(V_k) at
    each time step.  This is a Markovian rule that OBSERVES variance
    (unlike BS which uses a fixed sigma) but is still misspecified
    under rough volatility because it ignores path dependence.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Option maturity.
    r : float
        Risk-free rate (default 0).
    """

    def __init__(self, K: float, T: float, r: float = 0.0) -> None:
        self.K = K
        self.T = T
        self.r = r

    def compute_delta(self, t_k: Tensor, S_k: Tensor, V_k: Tensor) -> Tensor:
        """Plug-in delta: BS delta with sigma = sqrt(V_k).

        Args:
            t_k: scalar or (batch,) current time.
            S_k: (batch,) current spot price.
            V_k: (batch,) current instantaneous variance.

        Returns:
            delta: (batch,) in [0, 1].
        """
        tau = self.T - t_k
        sigma_k = torch.sqrt(torch.clamp(V_k, min=1e-12))
        d1 = _bs_d1(S_k, self.K, tau, sigma_k, self.r)
        delta = _normal_cdf(d1)
        return torch.clamp(delta, 0.0, 1.0)

    def hedge_paths(self, S: Tensor, V: Tensor) -> Tensor:
        """Compute plug-in delta at every step using instantaneous variance.

        Fully vectorised — no loops.

        Args:
            S: (batch, n_steps + 1) price paths.
            V: (batch, n_steps + 1) variance paths.

        Returns:
            deltas: (batch, n_steps) hedge ratios.
        """
        batch, n_plus_1 = S.shape
        n = n_plus_1 - 1
        dt = self.T / n

        t_k = torch.arange(n, dtype=S.dtype, device=S.device) * dt   # (n,)
        S_k = S[:, :-1]                                               # (batch, n)
        V_k = V[:, :-1]                                               # (batch, n)

        tau = self.T - t_k.unsqueeze(0)                                # (1, n)
        sigma_k = torch.sqrt(torch.clamp(V_k, min=1e-12))             # (batch, n)
        d1 = _bs_d1(S_k, self.K, tau, sigma_k, self.r)
        delta = _normal_cdf(d1)
        return torch.clamp(delta, 0.0, 1.0)
