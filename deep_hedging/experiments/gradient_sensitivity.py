"""
Gradient sensitivity: ∂ES_95 / ∂Θ for a fixed hedging strategy.

Leverages the differentiable rBergomi simulator (Prompt 1) to compute
parameter gradients via a single backward pass through the full
simulation + hedging pipeline.

This is, to our knowledge, the first time parameter gradients have been
computed directly through a rough volatility simulator. Existing rBergomi
codebases use NumPy and therefore cannot do this.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall


# =======================================================================
# Helpers
# =======================================================================

def _make_differentiable_simulator(
    H: float, eta: float, rho: float, xi0: float,
    n_steps: int, T: float,
) -> DifferentiableRoughBergomi:
    """Build an rBergomi simulator with H, eta, rho promoted to nn.Parameter."""
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    sim.volterra.make_H_parameter()
    sim.make_params_differentiable()
    return sim


def _generate_noise(
    n_paths: int, n_steps: int, seed: int, device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Reproducible noise tensors for the simulator."""
    g = torch.Generator(device=device).manual_seed(seed)
    Z_vol = torch.randn(n_paths, n_steps, 2, dtype=torch.float64, device=device, generator=g)
    Z_price = torch.randn(n_paths, n_steps, dtype=torch.float64, device=device, generator=g)
    return Z_vol, Z_price


def _compute_p0_at_theta(
    H: float, eta: float, rho: float, xi0: float,
    n_steps: int, T: float, S0: float, K: float,
    n_paths: int = 50_000, seed: int = 12345,
    device: torch.device | None = None,
) -> float:
    """MC initial capital from a non-differentiable simulator (no grad)."""
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    with torch.no_grad():
        S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed, device=device)
        payoff = compute_payoff(S, K, "call")
        return float(payoff.mean())


# =======================================================================
# BS-delta gradient
# =======================================================================

def compute_es_gradient_bs(
    H: float = 0.07,
    eta: float = 1.9,
    rho: float = -0.7,
    xi0: float = 0.235 ** 2,
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    n_steps: int = 100,
    n_paths: int = 50_000,
    sigma_assumed: float | None = None,
    alpha: float = 0.95,
    cost_lambda: float = 0.0,
    p0: float | None = None,
    seed: int = 2024,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compute ∂ES_α(PnL^{BS}) / ∂(H, η, ρ) at the supplied calibration point."""
    device = device or torch.device("cpu")
    if sigma_assumed is None:
        sigma_assumed = float(math.sqrt(xi0))

    sim = _make_differentiable_simulator(H, eta, rho, xi0, n_steps, T)
    Z_vol, Z_price = _generate_noise(n_paths, n_steps, seed, device)

    S, _ = sim(Z_vol, Z_price, S0=S0)

    bs = BlackScholesDelta(sigma=sigma_assumed, K=K, T=T)
    deltas = bs.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    if p0 is None:
        p0_val = float(payoff.detach().mean())
    else:
        p0_val = float(p0)

    pnl = compute_hedging_pnl(S, deltas, payoff, p0_val, cost_lambda)
    es = expected_shortfall(pnl, alpha)
    es.backward()

    grad_H = float(sim.volterra._H.grad)
    grad_eta = float(sim._eta.grad)
    grad_rho = float(sim._rho.grad)
    grad_l2 = float(np.sqrt(grad_H ** 2 + grad_eta ** 2 + grad_rho ** 2))

    return {
        "es_value": float(es.detach()),
        "p0": p0_val,
        "grad_H": grad_H,
        "grad_eta": grad_eta,
        "grad_rho": grad_rho,
        "grad_l2_norm": grad_l2,
        "n_paths": int(n_paths),
        "seed": int(seed),
    }


# =======================================================================
# Deep-hedger gradient
# =======================================================================

def compute_es_gradient_deep(
    hedger: DeepHedgerFNN,
    H: float = 0.07,
    eta: float = 1.9,
    rho: float = -0.7,
    xi0: float = 0.235 ** 2,
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    n_steps: int = 100,
    n_paths: int = 50_000,
    alpha: float = 0.95,
    cost_lambda: float = 0.0,
    p0: float | None = None,
    seed: int = 2024,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Same as compute_es_gradient_bs but for a fixed trained deep hedger.

    The hedger weights are frozen (``requires_grad=False``); gradients
    flow only through the rBergomi parameters via S → features → network
    → deltas → PnL → ES.
    """
    device = device or torch.device("cpu")

    sim = _make_differentiable_simulator(H, eta, rho, xi0, n_steps, T)
    Z_vol, Z_price = _generate_noise(n_paths, n_steps, seed, device)

    # Freeze hedger
    for p in hedger.parameters():
        p.requires_grad_(False)
    hedger.eval()

    S, _ = sim(Z_vol, Z_price, S0=S0)

    deltas = hedger.hedge_paths(S, T=T, S0=S0)
    deltas = deltas.to(S.dtype)

    payoff = compute_payoff(S, K, "call")
    if p0 is None:
        p0_val = float(payoff.detach().mean())
    else:
        p0_val = float(p0)

    pnl = compute_hedging_pnl(S, deltas, payoff, p0_val, cost_lambda)
    es = expected_shortfall(pnl, alpha)
    es.backward()

    grad_H = float(sim.volterra._H.grad)
    grad_eta = float(sim._eta.grad)
    grad_rho = float(sim._rho.grad)
    grad_l2 = float(np.sqrt(grad_H ** 2 + grad_eta ** 2 + grad_rho ** 2))

    return {
        "es_value": float(es.detach()),
        "p0": p0_val,
        "grad_H": grad_H,
        "grad_eta": grad_eta,
        "grad_rho": grad_rho,
        "grad_l2_norm": grad_l2,
        "n_paths": int(n_paths),
        "seed": int(seed),
    }


# =======================================================================
# Bootstrap
# =======================================================================

def gradient_sensitivity_bootstrap(
    compute_fn: Callable,
    n_seeds: int = 5,
    seed_base: int = 2024,
    **kwargs,
) -> dict[str, Any]:
    """Run ``compute_fn`` for n_seeds different seeds and report mean ± std."""
    samples: dict[str, list[float]] = {
        "grad_H": [], "grad_eta": [], "grad_rho": [],
        "grad_l2_norm": [], "es_value": [],
    }
    seeds_used: list[int] = []
    for i in range(n_seeds):
        seed = seed_base + 100 * i
        seeds_used.append(seed)
        out = compute_fn(seed=seed, **kwargs)
        for k in samples:
            samples[k].append(float(out[k]))

    out: dict[str, Any] = {"seeds": seeds_used, "n_seeds": int(n_seeds)}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        out[f"mean_{k}"] = float(arr.mean())
        out[f"std_{k}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[f"samples_{k}"] = vals
    return out
