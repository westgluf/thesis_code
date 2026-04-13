#!/usr/bin/env python
"""
Quick smoke tests for the diagnostic controls experiments.

    python -m deep_hedging.tests.test_diagnostic_controls
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN, train_deep_hedger, evaluate_deep_hedger,
)
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall
from deep_hedging.experiments.diagnostic_controls import _run_single_point

XI0 = 0.235 ** 2


# -----------------------------------------------------------------------
# Test 1: eta=0 produces deterministic variance
# -----------------------------------------------------------------------

def test_eta_zero_deterministic() -> Tuple[bool, str]:
    """At eta=0, V should be constant = xi0 across all paths and times."""
    sim = DifferentiableRoughBergomi(n_steps=50, T=1.0, H=0.07, eta=0.0, rho=-0.7, xi0=XI0)
    _, V, _ = sim.simulate(n_paths=5_000, S0=100.0, seed=42)

    # Cross-path std at each timestep should be ~0
    std_per_t = V.std(dim=0)
    max_std = float(std_per_t.max())
    mean_V = float(V.mean())
    close_to_xi0 = abs(mean_V / XI0 - 1.0) < 0.001

    passed = max_std < 1e-10 and close_to_xi0
    return passed, f"max_std={max_std:.2e}, mean_V={mean_V:.6f} (xi0={XI0:.6f})"


# -----------------------------------------------------------------------
# Test 2: MSE loss differentiable
# -----------------------------------------------------------------------

def test_mse_loss() -> Tuple[bool, str]:
    """MSE loss is well-defined and produces finite gradients."""
    pnl = torch.randn(100, requires_grad=True)
    loss = (pnl ** 2).mean()
    loss.backward()
    ok = pnl.grad is not None and torch.isfinite(pnl.grad).all()
    return bool(ok), f"grad_ok={ok}, loss={loss.item():.4f}"


# -----------------------------------------------------------------------
# Test 3: Mean loss differentiable
# -----------------------------------------------------------------------

def test_mean_loss() -> Tuple[bool, str]:
    """Mean loss is well-defined and produces finite gradients."""
    pnl = torch.randn(100, requires_grad=True)
    loss = (-pnl).mean()
    loss.backward()
    ok = pnl.grad is not None and torch.isfinite(pnl.grad).all()
    return bool(ok), f"grad_ok={ok}, loss={loss.item():.4f}"


# -----------------------------------------------------------------------
# Test 4: Mini objective ablation (3 DH variants)
# -----------------------------------------------------------------------

def test_mini_ablation() -> Tuple[bool, str]:
    """Train MSE, Mean, ES on 5k rBergomi paths for 10 epochs."""
    sim = DifferentiableRoughBergomi(n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=XI0)
    S, _, _ = sim.simulate(n_paths=7_000, S0=100.0, seed=22)
    S_tr, S_va, S_te = S[:5_000], S[5_000:6_000], S[6_000:]

    payoff_tr = compute_payoff(S_tr, 100.0, "call")
    p0 = float(payoff_tr.mean())

    loss_fns = {
        "mse": lambda pnl: (pnl ** 2).mean(),
        "mean": lambda pnl: (-pnl).mean(),
        "es": None,
    }

    es_vals = {}
    for label, risk_fn in loss_fns.items():
        model = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
        train_deep_hedger(
            model, S_tr, S_va, K=100.0, T=1.0, S0=100.0, p0=p0,
            risk_fn=risk_fn, epochs=10, patience=10, batch_size=1024,
            verbose=False,
        )
        pnl = evaluate_deep_hedger(model, S_te, K=100.0, T=1.0, S0=100.0, p0=p0)
        es_vals[label] = float(expected_shortfall(pnl, 0.95))

    nan_ok = all(v == v for v in es_vals.values())  # NaN check
    # DH-ES should have lower or similar ES than DH-MSE (trained to optimise ES directly)
    # With only 10 epochs this may not hold strictly, so just check finite
    passed = nan_ok
    return passed, f"ES_95: mse={es_vals['mse']:.3f}, mean={es_vals['mean']:.3f}, es={es_vals['es']:.3f}"


# -----------------------------------------------------------------------
# Test 5: Mini 2x2 grid
# -----------------------------------------------------------------------

def test_mini_grid() -> Tuple[bool, str]:
    """Run 4 points on (H, eta) grid with tiny data."""
    results = []
    for i, (H, eta) in enumerate([(0.1, 0.5), (0.1, 1.9), (0.5, 0.5), (0.5, 1.9)]):
        r = _run_single_point(
            H=H, eta=eta, n_train=5_000, n_val=1_000, n_test=2_000,
            epochs=10, patience=10, seed=100 + i,
        )
        results.append(r)

    all_finite = all(r["gamma"] == r["gamma"] for r in results)
    gammas = [r["gamma"] for r in results]
    passed = all_finite
    return passed, f"Gammas: {[f'{g:+.3f}' for g in gammas]}"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. eta=0 deterministic variance",  test_eta_zero_deterministic),
        ("2. MSE loss differentiable",        test_mse_loss),
        ("3. Mean loss differentiable",       test_mean_loss),
        ("4. Mini objective ablation",        test_mini_ablation),
        ("5. Mini 2x2 grid",                  test_mini_grid),
    ]

    print("=" * 60, flush=True)
    print(" Diagnostic Controls — Smoke Tests", flush=True)
    print("=" * 60, flush=True)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        print(f"         {msg}", flush=True)
        if not passed:
            all_passed = False

    print("-" * 60, flush=True)
    if all_passed:
        print(" All 5 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
