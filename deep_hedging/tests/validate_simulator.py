#!/usr/bin/env python
"""
Validation suite for the differentiable rBergomi / GBM / Heston simulators.

Six tests covering statistical correctness, roughness, correlation,
performance, gradient flow, and GBM sanity.  Run with:

    python -m deep_hedging.tests.validate_simulator
"""
from __future__ import annotations

import math
import sys
import time
from typing import Tuple

import torch

from deep_hedging.core.volterra import FractionalBM
from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.core.gbm import GBM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """Simple OLS slope of y on x (1-D tensors)."""
    x_m, y_m = x.mean(), y.mean()
    return float(((x - x_m) * (y - y_m)).sum() / ((x - x_m) ** 2).sum())


# ---------------------------------------------------------------------------
# Test 1: Forward-variance check  E[v_k] ≈ ξ₀
# ---------------------------------------------------------------------------

def test_forward_variance() -> Tuple[bool, str]:
    """Verify E[v_{t_k}] ≈ ξ₀ at several grid points (200 K paths)."""
    xi0 = 0.235 ** 2
    model = DifferentiableRoughBergomi(
        n_steps=100, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=xi0,
    )
    gen = torch.Generator().manual_seed(42)
    S, V, _ = model.simulate(n_paths=200_000, S0=100.0, seed=42)

    check_indices = [0, 10, 25, 50, 75, 100]
    V_mean = V.mean(dim=0)
    rel_errors = {k: abs(float(V_mean[k]) / xi0 - 1.0) for k in check_indices}
    max_err = max(rel_errors.values())
    details = ", ".join(f"k={k}: {e:.4f}" for k, e in rel_errors.items())

    passed = max_err < 0.05
    return passed, f"max rel_error={max_err:.4f} [{details}]"


# ---------------------------------------------------------------------------
# Test 2: Roughness scaling  — variogram regression → Ĥ
# ---------------------------------------------------------------------------

def test_roughness_scaling() -> Tuple[bool, str]:
    """Estimate H from the log-log variogram of fBm paths (100 K paths)."""
    H_true = 0.07
    n_steps = 200
    T = 1.0
    dt = T / n_steps

    fbm = FractionalBM(n_steps=n_steps, T=T, H=H_true)
    WH = fbm(n_paths=100_000, generator=torch.Generator().manual_seed(123))

    lags = [1, 2, 3, 5, 8, 10, 15, 20]
    log_lags: list[float] = []
    log_vars: list[float] = []
    for lag in lags:
        diffs = WH[:, lag:] - WH[:, :-lag]
        var = float((diffs ** 2).mean())
        log_lags.append(math.log(lag * dt))
        log_vars.append(math.log(max(var, 1e-30)))

    x = torch.tensor(log_lags, dtype=torch.float64)
    y = torch.tensor(log_vars, dtype=torch.float64)
    slope = _ols_slope(x, y)
    H_hat = slope / 2.0

    err = abs(H_hat - H_true)
    passed = err < 0.05
    return passed, f"H_hat={H_hat:.4f}, H_true={H_true}, |err|={err:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Correlation check  — corr(Δlog S, Δlog V) has correct sign
# ---------------------------------------------------------------------------

def test_correlation() -> Tuple[bool, str]:
    """Check that log-return correlation between S and V matches ρ sign."""
    model = DifferentiableRoughBergomi(
        n_steps=100, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2,
    )
    S, V, _ = model.simulate(n_paths=200_000, S0=100.0, seed=99)

    log_ret_S = torch.log(S[:, 1:] / S[:, :-1])
    log_ret_V = torch.log(V[:, 1:].clamp(min=1e-30) / V[:, :-1].clamp(min=1e-30))

    x = log_ret_S.flatten()
    y = log_ret_V.flatten()
    corr = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))

    passed = corr < 0  # ρ = −0.7 → negative correlation
    return passed, f"corr(Δlog S, Δlog V) = {corr:.4f} (expected < 0)"


# ---------------------------------------------------------------------------
# Test 4: Timing benchmark
# ---------------------------------------------------------------------------

def test_timing() -> Tuple[bool, str]:
    """Time 10 K-path simulation (5 repeats), report paths/sec."""
    model = DifferentiableRoughBergomi(
        n_steps=100, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2,
    )

    # warm-up
    model.simulate(n_paths=1_000, S0=100.0, seed=0)

    n_paths = 10_000
    repeats = 5
    start = time.perf_counter()
    for i in range(repeats):
        model.simulate(n_paths=n_paths, S0=100.0, seed=i)
    elapsed = time.perf_counter() - start

    avg = elapsed / repeats
    rate = n_paths / avg
    passed = True  # informational — always passes
    return passed, f"{n_paths} paths in {avg:.3f}s ({rate:.0f} paths/s)"


# ---------------------------------------------------------------------------
# Test 5: Gradient flow  (Proposition 6.1)
# ---------------------------------------------------------------------------

def test_gradient_flow() -> Tuple[bool, str]:
    """Verify ∂L/∂η and ∂L/∂ρ exist and are finite after backward()."""
    model = DifferentiableRoughBergomi(
        n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2,
    )
    model.make_params_differentiable()

    gen = torch.Generator().manual_seed(77)
    Z_vol = torch.randn(500, 50, 2, dtype=torch.float64, generator=gen)
    Z_price = torch.randn(500, 50, dtype=torch.float64, generator=gen)

    S, V = model(Z_vol, Z_price, S0=100.0)

    K = 100.0
    loss = torch.relu(S[:, -1] - K).mean()  # call payoff proxy
    loss.backward()

    eta_grad = model._eta.grad
    rho_grad = model._rho.grad

    eta_ok = eta_grad is not None and torch.isfinite(eta_grad).all().item()
    rho_ok = rho_grad is not None and torch.isfinite(rho_grad).all().item()
    s_ok = (S > 0).all().item() and torch.isfinite(S).all().item()
    v_ok = torch.isfinite(V).all().item()

    passed = eta_ok and rho_ok and s_ok and v_ok
    details = (
        f"∂L/∂η={eta_grad}, ∂L/∂ρ={rho_grad}, "
        f"S>0={s_ok}, finite(V)={v_ok}"
    )
    return passed, details


# ---------------------------------------------------------------------------
# Test 6: GBM sanity  — E[S_T] ≈ S₀, Var[log(S_T/S₀)] ≈ σ²T
# ---------------------------------------------------------------------------

def test_gbm_sanity() -> Tuple[bool, str]:
    """Basic moment checks for 200 K GBM paths."""
    sigma = 0.235
    S0 = 100.0
    T = 1.0
    model = GBM(n_steps=100, T=T, sigma=sigma)
    S, V, _ = model.simulate(n_paths=200_000, S0=S0, seed=321)

    mean_ST = float(S[:, -1].mean())
    log_ret = torch.log(S[:, -1] / S[:, 0])
    var_log = float(log_ret.var())

    target_var = sigma ** 2 * T
    mean_ok = abs(mean_ST / S0 - 1.0) < 0.01
    var_ok = abs(var_log - target_var) < 0.005

    passed = mean_ok and var_ok
    return passed, (
        f"E[S_T]={mean_ST:.2f} (exp {S0}), "
        f"Var[log(S_T/S₀)]={var_log:.5f} (exp {target_var:.5f})"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Forward variance  E[v_k] ≈ ξ₀", test_forward_variance),
        ("2. Roughness scaling  Ĥ ≈ H",       test_roughness_scaling),
        ("3. Correlation sign check",          test_correlation),
        ("4. Timing benchmark",                test_timing),
        ("5. Gradient flow  (Prop. 6.1)",      test_gradient_flow),
        ("6. GBM sanity check",                test_gbm_sanity),
    ]

    print("=" * 70)
    print(" Differentiable rBergomi Simulator — Validation Suite")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            passed, msg = False, f"EXCEPTION: {exc}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(" All 6 validation tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
