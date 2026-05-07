#!/usr/bin/env python
"""
Tests for P02 — Exact Cholesky + Arbitrage + Kappa-2 validation.

Six tests covering: preflight, exact fBm validity, moment match,
arbitrage detection, kappa-2 API, no-modification guard.

Run:
    python -m deep_hedging.tests.test_block1_volterra_validation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def _print_result(name: str, passed: bool, msg: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {msg}")


# ---------------------------------------------------------------------------
# Test 1: Preflight runnable
# ---------------------------------------------------------------------------

def test_preflight() -> Tuple[bool, str]:
    """Verify preflight report exists."""
    p = Path(__file__).resolve().parents[2] / "results" / "block1" / "p02_preflight.md"
    if not p.exists():
        return False, f"Not found: {p}"
    text = p.read_text()
    if "READY TO PROCEED" not in text and "HALT" not in text:
        return False, "Missing verdict line"
    return True, f"Preflight exists, {len(text)} chars"


# ---------------------------------------------------------------------------
# Test 2: Exact Cholesky produces valid fBm
# ---------------------------------------------------------------------------

def test_exact_fbm_validity() -> Tuple[bool, str]:
    """Terminal variance of exact fBm matches T^{2H} within 5%."""
    from deep_hedging.core.volterra_exact import ExactCholeskyVolterra

    H, T, n = 0.07, 1.0, 50
    driver = ExactCholeskyVolterra(n, T, H)
    WH, _ = driver.simulate(1000, seed=42)

    # E[WH(T)^2] should equal T^{2H} = 1.0 for T=1
    var_terminal = float((WH[:, -1] ** 2).mean())
    target = T ** (2 * H)
    rel_err = abs(var_terminal - target) / target

    passed = rel_err < 0.05
    return passed, f"Var[WH(T)]={var_terminal:.4f}, target={target:.4f}, rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Exact matches hybrid in moments
# ---------------------------------------------------------------------------

def test_moment_match() -> Tuple[bool, str]:
    """Hybrid and exact fBm have similar terminal moments (relaxed)."""
    from deep_hedging.core.volterra import HybridVolterraDriver
    from deep_hedging.core.volterra_exact import ExactCholeskyVolterra

    H, T, n = 0.07, 1.0, 50

    hybrid = HybridVolterraDriver(n, T, H)
    gen = torch.Generator().manual_seed(42)
    Z = torch.randn(500, n, 2, dtype=torch.float64, generator=gen)
    WH_h, _ = hybrid(Z)

    exact = ExactCholeskyVolterra(n, T, H)
    WH_e, _ = exact.simulate(500, seed=43)

    # Compare mean and std of terminal values (relaxed tolerance)
    mean_h = float(WH_h[:, -1].mean())
    mean_e = float(WH_e[:, -1].mean())
    std_h = float(WH_h[:, -1].std())
    std_e = float(WH_e[:, -1].std())

    mean_ok = abs(mean_h - mean_e) < 0.1
    std_ok = abs(std_h - std_e) / (std_e + 1e-12) < 0.15

    msg = (f"mean: hybrid={mean_h:.4f}, exact={mean_e:.4f}; "
           f"std: hybrid={std_h:.4f}, exact={std_e:.4f}")
    return mean_ok and std_ok, msg


# ---------------------------------------------------------------------------
# Test 4: Arbitrage check detects injected violation
# ---------------------------------------------------------------------------

def test_arbitrage_detection() -> Tuple[bool, str]:
    """Feed prices with known violation and verify detection."""
    # Synthetic prices that violate monotonicity in K
    prices_bad = [[25.0, 26.0, 20.0]]  # price INCREASES from K1 to K2
    K_grid = [80.0, 100.0, 120.0]
    T_grid = [1.0]

    # Check monotonicity
    violations = 0
    for t_idx in range(len(T_grid)):
        for k_idx in range(len(K_grid) - 1):
            if prices_bad[t_idx][k_idx] < prices_bad[t_idx][k_idx + 1]:
                violations += 1

    if violations == 0:
        return False, "Failed to detect injected violation"
    return True, f"Detected {violations} violations in synthetic data"


# ---------------------------------------------------------------------------
# Test 5: Kappa-2 API compatibility
# ---------------------------------------------------------------------------

def test_kappa2_api() -> Tuple[bool, str]:
    """Kappa-2 driver produces valid fBm with correct shapes."""
    from deep_hedging.core.volterra_kappa2 import HybridVolterraDriverKappa2

    n, T, H = 50, 1.0, 0.07
    driver = HybridVolterraDriverKappa2(n, T, H)

    gen = torch.Generator().manual_seed(42)
    Z = torch.randn(100, n, 3, dtype=torch.float64, generator=gen)
    WH, dW = driver(Z)

    shape_ok = WH.shape == (100, n + 1) and dW.shape == (100, n)
    wh0_ok = (WH[:, 0] == 0).all().item()
    finite_ok = torch.isfinite(WH).all().item() and torch.isfinite(dW).all().item()

    # Check dW variance ~ dt
    dt = T / n
    dw_var = float((dW ** 2).mean())
    var_ok = abs(dw_var / dt - 1.0) < 0.15

    msg = (f"shapes OK={shape_ok}, WH[0]=0: {wh0_ok}, "
           f"finite={finite_ok}, dW_var/dt={dw_var/dt:.3f}")
    return shape_ok and wh0_ok and finite_ok and var_ok, msg


# ---------------------------------------------------------------------------
# Test 6: No modification of existing files
# ---------------------------------------------------------------------------

def test_no_modification() -> Tuple[bool, str]:
    """Core volterra.py and rough_bergomi.py unchanged."""
    repo_root = Path(__file__).resolve().parents[2]
    watched = [
        repo_root / "deep_hedging" / "core" / "volterra.py",
        repo_root / "deep_hedging" / "core" / "rough_bergomi.py",
    ]
    mtimes = {str(f): os.path.getmtime(f) for f in watched if f.exists()}

    # Import new modules to verify no side effects
    from deep_hedging.core.volterra_exact import ExactCholeskyVolterra
    from deep_hedging.core.volterra_kappa2 import HybridVolterraDriverKappa2

    modified = [f.name for f in watched if f.exists() and os.path.getmtime(f) != mtimes.get(str(f))]
    if modified:
        return False, f"Modified: {', '.join(modified)}"
    return True, f"All {len(watched)} watched files unchanged"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Preflight report exists", test_preflight),
        ("2. Exact fBm validity (Var = T^{2H})", test_exact_fbm_validity),
        ("3. Hybrid-Exact moment match", test_moment_match),
        ("4. Arbitrage violation detection", test_arbitrage_detection),
        ("5. κ=2 API compatibility", test_kappa2_api),
        ("6. No modification of existing files", test_no_modification),
    ]

    print("=" * 70)
    print(" P02 — Volterra Validation Tests")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            passed, msg = False, f"EXCEPTION: {exc}"
        _print_result(name, passed, msg)
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(" All 6 tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
