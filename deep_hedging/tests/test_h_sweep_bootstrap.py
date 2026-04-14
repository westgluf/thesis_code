"""Tests for H-sweep bootstrap analysis."""
from __future__ import annotations

import sys
from typing import Tuple

import numpy as np

from deep_hedging.experiments.h_sweep_analysis import (
    bootstrap_power_law_slope,
    compute_slope_noise_floor,
)


def test_bootstrap_slope_is_reproducible() -> Tuple[bool, str]:
    """Same inputs + same seed -> bit-identical bootstrap result."""
    H = np.array([0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.49])
    rng = np.random.default_rng(42)
    gamma = 1.0 + rng.normal(0, 0.1, size=9)
    a = bootstrap_power_law_slope(H, gamma, n_bootstrap=200, seed=99)
    b = bootstrap_power_law_slope(H, gamma, n_bootstrap=200, seed=99)
    ok = (a["beta_ci_bootstrap_95"] == b["beta_ci_bootstrap_95"]
          and a["beta_hat"] == b["beta_hat"])
    return ok, f"reproducible={ok}"


def test_bootstrap_slope_near_zero_for_flat_gamma() -> Tuple[bool, str]:
    """If Gamma(H) is constant, slope should be near zero and CI should bracket zero."""
    H = np.array([0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.49])
    gamma = np.ones_like(H) * 1.0
    out = bootstrap_power_law_slope(H, gamma, n_bootstrap=1000, seed=2024)
    beta_ok = abs(out["beta_hat"]) < 1e-9
    lo, hi = out["beta_ci_bootstrap_95"]
    ci_ok = lo <= 0.0 <= hi
    return beta_ok and ci_ok, f"beta={out['beta_hat']:.6f}, CI=[{lo:.3f}, {hi:.3f}]"


def test_bootstrap_slope_recovers_known_power_law() -> Tuple[bool, str]:
    """If Gamma(H) = c * (1/2 - H)^0.5, fit should recover slope ~0.5."""
    H = np.array([0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.49])
    gamma = 2.0 * (0.5 - H) ** 0.5
    out = bootstrap_power_law_slope(H, gamma, n_bootstrap=1000, seed=2024)
    ok = abs(out["beta_hat"] - 0.5) < 1e-6
    return ok, f"beta_hat={out['beta_hat']:.6f} (expected 0.5)"


def test_noise_floor_detects_indistinguishable_slope() -> Tuple[bool, str]:
    """A tiny slope with large noise -> inside noise band."""
    H = np.array([0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.49])
    gamma = np.array([1.0 + 0.001 * i for i in range(9)])
    out = compute_slope_noise_floor(H, es_halfwidth_per_point=0.5, gamma_values=gamma)
    return out["beta_inside_noise_band"], f"inside={out['beta_inside_noise_band']}, floor={out['beta_noise_floor']:.4f}"


def test_noise_floor_detects_significant_slope() -> Tuple[bool, str]:
    """A large slope with small noise -> outside noise band."""
    H = np.array([0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.49])
    gamma = 2.0 * (0.5 - H) ** 1.0
    out = compute_slope_noise_floor(H, es_halfwidth_per_point=0.05, gamma_values=gamma)
    ok = not out["beta_inside_noise_band"]
    return ok, f"inside={out['beta_inside_noise_band']}, beta={out['beta_hat']:.3f}, floor={out['beta_noise_floor']:.4f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Bootstrap reproducible",           test_bootstrap_slope_is_reproducible),
        ("2. Flat Gamma -> slope near zero",     test_bootstrap_slope_near_zero_for_flat_gamma),
        ("3. Known power law recovery",          test_bootstrap_slope_recovers_known_power_law),
        ("4. Noise floor detects small slope",   test_noise_floor_detects_indistinguishable_slope),
        ("5. Noise floor detects large slope",   test_noise_floor_detects_significant_slope),
    ]

    print("=" * 65, flush=True)
    print(" H-Sweep Bootstrap — Tests", flush=True)
    print("=" * 65, flush=True)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as e:
            import traceback
            passed, msg = False, f"EXCEPTION: {e}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        print(f"         {msg}", flush=True)
        if not passed:
            all_passed = False

    print("-" * 65, flush=True)
    if all_passed:
        print(f" All {len(tests)} tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
