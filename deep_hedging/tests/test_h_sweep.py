#!/usr/bin/env python
"""
Quick validation tests for the H-sweep experiment.

Run before the full sweep (< 5 min total):

    python -m deep_hedging.tests.test_h_sweep
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.experiments.h_sweep import HurstSweepExperiment

XI0 = 0.235 ** 2


# ---------------------------------------------------------------------------
# Test 1: rBergomi at H=0.5 — lognormal-like, roughness slope ≈ 1.0
# ---------------------------------------------------------------------------

def test_H_half() -> Tuple[bool, str]:
    """At H=0.5 the Volterra kernel is trivial → BM-driven stoch vol."""
    model = DifferentiableRoughBergomi(
        n_steps=200, T=1.0, H=0.5, eta=1.9, rho=-0.7, xi0=XI0,
    )
    S, V, _ = model.simulate(n_paths=50_000, S0=100.0, seed=42)

    # E[V_t] ≈ xi0
    mean_V = float(V.mean())
    v_ok = abs(mean_V / XI0 - 1.0) < 0.10

    # Roughness: variogram slope should be ≈ 2*0.5 = 1.0
    logV = torch.log(V.clamp(min=1e-30))
    dt = 1.0 / 200
    lags = [1, 2, 4, 8, 16]
    log_lags, log_vars = [], []
    for lag in lags:
        d = logV[:, lag:] - logV[:, :-lag]
        log_lags.append(math.log(lag * dt))
        log_vars.append(math.log(float((d ** 2).mean())))
    x = torch.tensor(log_lags, dtype=torch.float64)
    y = torch.tensor(log_vars, dtype=torch.float64)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    H_hat = slope / 2.0
    slope_ok = abs(H_hat - 0.5) < 0.15  # generous tolerance at boundary

    passed = v_ok and slope_ok
    return passed, f"E[V]={mean_V:.6f} (xi0={XI0:.6f}), H_hat={H_hat:.3f} (target 0.5)"


# ---------------------------------------------------------------------------
# Test 2: rBergomi at H=0.01 — ultra-rough, no NaN
# ---------------------------------------------------------------------------

def test_H_001() -> Tuple[bool, str]:
    """Ultra-rough paths: no NaN, high short-term variability."""
    model = DifferentiableRoughBergomi(
        n_steps=100, T=1.0, H=0.01, eta=1.9, rho=-0.7, xi0=XI0,
    )
    S, V, _ = model.simulate(n_paths=10_000, S0=100.0, seed=99)

    finite_S = bool(torch.isfinite(S).all())
    finite_V = bool(torch.isfinite(V).all())
    positive_S = bool((S > 0).all())

    # V should have high short-term variability
    logV = torch.log(V.clamp(min=1e-30))
    m2_lag1 = float((logV[:, 1:] - logV[:, :-1]).pow(2).mean())

    passed = finite_S and finite_V and positive_S
    return passed, (
        f"finite_S={finite_S}, finite_V={finite_V}, S>0={positive_S}, "
        f"m2(lag1)={m2_lag1:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Mini 3-value sweep
# ---------------------------------------------------------------------------

def test_mini_sweep() -> Tuple[bool, str]:
    """Run sweep for H in {0.1, 0.3, 0.5} with small data."""
    exp = HurstSweepExperiment(
        H_values=[0.1, 0.3, 0.5],
        n_train=10_000, n_val=2_000, n_test=5_000,
        epochs=30, patience=10,
    )
    results = exp.run_full_sweep()

    # Basic structure
    len_ok = len(results) == 3
    keys_ok = all(
        {"H", "gamma", "bs_metrics", "dh_metrics"}.issubset(r.keys())
        for r in results
    )
    nan_ok = all(
        math.isfinite(r["gamma"]) and math.isfinite(r["bs_metrics"]["es_95"])
        for r in results
    )

    # Directional: Gamma(0.5) < Gamma(0.1)
    g_by_H = {r["H"]: r["gamma"] for r in results}
    dir_ok = g_by_H[0.5] < g_by_H[0.1]

    passed = len_ok and keys_ok and nan_ok and dir_ok
    return passed, (
        f"Gamma(0.1)={g_by_H.get(0.1, float('nan')):+.3f}, "
        f"Gamma(0.3)={g_by_H.get(0.3, float('nan')):+.3f}, "
        f"Gamma(0.5)={g_by_H.get(0.5, float('nan')):+.3f}, "
        f"dir_ok={dir_ok}"
    )


# ---------------------------------------------------------------------------
# Test 4: Save/load JSON round-trip
# ---------------------------------------------------------------------------

def test_json_roundtrip() -> Tuple[bool, str]:
    """Save results to JSON and load back."""
    exp = HurstSweepExperiment(
        H_values=[0.3],
        n_train=5_000, n_val=1_000, n_test=2_000,
        epochs=5, patience=5,
    )
    results = exp.run_full_sweep()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    exp.save_results(results, tmp)

    with open(tmp) as f:
        loaded = json.load(f)
    tmp.unlink()

    ok = (
        len(loaded) == 1
        and abs(loaded[0]["H"] - 0.3) < 1e-6
        and abs(loaded[0]["gamma"] - results[0]["gamma"]) < 1e-6
    )
    return ok, f"H={loaded[0]['H']}, gamma={loaded[0]['gamma']:.4f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. rBergomi at H=0.5",       test_H_half),
        ("2. rBergomi at H=0.01",       test_H_001),
        ("3. Mini 3-value sweep",        test_mini_sweep),
        ("4. JSON save/load round-trip", test_json_roundtrip),
    ]

    print("=" * 60, flush=True)
    print(" H-Sweep — Quick Validation", flush=True)
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
        print(" All 4 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
