#!/usr/bin/env python
"""
Smoke tests for the lean H4 sweep.

All tests must pass in < 3 minutes before launching the 2.5-hour
production sweep.

    python -m deep_hedging.tests.test_lean_h4_sweep
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.experiments.run_lean_h4_sweep import (
    FIXED_BUDGET, NEW_H_VALUES, STAGE_1_JSON, OUT_DIR,
    generate_paths, run_bs_delta, run_single_H,
    save_single_H_result, load_stage_1_results, compute_trend_verdict,
)


# -----------------------------------------------------------------------
# Test 1: Stage 1 loader returns correct structure
# -----------------------------------------------------------------------

def test_stage_1_loader() -> Tuple[bool, str]:
    """Load Stage 1 JSON, check structure and key fields."""
    if not STAGE_1_JSON.exists():
        return False, f"Stage 1 file not found at {STAGE_1_JSON}"

    result = load_stage_1_results(STAGE_1_JSON)

    expected_keys = {
        "H", "p0",
        "bs_metrics", "flat_metrics", "sig3_metrics", "sigfull_metrics",
        "gamma_flat", "gamma_sig3", "gamma_sigfull", "roughness_adv",
    }
    missing = expected_keys - set(result.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    # Check Stage 1 values (from the approved run)
    expected_es_95 = {
        "bs_metrics": 11.595,
        "flat_metrics": 10.478,
        "sig3_metrics": 10.776,
        "sigfull_metrics": 10.508,
    }
    for key, expected in expected_es_95.items():
        actual = result[key]["es_95"]
        if abs(actual - expected) > 0.01:
            return False, f"{key} es_95: expected {expected}, got {actual}"

    # Check derived Gamma
    if abs(result["gamma_flat"] - 1.117) > 0.01:
        return False, f"gamma_flat: expected ~1.117, got {result['gamma_flat']}"
    if abs(result["roughness_adv"] - (-0.029)) > 0.01:
        return False, f"roughness_adv: expected ~-0.029, got {result['roughness_adv']}"

    return True, (
        f"H={result['H']}, "
        f"Gamma_flat={result['gamma_flat']:+.3f}, "
        f"Gamma_sigfull={result['gamma_sigfull']:+.3f}, "
        f"rough_adv={result['roughness_adv']:+.3f}"
    )


# -----------------------------------------------------------------------
# Test 2: Budget is immutable
# -----------------------------------------------------------------------

def test_budget_immutable() -> Tuple[bool, str]:
    """Verify FIXED_BUDGET values and absence of argparse override."""
    # Expected values
    expected = {
        "n_train": 80_000, "n_val": 20_000, "n_test": 50_000,
        "epochs": 200, "patience": 30, "batch_size": 2048,
        "lr": 1e-3, "hidden_dim": 128, "n_res_blocks": 2,
        "weight_decay": 1e-5,
    }
    for k, v in expected.items():
        if FIXED_BUDGET.get(k) != v:
            return False, f"FIXED_BUDGET[{k!r}] = {FIXED_BUDGET.get(k)}, expected {v}"

    # Verify the sweep script has no argparse for budget fields
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_lean_h4_sweep.py"
    src = script_path.read_text()
    forbidden_flags = ["--n-train", "--n_train", "--epochs", "--batch-size", "--lr"]
    for flag in forbidden_flags:
        if flag in src:
            return False, f"Found forbidden CLI flag {flag!r} in script"

    return True, f"n_train={FIXED_BUDGET['n_train']}, epochs={FIXED_BUDGET['epochs']}, no CLI overrides"


# -----------------------------------------------------------------------
# Test 3: Tiny run_single_H smoke test
# -----------------------------------------------------------------------

def test_tiny_run_single_H() -> Tuple[bool, str]:
    """Run single H with a tiny budget override; check structure."""
    tiny_budget = {
        "n_train": 1500,
        "n_val": 300,
        "n_test": 700,
        "epochs": 4,
        "patience": 4,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_dim": 32,
        "n_res_blocks": 1,
        "weight_decay": 1e-5,
    }
    result = run_single_H(
        H=0.2, seed=99, device=torch.device("cpu"), budget=tiny_budget,
    )

    required = {
        "H", "p0", "bs_metrics", "flat_metrics", "sig3_metrics", "sigfull_metrics",
        "bs_pnl", "flat_pnl", "sig3_pnl", "sigfull_pnl",
        "gamma_flat", "gamma_sig3", "gamma_sigfull", "roughness_adv",
    }
    missing = required - set(result.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    # Finiteness
    for k in ["gamma_flat", "gamma_sig3", "gamma_sigfull", "roughness_adv"]:
        if not math.isfinite(result[k]):
            return False, f"{k} is not finite: {result[k]}"

    # PnL tensor shapes
    if result["bs_pnl"].shape[0] != tiny_budget["n_test"]:
        return False, f"bs_pnl shape mismatch"
    if result["flat_pnl"].shape[0] != tiny_budget["n_test"]:
        return False, f"flat_pnl shape mismatch"

    return True, (
        f"gamma_flat={result['gamma_flat']:+.3f}, "
        f"gamma_sigfull={result['gamma_sigfull']:+.3f}, "
        f"rough_adv={result['roughness_adv']:+.3f}"
    )


# -----------------------------------------------------------------------
# Test 4: Save / load round trip
# -----------------------------------------------------------------------

def test_save_load_roundtrip() -> Tuple[bool, str]:
    """Save a fake per-H result and reload via JSON."""
    fake = {
        "H": 0.1, "p0": 7.5,
        "bs_metrics": {"es_95": 11.0, "mean_pnl": 0.0, "std_pnl": 4.0},
        "flat_metrics": {"es_95": 10.1, "mean_pnl": 0.0, "std_pnl": 3.9},
        "sig3_metrics": {"es_95": 10.3, "mean_pnl": 0.0, "std_pnl": 3.9},
        "sigfull_metrics": {"es_95": 10.2, "mean_pnl": 0.0, "std_pnl": 3.9},
        "bs_pnl": torch.randn(10),
        "flat_pnl": torch.randn(10),
        "sig3_pnl": torch.randn(10),
        "sigfull_pnl": torch.randn(10),
        "gamma_flat": 0.9,
        "gamma_sig3": 0.7,
        "gamma_sigfull": 0.8,
        "roughness_adv": -0.1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_single_H_result(fake, 0.1, tmp_path)

        json_file = tmp_path / "lean_h4_H0.10.json"
        pt_file = tmp_path / "lean_h4_H0.10_pnl.pt"
        if not json_file.exists() or not pt_file.exists():
            return False, f"Files not created: {list(tmp_path.iterdir())}"

        with open(json_file) as f:
            loaded = json.load(f)

        # Check round-trip on scalars
        if abs(loaded["gamma_flat"] - 0.9) > 1e-6:
            return False, f"gamma_flat mismatch: {loaded['gamma_flat']}"
        if abs(loaded["bs_metrics"]["es_95"] - 11.0) > 1e-6:
            return False, f"bs_metrics.es_95 mismatch"

        # PnL round-trip
        pnl_loaded = torch.load(pt_file, weights_only=True)
        if not torch.allclose(pnl_loaded["bs_pnl"], fake["bs_pnl"].float(), atol=1e-4):
            return False, "bs_pnl tensor round-trip failed"

    return True, f"json + pt files round-trip OK"


# -----------------------------------------------------------------------
# Test 5: Trend detection logic
# -----------------------------------------------------------------------

def test_trend_detection() -> Tuple[bool, str]:
    """Feed synthetic results and check verdict classification."""
    # Case 1: weak H4 support (positive advantage at small H)
    case1 = {
        0.02: {"roughness_adv": 0.5},
        0.05: {"roughness_adv": 0.3},
        0.25: {"roughness_adv": 0.1},
    }
    v1 = compute_trend_verdict(case1)
    if v1["verdict"] != "weak_h4_support":
        return False, f"Case 1 expected weak_h4_support, got {v1['verdict']}"

    # Case 2: H4 refuted (all near-zero or negative)
    case2 = {
        0.02: {"roughness_adv": -0.1},
        0.05: {"roughness_adv": -0.03},
        0.25: {"roughness_adv": 0.02},
    }
    v2 = compute_trend_verdict(case2)
    if v2["verdict"] != "h4_refuted":
        return False, f"Case 2 expected h4_refuted, got {v2['verdict']}"

    # Case 3: positive advantage only at large H (not H4-supportive)
    case3 = {
        0.02: {"roughness_adv": -0.1},
        0.05: {"roughness_adv": 0.0},
        0.25: {"roughness_adv": 0.4},
    }
    v3 = compute_trend_verdict(case3)
    if v3["verdict"] != "h4_refuted":
        return False, f"Case 3 expected h4_refuted, got {v3['verdict']}"

    return True, f"weak_h4={v1['verdict']}, refuted={v2['verdict']}, refuted={v3['verdict']}"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Stage 1 loader returns structure", test_stage_1_loader),
        ("2. Budget is immutable",               test_budget_immutable),
        ("3. Tiny run_single_H smoke",           test_tiny_run_single_H),
        ("4. Save / load round-trip",            test_save_load_roundtrip),
        ("5. Trend detection logic",             test_trend_detection),
    ]

    print("=" * 65, flush=True)
    print(" Lean H4 Sweep — Smoke Tests", flush=True)
    print("=" * 65, flush=True)

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

    print("-" * 65, flush=True)
    if all_passed:
        print(" All 5 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
