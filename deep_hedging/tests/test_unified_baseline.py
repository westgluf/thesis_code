"""Smoke tests for the unified Section 6.3 baseline runner.

Full runs are too expensive for CI; these tests exercise the helpers
with miniature dimensions to catch regressions in the plumbing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.experiments import run_unified_baseline as rub


# ---------------------------------------------------------------------------
# Test 1: Constants frozen
# ---------------------------------------------------------------------------

def test_constants_are_frozen() -> Tuple[bool, str]:
    """Regression test: frozen calibration must not drift."""
    checks = [
        rub.H == 0.07,
        rub.ETA == 1.9,
        rub.RHO == -0.7,
        abs(rub.XI0 - 0.235 ** 2) < 1e-12,
        rub.MASTER_TEST_SEED == 2024,
        rub.N_TEST == 50_000,
        rub.TRAIN_SEED != rub.MASTER_TEST_SEED,
    ]
    passed = all(checks)
    return passed, f"All frozen constants correct: {passed}"


# ---------------------------------------------------------------------------
# Test 2: Seed independence
# ---------------------------------------------------------------------------

def test_frozen_calibration_seed_independence() -> Tuple[bool, str]:
    """Training and test seeds must not coincide."""
    passed = rub.TRAIN_SEED != rub.MASTER_TEST_SEED
    return passed, f"TRAIN_SEED={rub.TRAIN_SEED} != MASTER_TEST_SEED={rub.MASTER_TEST_SEED}: {passed}"


# ---------------------------------------------------------------------------
# Test 3: Master test set determinism
# ---------------------------------------------------------------------------

def test_master_test_set_deterministic(tmp_path: Path) -> Tuple[bool, str]:
    """Same seed + same params -> identical paths, bit-for-bit."""
    orig_path = rub.MASTER_TEST_SET_PATH
    orig_n = rub.N_TEST
    try:
        rub.MASTER_TEST_SET_PATH = tmp_path / "t.pt"
        rub.N_TEST = 32  # tiny for speed

        # First generation
        S_a, V_a = rub.load_or_generate_master_test_set()
        # Second call (should hit cache)
        S_b, V_b = rub.load_or_generate_master_test_set()
        match = bool(torch.equal(S_a, S_b) and torch.equal(V_a, V_b))
        return match, f"Deterministic cache: {match}, S shape={S_a.shape}"
    finally:
        rub.MASTER_TEST_SET_PATH = orig_path
        rub.N_TEST = orig_n


# ---------------------------------------------------------------------------
# Test 4: Output schema (mini end-to-end)
# ---------------------------------------------------------------------------

def test_output_schema(tmp_path: Path) -> Tuple[bool, str]:
    """Output JSON has required keys and all strategies."""
    orig = {
        "N_TEST": rub.N_TEST,
        "N_TRAIN": rub.N_TRAIN,
        "N_VAL": rub.N_VAL,
        "DH_EPOCHS": rub.DH_EPOCHS,
        "DH_PATIENCE": rub.DH_PATIENCE,
        "OUTPUT_JSON": rub.OUTPUT_JSON,
        "MASTER_TEST_SET_PATH": rub.MASTER_TEST_SET_PATH,
        "DH_CHECKPOINT_PATH": rub.DH_CHECKPOINT_PATH,
        "GBM_PRETRAINED_PATH": rub.GBM_PRETRAINED_PATH,
    }
    try:
        rub.N_TEST = 256
        rub.N_TRAIN = 512
        rub.N_VAL = 128
        rub.DH_EPOCHS = 2
        rub.DH_PATIENCE = 2
        rub.OUTPUT_JSON = tmp_path / "out.json"
        rub.MASTER_TEST_SET_PATH = tmp_path / "testset.pt"
        rub.DH_CHECKPOINT_PATH = tmp_path / "ckpt.pt"
        rub.GBM_PRETRAINED_PATH = tmp_path / "does_not_exist.pt"

        rub.main(allow_missing_pretrained=True)

        with open(tmp_path / "out.json") as f:
            out = json.load(f)

        # Check meta keys
        meta_keys = {"master_test_seed", "n_test", "rbergomi",
                     "dh_protocol", "p0_mc_estimate", "source_script"}
        meta_ok = meta_keys <= set(out["meta"])

        # Check results structure
        results_ok = True
        for cost_key in ("0.0", "0.001"):
            if cost_key not in out["results"]:
                results_ok = False
                break
            row = out["results"][cost_key]
            required = {"BS Delta", "Plug-in Delta", "DH full-budget"}
            if not required <= set(row):
                results_ok = False
                break
            for strat_name, strat in row.items():
                if strat.get("unavailable"):
                    continue
                if "metrics" not in strat or "mean_turnover" not in strat:
                    results_ok = False
                    break
                for key in ("mean_pnl", "std_pnl", "es_95", "es_99", "var_95"):
                    if key not in strat["metrics"]:
                        results_ok = False
                        break

        passed = meta_ok and results_ok
        return passed, f"meta_ok={meta_ok}, results_ok={results_ok}"
    finally:
        for k, v in orig.items():
            setattr(rub, k, v)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    import tempfile

    tests = [
        ("1. Constants frozen",              test_constants_are_frozen),
        ("2. Seed independence",             test_frozen_calibration_seed_independence),
        ("3. Master test set determinism",   None),
        ("4. Output schema (mini e2e)",      None),
    ]

    print("=" * 70)
    print(" Unified Baseline — Smoke Tests")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            if fn is not None:
                passed, msg = fn()
            else:
                # Tests needing tmp_path
                with tempfile.TemporaryDirectory() as td:
                    tp = Path(td)
                    if "determinism" in name:
                        passed, msg = test_master_test_set_deterministic(tp)
                    else:
                        passed, msg = test_output_schema(tp)
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(f" All {len(tests)} tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
