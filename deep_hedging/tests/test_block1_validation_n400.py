#!/usr/bin/env python
"""
Tests for P01.6 — Validation Cell at n=400.

Five tests: preflight, tiny end-to-end, bootstrap CI, verdict logic,
and no-modification of existing files.

Run:
    python -m deep_hedging.tests.test_block1_validation_n400
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

def _print_result(name: str, passed: bool, msg: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {msg}")

# ---------------------------------------------------------------------------
# Test 1: Preflight script runnable
# ---------------------------------------------------------------------------

def test_preflight() -> Tuple[bool, str]:
    """Verify the preflight report exists or can be generated."""
    preflight_path = Path(__file__).resolve().parents[2] / "results" / "block1" / "p016_preflight.md"
    if not preflight_path.exists():
        return False, f"Preflight report not found at {preflight_path}"
    text = preflight_path.read_text()
    if "READY TO PROCEED" not in text and "HALT" not in text:
        return False, "Preflight report missing verdict line"
    return True, f"Preflight exists with verdict, {len(text)} chars"

# ---------------------------------------------------------------------------
# Test 2: Tiny end-to-end
# ---------------------------------------------------------------------------

def test_tiny_run() -> Tuple[bool, str]:
    """Run with minimal config; verify all outputs."""
    from deep_hedging.experiments.block1_validation_n400 import run_experiment

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results" / "block1"
        figures_dir = Path(tmpdir) / "figures" / "block1"

        output = run_experiment(
            n_steps=100,          # Use n=100 for speed
            dh_seeds=[7401],      # Single seed
            n_train=200,
            n_val=50,
            n_test=500,
            epochs=3,
            patience=5,
            batch_size=128,
            bootstrap_draws=20,
            results_dir=results_dir,
            figures_dir=figures_dir,
        )

        if not isinstance(output, dict):
            return False, "run_experiment did not return a dict"

        # Check JSON
        json_path = results_dir / "validation_n400.json"
        if not json_path.exists():
            return False, "JSON not created"
        data = json.loads(json_path.read_text())

        required_keys = ["meta", "config", "canonical_n100", "n400_results", "gamma_analysis"]
        for k in required_keys:
            if k not in data:
                return False, f"Missing key '{k}' in JSON"

        # Check gamma_analysis
        ga = data["gamma_analysis"]
        for k in ["gamma_per_seed", "gamma_mean", "gamma_std",
                   "gamma_bootstrap_ci_95", "gamma_vs_baseline"]:
            if k not in ga:
                return False, f"Missing gamma_analysis key '{k}'"

        # Check verdict exists
        verdict = ga["gamma_vs_baseline"].get("verdict")
        valid_verdicts = {"GAMMA_PRESERVED", "GAMMA_MODESTLY_SHIFTED",
                          "GAMMA_COLLAPSED", "SIGN_FLIP"}
        if verdict not in valid_verdicts:
            return False, f"Invalid verdict: {verdict}"

        # Check figures
        for fig in ["gamma_n100_vs_n400.png", "pl_distributions_n400.png",
                     "per_seed_gamma_bars.png"]:
            fp = figures_dir / fig
            if not fp.exists():
                return False, f"Figure missing: {fig}"
            if fp.stat().st_size < 100:
                return False, f"Figure too small: {fig}"

        # Check README
        readme = results_dir / "validation_n400_readme.md"
        if not readme.exists():
            return False, "README not created"

        # Check checkpoint
        ckpt = results_dir / "checkpoints_n400" / "seed_7401.pt"
        if not ckpt.exists():
            return False, "Checkpoint not saved"

        return True, f"Tiny run OK: verdict={verdict}, JSON={json_path.stat().st_size}B"

# ---------------------------------------------------------------------------
# Test 3: Bootstrap CI correctness
# ---------------------------------------------------------------------------

def test_bootstrap_ci() -> Tuple[bool, str]:
    """Verify bootstrap CI contains known true Gamma on synthetic data."""
    from deep_hedging.experiments.block1_validation_n400 import bootstrap_gamma_ci

    rng = np.random.default_rng(42)
    n = 10_000

    # Create two PnL arrays with known tail behaviour
    pnl_bs = rng.normal(0, 5, size=n)  # BS P&L
    pnl_dh = rng.normal(0.5, 4, size=n)  # DH P&L (slightly better)

    # Compute "true" gamma on full sample
    k = max(1, int(np.ceil(n * 0.05)))
    es_bs_true = -np.sort(pnl_bs)[:k].mean()
    es_dh_true = -np.sort(pnl_dh)[:k].mean()
    gamma_true = es_bs_true - es_dh_true

    ci_lo, ci_hi = bootstrap_gamma_ci(pnl_bs, pnl_dh, n_draws=500, rng_seed=42)

    # CI should contain the point estimate (with reasonable tolerance)
    contained = ci_lo <= gamma_true <= ci_hi
    reasonable_width = 0 < (ci_hi - ci_lo) < 5.0  # Not too wide

    msg = (
        f"Gamma_true={gamma_true:.3f}, CI=[{ci_lo:.3f}, {ci_hi:.3f}], "
        f"contained={contained}, width={ci_hi-ci_lo:.3f}"
    )
    return contained and reasonable_width, msg

# ---------------------------------------------------------------------------
# Test 4: Verdict logic unit test
# ---------------------------------------------------------------------------

def test_verdict_logic() -> Tuple[bool, str]:
    """Test all four verdict regions with synthetic inputs."""
    from deep_hedging.experiments.block1_validation_n400 import compute_verdict

    cases = [
        # (gamma_n100, gamma_n400, ci_low, ci_high, expected_verdict)
        (1.17, 1.10, 0.8, 1.4, "GAMMA_PRESERVED"),       # <20%, canonical in CI
        (1.17, 0.80, 0.5, 1.1, "GAMMA_MODESTLY_SHIFTED"), # 20-50%, sign same
        (1.17, 0.30, 0.1, 0.5, "GAMMA_COLLAPSED"),        # >50%
        (1.17, -0.5, -0.8, -0.2, "SIGN_FLIP"),            # Sign flip
        (1.17, 0.05, -0.1, 0.2, "GAMMA_COLLAPSED"),       # Gamma < 2*CI_half_width
    ]

    errors = []
    for gamma100, gamma400, ci_lo, ci_hi, expected in cases:
        got = compute_verdict(gamma100, gamma400, ci_lo, ci_hi)
        if got != expected:
            errors.append(
                f"  compute_verdict({gamma100}, {gamma400}, [{ci_lo},{ci_hi}]) "
                f"= {got}, expected {expected}"
            )

    if errors:
        return False, f"{len(errors)} verdict mismatches:\n" + "\n".join(errors)
    return True, f"All {len(cases)} verdict cases correct"

# ---------------------------------------------------------------------------
# Test 5: No modification of existing files
# ---------------------------------------------------------------------------

def test_no_modification() -> Tuple[bool, str]:
    """Verify core files unchanged after tiny experiment."""
    repo_root = Path(__file__).resolve().parents[2]
    watched = [
        repo_root / "deep_hedging" / "core" / "volterra.py",
        repo_root / "deep_hedging" / "core" / "rough_bergomi.py",
        repo_root / "deep_hedging" / "hedging" / "deep_hedger.py",
        repo_root / "deep_hedging" / "objectives" / "pnl.py",
        repo_root / "deep_hedging" / "objectives" / "risk_measures.py",
        repo_root / "figures" / "unified_baseline_results.json",
        repo_root / "figures" / "section6_numbers.json",
    ]

    mtimes_before = {}
    for f in watched:
        if f.exists():
            mtimes_before[str(f)] = os.path.getmtime(f)

    # Run tiny experiment
    from deep_hedging.experiments.block1_validation_n400 import run_experiment
    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(
            n_steps=50,
            dh_seeds=[7401],
            n_train=100, n_val=50, n_test=100,
            epochs=2, patience=5, batch_size=64,
            bootstrap_draws=10,
            results_dir=Path(tmpdir) / "r",
            figures_dir=Path(tmpdir) / "f",
        )

    modified = []
    for f in watched:
        if f.exists() and str(f) in mtimes_before:
            if os.path.getmtime(f) != mtimes_before[str(f)]:
                modified.append(f.name)

    if modified:
        return False, f"Modified: {', '.join(modified)}"
    return True, f"All {len(watched)} watched files unchanged"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Preflight report exists", test_preflight),
        ("2. Tiny end-to-end run", test_tiny_run),
        ("3. Bootstrap CI correctness", test_bootstrap_ci),
        ("4. Verdict logic unit test", test_verdict_logic),
        ("5. No modification of existing files", test_no_modification),
    ]

    print("=" * 70)
    print(" P01.6 — Validation Cell Tests")
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
        print(" All 5 tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
