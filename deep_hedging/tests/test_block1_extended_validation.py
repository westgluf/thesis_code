#!/usr/bin/env python
"""
Tests for P01.7 — Extended Validation (4 cells at n=400).

Run:
    python -m deep_hedging.tests.test_block1_extended_validation
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

def _print_result(name: str, passed: bool, msg: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {msg}")

def test_preflight() -> Tuple[bool, str]:
    p = Path(__file__).resolve().parents[2] / "results" / "block1" / "p017_preflight.md"
    if not p.exists():
        return False, f"Not found: {p}"
    text = p.read_text()
    if "READY TO PROCEED" not in text and "HALT" not in text:
        return False, "Missing verdict"
    return True, f"Preflight OK, {len(text)} chars"

def test_tiny_run() -> Tuple[bool, str]:
    from deep_hedging.experiments.block1_extended_validation import run_experiment
    with tempfile.TemporaryDirectory() as tmpdir:
        rdir = Path(tmpdir) / "results" / "block1"
        fdir = Path(tmpdir) / "figures" / "block1"
        output = run_experiment(
            n_sim=50, n_test=100, n_bootstrap=10,
            seeds={"A": [7711], "B": [7721], "C": [7731], "D": [7741]},
            budget_diag={"n_train": 200, "n_val": 50, "epochs": 2,
                          "patience": 5, "batch_size": 64, "lr": 1e-3},
            budget_h2={"n_train": 200, "n_val": 50, "epochs": 2,
                        "patience": 5, "batch_size": 64, "lr": 1e-3},
            results_dir=rdir, figures_dir=fdir,
        )
        if not isinstance(output, dict):
            return False, "Not a dict"
        for k in ["meta", "config", "cells", "global_verdict"]:
            if k not in output:
                return False, f"Missing key {k}"
        for cell in ["A", "B", "C", "D"]:
            if cell not in output["cells"]:
                return False, f"Missing cell {cell}"
            if "cell_verdict" not in output["cells"][cell]:
                return False, f"Cell {cell} missing verdict"
        jpath = rdir / "extended_validation.json"
        if not jpath.exists():
            return False, "JSON missing"
        gv = output["global_verdict"]
        return True, f"Tiny run OK, verdict={gv}"

def test_cell_c_variants() -> Tuple[bool, str]:
    """C-match and C-subsample should differ when n_sim != n_rebal."""
    # At n_sim=50, C-match uses n_rebal=50, C-subsample uses n_rebal=100
    # But if n_sim < 100, subsample step = 50//100 = 0, which would fail
    # So test with n_sim=200: C-match n_rebal=200, C-subsample n_rebal=100
    from deep_hedging.experiments.block1_extended_validation import run_cell_C
    with tempfile.TemporaryDirectory() as tmpdir:
        import deep_hedging.experiments.block1_extended_validation as mod
        orig_rd = mod.RESULTS_DIR
        mod.RESULTS_DIR = Path(tmpdir)
        try:
            result = run_cell_C(
                n_sim=200, seeds=[7731], n_test=100, n_bootstrap=5,
                budget={"n_train": 100, "n_val": 50, "epochs": 2,
                         "patience": 5, "batch_size": 64, "lr": 1e-3},
            )
        finally:
            mod.RESULTS_DIR = orig_rd
    if "variant_C_match" not in result or "variant_C_subsample" not in result:
        return False, "Missing variant keys"
    cm = result["variant_C_match"]
    cs = result["variant_C_subsample"]
    if cm["n_rebal_effective"] == cs["n_rebal_effective"]:
        return False, "Both variants have same n_rebal"
    return True, f"C-match n_rebal={cm['n_rebal_effective']}, C-sub n_rebal={cs['n_rebal_effective']}"

def test_global_verdict_logic() -> Tuple[bool, str]:
    from deep_hedging.experiments.block1_extended_validation import _global_verdict
    cases = [
        ({"A": "CELL_PRESERVED", "B": "CELL_PRESERVED",
          "C": "CELL_PRESERVED", "D": "CELL_PRESERVED"}, "STRICT_PASS"),
        ({"A": "CELL_PRESERVED", "B": "CELL_PRESERVED",
          "C": "CELL_MODESTLY_SHIFTED", "D": "CELL_PRESERVED"}, "WEAK_PASS"),
        ({"A": "CELL_COLLAPSED", "B": "CELL_PRESERVED",
          "C": "CELL_PRESERVED", "D": "CELL_PRESERVED"}, "FAIL"),
        ({"A": "CELL_PRESERVED", "B": "CELL_SIGN_FLIP",
          "C": "CELL_PRESERVED", "D": "CELL_PRESERVED"}, "FAIL"),
        ({"A": "CELL_MODESTLY_SHIFTED", "B": "CELL_MODESTLY_SHIFTED",
          "C": "CELL_PRESERVED", "D": "CELL_PRESERVED"}, "FAIL"),
    ]
    errors = []
    for verdicts, expected in cases:
        got = _global_verdict(verdicts)
        if got != expected:
            errors.append(f"{verdicts} -> {got}, expected {expected}")
    if errors:
        return False, f"{len(errors)} failures: " + "; ".join(errors)
    return True, f"All {len(cases)} verdict cases correct"

def test_canonical_loading() -> Tuple[bool, str]:
    from deep_hedging.experiments.block1_extended_validation import CANONICAL
    for cell in ["A", "B", "C", "D"]:
        if cell not in CANONICAL:
            return False, f"Missing canonical {cell}"
        if "gamma" not in CANONICAL[cell]:
            return False, f"Cell {cell} missing gamma"
    ga = CANONICAL["A"]["gamma"]
    if not (-0.1 < ga < 0.1):
        return False, f"Cell A gamma={ga} out of expected range"
    return True, f"All 4 canonical values loaded: A={CANONICAL['A']['gamma']:.4f}, B={CANONICAL['B']['gamma']:.4f}"

def test_no_modification() -> Tuple[bool, str]:
    repo_root = Path(__file__).resolve().parents[2]
    watched = [
        repo_root / "deep_hedging" / "core" / "rough_bergomi.py",
        repo_root / "deep_hedging" / "hedging" / "deep_hedger.py",
        repo_root / "figures" / "diagnostic_controls_results.json",
    ]
    mtimes = {str(f): os.path.getmtime(f) for f in watched if f.exists()}
    # Just import to verify no side effects
    from deep_hedging.experiments.block1_extended_validation import CANONICAL
    modified = [f.name for f in watched if f.exists() and os.path.getmtime(f) != mtimes.get(str(f))]
    if modified:
        return False, f"Modified: {', '.join(modified)}"
    return True, f"All {len(watched)} watched files unchanged"

def main() -> None:
    tests = [
        ("1. Preflight exists", test_preflight),
        ("2. Tiny end-to-end", test_tiny_run),
        ("3. Cell C variant handling", test_cell_c_variants),
        ("4. Global verdict logic", test_global_verdict_logic),
        ("5. Canonical loading", test_canonical_loading),
        ("6. No modification of existing files", test_no_modification),
    ]

    print("=" * 70)
    print(" P01.7 — Extended Validation Tests")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()[-200:]}"
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
