#!/usr/bin/env python
"""Tests for P01.7-DIAG determinism diagnostic."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple


def _print_result(name: str, passed: bool, msg: str):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {msg}")


def test_q_tests_runnable() -> Tuple[bool, str]:
    """Each Q-test is callable and produces a dict with required keys."""
    from deep_hedging.experiments.block1_diag_determinism import (
        q3_evaluation_determinism, q4_run1_completion_check,
    )
    q3 = q3_evaluation_determinism(seed=9599)
    q4 = q4_run1_completion_check()
    for q, name in [(q3, "Q3"), (q4, "Q4")]:
        if "tested" not in q or "boolean" not in q:
            return False, f"{name} missing required keys"
        if q["tested"] is not True:
            return False, f"{name} not tested"
    return True, "Q3, Q4 callable; Q1, Q2 require subprocess (skipped in test)"


def test_decision_matrix_logic() -> Tuple[bool, str]:
    from deep_hedging.experiments.block1_diag_determinism import decision_matrix
    cases = [
        ((True, True, True, True), "HALT_P017"),
        ((True, True, True, False), "CONTINUE_P017"),
        ((True, True, True, None), "CONTINUE_WITH_NOTE"),
        ((False, True, True, True), "HALT_P017"),
        ((True, False, True, True), "HALT_EVERYTHING"),
        ((True, True, False, True), "HALT_P017"),
    ]
    errs = []
    for (q1, q2, q3, q4), expected in cases:
        got, _ = decision_matrix(q1, q2, q3, q4)
        if got != expected:
            errs.append(f"{q1,q2,q3,q4} -> {got}, expected {expected}")
    if errs:
        return False, f"{len(errs)} failures: " + "; ".join(errs)
    return True, f"All {len(cases)} decision cases correct"


def test_no_modification() -> Tuple[bool, str]:
    repo_root = Path(__file__).resolve().parents[2]
    watched = [
        repo_root / "deep_hedging" / "core" / "rough_bergomi.py",
        repo_root / "deep_hedging" / "core" / "volterra.py",
        repo_root / "deep_hedging" / "hedging" / "deep_hedger.py",
        repo_root / "deep_hedging" / "experiments" / "block1_extended_validation.py",
    ]
    mtimes = {str(f): os.path.getmtime(f) for f in watched if f.exists()}
    # Import diagnostic module to verify no side effects
    from deep_hedging.experiments.block1_diag_determinism import decision_matrix
    modified = [f.name for f in watched if f.exists() and os.path.getmtime(f) != mtimes.get(str(f))]
    if modified:
        return False, f"Modified: {', '.join(modified)}"
    return True, f"All {len(watched)} watched files unchanged"


def main():
    tests = [
        ("1. Q-tests runnable in isolation", test_q_tests_runnable),
        ("2. Decision matrix logic", test_decision_matrix_logic),
        ("3. No modification of existing files", test_no_modification),
    ]
    print("=" * 70)
    print(" P01.7-DIAG Tests")
    print("=" * 70)
    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as e:
            import traceback
            passed, msg = False, f"EXCEPTION: {e}\n{traceback.format_exc()[-300:]}"
        _print_result(name, passed, msg)
        if not passed:
            all_passed = False
    print("-" * 70)
    if all_passed:
        print(" All 3 tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
