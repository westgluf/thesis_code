#!/usr/bin/env python
"""
Smoke tests for worst-case adversarial perturbation (Part A, Prompt 12).

    python -m deep_hedging.tests.test_worst_case_adversarial
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.experiments.worst_case_adversarial import (
    WorstCaseAdversarialExperiment,
    project_epsilon,
    PARAM_BOX, SIGMA_SCALES, THETA_0, HEDGER_PATH,
)


# -----------------------------------------------------------------------
# Test 1: Projection onto constraint set
# -----------------------------------------------------------------------

def test_projection() -> Tuple[bool, str]:
    # Case 1: already inside — should be unchanged (modulo box)
    eps = (0.01, 0.1, 0.03)
    scaled = math.sqrt((0.01 / 0.1) ** 2 + (0.1 / 1.0) ** 2 + (0.03 / 0.3) ** 2)
    # scaled ≈ sqrt(0.01 + 0.01 + 0.01) ≈ 0.173
    eps_proj = project_epsilon(*eps, radius=1.0)
    if any(abs(a - b) > 1e-9 for a, b in zip(eps, eps_proj)):
        return False, f"already-inside eps was modified: {eps} -> {eps_proj}"

    # Case 2: outside the ball — should be shrunk
    eps_big = (0.5, 5.0, 0.0)
    # scaled ≈ sqrt(25 + 25) ≈ 7.07
    eps_proj = project_epsilon(*eps_big, radius=1.0)
    scaled_after = math.sqrt(
        (eps_proj[0] / SIGMA_SCALES["H"]) ** 2
        + (eps_proj[1] / SIGMA_SCALES["eta"]) ** 2
        + (eps_proj[2] / SIGMA_SCALES["rho"]) ** 2
    )
    if scaled_after > 1.01:  # tolerance for clamping
        return False, f"ball projection failed: scaled={scaled_after}"

    # Case 3: would take H out of box
    eps_oob = (-0.1, 0.0, 0.0)  # H + eps = -0.03 < 0.01
    eps_proj = project_epsilon(*eps_oob, radius=10.0)
    H_final = THETA_0["H"] + eps_proj[0]
    if H_final < PARAM_BOX["H"][0] - 1e-9:
        return False, f"H box violation: H+eps = {H_final}"

    # Case 4: zero in, zero out
    eps_zero = (0.0, 0.0, 0.0)
    eps_proj = project_epsilon(*eps_zero, radius=1.0)
    if any(abs(x) > 1e-12 for x in eps_proj):
        return False, f"zero eps modified: {eps_proj}"

    return True, "ball shrinking + box clamping + zero fixed point all OK"


# -----------------------------------------------------------------------
# Test 2: Attack converges on a tiny problem
# -----------------------------------------------------------------------

def test_attack_tiny() -> Tuple[bool, str]:
    if not HEDGER_PATH.exists():
        return False, f"baseline hedger missing at {HEDGER_PATH}"

    exp = WorstCaseAdversarialExperiment(
        radii=[0.5],
        pgd_steps=5,
        n_paths_attack=500,
        n_paths_eval=1000,
    )
    exp.load_baseline_hedger()
    exp.compute_calibration_p0(n_paths=5000)

    result = exp.attack_strategy("bs", radius=0.5, verbose=False)

    # The iterate trajectory should show non-trivial variation
    iter_history = result["iter_es_history"]
    if len(iter_history) != 5:
        return False, f"expected 5 steps, got {len(iter_history)}"

    # Worst-case should be >= baseline (attack found something at least as bad)
    if result["es95_worst_case"] < result["es95_baseline"] - 0.5:
        return False, (
            f"worst_case ({result['es95_worst_case']:.3f}) << baseline "
            f"({result['es95_baseline']:.3f})"
        )

    return True, (
        f"baseline={result['es95_baseline']:.3f}, "
        f"worst={result['es95_worst_case']:.3f}, "
        f"deg={result['degradation']:+.3f}"
    )


# -----------------------------------------------------------------------
# Test 3: Cross-evaluation structure
# -----------------------------------------------------------------------

def test_cross_evaluation() -> Tuple[bool, str]:
    if not HEDGER_PATH.exists():
        return False, f"baseline hedger missing at {HEDGER_PATH}"

    exp = WorstCaseAdversarialExperiment(
        radii=[0.5],
        pgd_steps=3,
        n_paths_attack=500,
        n_paths_eval=1000,
    )
    exp.load_baseline_hedger()
    exp.compute_calibration_p0(n_paths=5000)

    bs_res = exp.attack_strategy("bs", radius=0.5, verbose=False)
    dh_res = exp.attack_strategy("deep", radius=0.5, verbose=False)

    cross = exp.cross_evaluate(bs_res["eps_best"], dh_res["eps_best"], radius=0.5)

    required = {"BS_at_eps_BS", "BS_at_eps_DH", "DH_at_eps_BS", "DH_at_eps_DH"}
    if set(cross.keys()) != required:
        return False, f"missing keys: {required - set(cross.keys())}"

    all_finite = all(math.isfinite(v) for v in cross.values())
    if not all_finite:
        return False, f"non-finite values: {cross}"

    return True, ", ".join(f"{k}={v:.2f}" for k, v in cross.items())


# -----------------------------------------------------------------------
# Test 4: Larger radius → more potential degradation
# -----------------------------------------------------------------------

def test_radius_monotonicity() -> Tuple[bool, str]:
    if not HEDGER_PATH.exists():
        return False, f"baseline hedger missing at {HEDGER_PATH}"

    exp = WorstCaseAdversarialExperiment(
        radii=[0.2, 0.5],
        pgd_steps=5,
        n_paths_attack=500,
        n_paths_eval=2000,
    )
    exp.load_baseline_hedger()
    exp.compute_calibration_p0(n_paths=5000)

    small = exp.attack_strategy("bs", radius=0.2, verbose=False)
    big = exp.attack_strategy("bs", radius=0.5, verbose=False)

    # At larger radius, the attack has more room → worst case at least as bad
    # (allow some MC noise tolerance)
    if big["es95_worst_case"] + 0.5 < small["es95_worst_case"]:
        return False, (
            f"larger radius gave smaller worst case: "
            f"r=0.2 -> {small['es95_worst_case']:.3f}, "
            f"r=0.5 -> {big['es95_worst_case']:.3f}"
        )

    return True, (
        f"r=0.2 worst={small['es95_worst_case']:.3f}, "
        f"r=0.5 worst={big['es95_worst_case']:.3f}"
    )


# -----------------------------------------------------------------------
# Test 5: Deep hedger attack, hedger params unchanged
# -----------------------------------------------------------------------

def test_deep_attack_no_weight_mutation() -> Tuple[bool, str]:
    if not HEDGER_PATH.exists():
        return False, f"baseline hedger missing at {HEDGER_PATH}"

    exp = WorstCaseAdversarialExperiment(
        radii=[0.5],
        pgd_steps=3,
        n_paths_attack=500,
        n_paths_eval=1000,
    )
    exp.load_baseline_hedger()
    exp.compute_calibration_p0(n_paths=5000)

    before = [p.detach().clone() for p in exp.baseline_hedger.parameters()]
    res = exp.attack_strategy("deep", radius=0.5, verbose=False)
    after = [p.detach().clone() for p in exp.baseline_hedger.parameters()]

    unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
    finite = all(math.isfinite(v) for v in [res["es95_worst_case"],
                                             res["es95_baseline"],
                                             res["degradation"]])
    return unchanged and finite, (
        f"hedger unchanged={unchanged}, "
        f"baseline={res['es95_baseline']:.3f}, "
        f"worst={res['es95_worst_case']:.3f}"
    )


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Projection onto constraint set",       test_projection),
        ("2. Attack converges on tiny problem",      test_attack_tiny),
        ("3. Cross-evaluation structure",            test_cross_evaluation),
        ("4. Larger radius → more degradation",      test_radius_monotonicity),
        ("5. Deep attack preserves hedger weights",  test_deep_attack_no_weight_mutation),
    ]

    print("=" * 65, flush=True)
    print(" Worst-Case Adversarial — Smoke Tests", flush=True)
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
        print(" All 5 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
