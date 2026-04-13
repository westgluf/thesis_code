#!/usr/bin/env python
"""
Smoke tests for adversarial robustness (Prompt 11).

    python -m deep_hedging.tests.test_adversarial_robustness
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.hedging.deep_hedger import DeepHedgerFNN
from deep_hedging.experiments.gradient_sensitivity import (
    compute_es_gradient_bs,
    compute_es_gradient_deep,
    gradient_sensitivity_bootstrap,
)
from deep_hedging.experiments.adversarial_robustness import (
    AdversarialRobustnessExperiment,
    THETA_0, EPS_H, EPS_ETA, EPS_RHO,
)


# -----------------------------------------------------------------------
# Test 1: Gradient sensitivity returns finite values
# -----------------------------------------------------------------------

def test_gradient_finite() -> Tuple[bool, str]:
    out = compute_es_gradient_bs(
        H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
        xi0=THETA_0["xi0"],
        n_steps=50, n_paths=5_000,
        seed=42,
    )
    finite = (math.isfinite(out["grad_H"])
              and math.isfinite(out["grad_eta"])
              and math.isfinite(out["grad_rho"])
              and math.isfinite(out["grad_l2_norm"])
              and out["grad_l2_norm"] > 0)
    return finite, (
        f"grad_H={out['grad_H']:+.4f}, grad_eta={out['grad_eta']:+.4f}, "
        f"grad_rho={out['grad_rho']:+.4f}, L2={out['grad_l2_norm']:.4f}"
    )


# -----------------------------------------------------------------------
# Test 2: Gradient sign convention (informational)
# -----------------------------------------------------------------------

def test_gradient_signs() -> Tuple[bool, str]:
    """Just record the signs — don't enforce a specific sign because the
    actual sign at a given (Theta, dataset) is empirical."""
    out = compute_es_gradient_bs(
        H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
        xi0=THETA_0["xi0"],
        n_steps=50, n_paths=5_000,
        seed=43,
    )
    signs = (
        f"sign(dH)={'+' if out['grad_H']>=0 else '-'}, "
        f"sign(deta)={'+' if out['grad_eta']>=0 else '-'}, "
        f"sign(drho)={'+' if out['grad_rho']>=0 else '-'}"
    )
    return True, signs


# -----------------------------------------------------------------------
# Test 3: Bootstrap reproducibility
# -----------------------------------------------------------------------

def test_bootstrap_reproducibility() -> Tuple[bool, str]:
    kwargs = dict(
        H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
        xi0=THETA_0["xi0"],
        n_steps=50, n_paths=2_000,
    )
    a = gradient_sensitivity_bootstrap(
        compute_es_gradient_bs, n_seeds=3, seed_base=100, **kwargs,
    )
    b = gradient_sensitivity_bootstrap(
        compute_es_gradient_bs, n_seeds=3, seed_base=100, **kwargs,
    )
    c = gradient_sensitivity_bootstrap(
        compute_es_gradient_bs, n_seeds=3, seed_base=999, **kwargs,
    )

    same_seed_eq = (
        abs(a["mean_grad_H"] - b["mean_grad_H"]) < 1e-9
        and abs(a["mean_grad_l2_norm"] - b["mean_grad_l2_norm"]) < 1e-9
    )
    diff_seed_diff = abs(a["mean_grad_l2_norm"] - c["mean_grad_l2_norm"]) > 1e-6

    passed = same_seed_eq and diff_seed_diff
    return passed, (
        f"same_seed_eq={same_seed_eq}, diff_seed_diff={diff_seed_diff}, "
        f"a_L2={a['mean_grad_l2_norm']:.4f}, c_L2={c['mean_grad_l2_norm']:.4f}"
    )


# -----------------------------------------------------------------------
# Test 4: Sweep at eps=0 recovers baseline
# -----------------------------------------------------------------------

def test_sweep_eps_zero() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        # Use a tiny untrained hedger; we don't care about quality
        exp = AdversarialRobustnessExperiment(
            figures_dir=tmp,
            n_train=200, n_val=50, n_test_per_perturbation=2000,
            n_grad_paths=2000, n_grad_seeds=1,
        )
        # Force-skip cache: replace HEDGER_PATH
        from deep_hedging.experiments import adversarial_robustness as mod
        original_path = mod.HEDGER_PATH
        mod.HEDGER_PATH = Path(tmp) / "hedger.pt"
        try:
            # Use a tiny untrained hedger directly to avoid the training cost
            hedger = DeepHedgerFNN(input_dim=4, hidden_dim=16, n_res_blocks=1)
            hedger.eval()
            exp.hedger = hedger
            exp.p0_calibration = 8.0  # arbitrary fixed p0

            result = exp.run_perturbation_sweep_single_axis(
                "H", [0.0], p0_option="fixed",
            )
        finally:
            mod.HEDGER_PATH = original_path

    deg_bs = result["degradation_bs"][0.0]
    deg_dh = result["degradation_deep"][0.0]
    passed = abs(deg_bs) < 1e-9 and abs(deg_dh) < 1e-9
    return passed, f"deg_bs={deg_bs:.4e}, deg_dh={deg_dh:.4e}"


# -----------------------------------------------------------------------
# Test 5: Sweep returns expected structure
# -----------------------------------------------------------------------

def test_sweep_structure() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exp = AdversarialRobustnessExperiment(
            figures_dir=tmp,
            n_train=200, n_val=50, n_test_per_perturbation=2000,
        )
        from deep_hedging.experiments import adversarial_robustness as mod
        original_path = mod.HEDGER_PATH
        mod.HEDGER_PATH = Path(tmp) / "hedger.pt"
        try:
            hedger = DeepHedgerFNN(input_dim=4, hidden_dim=16, n_res_blocks=1)
            hedger.eval()
            exp.hedger = hedger
            exp.p0_calibration = 8.0

            eps_grid = [-0.02, 0.0, 0.02]
            result = exp.run_perturbation_sweep_single_axis(
                "H", eps_grid, p0_option="fixed",
            )
        finally:
            mod.HEDGER_PATH = original_path

    n_bs = len(result["bs"])
    n_dh = len(result["deep"])
    if n_bs != 3 or n_dh != 3:
        return False, f"n_bs={n_bs}, n_dh={n_dh}"

    all_finite = all(
        math.isfinite(result["bs"][float(e)]["metrics"]["es_95"])
        and math.isfinite(result["deep"][float(e)]["metrics"]["es_95"])
        for e in eps_grid
    )
    return all_finite, f"3 cells per side, all_finite={all_finite}"


# -----------------------------------------------------------------------
# Test 6: Hedger weights NOT modified by sweep
# -----------------------------------------------------------------------

def test_hedger_unchanged_after_sweep() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exp = AdversarialRobustnessExperiment(
            figures_dir=tmp,
            n_train=200, n_val=50, n_test_per_perturbation=2000,
        )
        from deep_hedging.experiments import adversarial_robustness as mod
        original_path = mod.HEDGER_PATH
        mod.HEDGER_PATH = Path(tmp) / "hedger.pt"
        try:
            hedger = DeepHedgerFNN(input_dim=4, hidden_dim=16, n_res_blocks=1)
            hedger.eval()
            exp.hedger = hedger
            exp.p0_calibration = 8.0

            # Snapshot params
            params_before = [p.detach().clone() for p in hedger.parameters()]

            exp.run_perturbation_sweep_single_axis(
                "H", [-0.02, 0.0, 0.02], p0_option="fixed",
            )

            params_after = [p.detach().clone() for p in hedger.parameters()]
        finally:
            mod.HEDGER_PATH = original_path

    all_unchanged = all(
        torch.equal(a, b) for a, b in zip(params_before, params_after)
    )
    return all_unchanged, f"all parameters unchanged: {all_unchanged}"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Gradient finite",                test_gradient_finite),
        ("2. Gradient signs (informational)",  test_gradient_signs),
        ("3. Bootstrap reproducibility",       test_bootstrap_reproducibility),
        ("4. Sweep eps=0 baseline recovery",   test_sweep_eps_zero),
        ("5. Sweep returns valid structure",   test_sweep_structure),
        ("6. Hedger params unchanged by sweep",test_hedger_unchanged_after_sweep),
    ]

    print("=" * 65, flush=True)
    print(" Adversarial Robustness — Smoke Tests", flush=True)
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
        print(" All 6 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
