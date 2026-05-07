#!/usr/bin/env python
"""
Smoke tests for transfer learning (Part B, Prompt 12).

    python -m deep_hedging.tests.test_transfer_learning
"""
from __future__ import annotations

import copy
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.hedging.deep_hedger import DeepHedgerFNN
from deep_hedging.experiments.transfer_learning import (
    TransferLearningExperiment,
    _make_fresh_hedger,
)


def _make_exp_tiny() -> TransferLearningExperiment:
    """Create an experiment instance with tiny budgets for smoke testing."""
    return TransferLearningExperiment(
        n_ft_values=[0, 500],
        n_test=500,
        epochs_pretrain=3,
        epochs_finetune=3,
        epochs_scratch=3,
    )


# -----------------------------------------------------------------------
# Test 1: GBM pretraining produces a trained hedger
# -----------------------------------------------------------------------

def test_pretrain_gbm() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        from deep_hedging.experiments import transfer_learning as mod
        orig_gbm_path = mod.GBM_PATH
        mod.GBM_PATH = Path(tmp) / "gbm.pt"
        try:
            exp = _make_exp_tiny()
            out = exp.pretrain_on_gbm(n_train=2000, n_val=500, seed=42)
            hedger = out["hedger"]
            if out.get("history") is None:
                return False, "no history returned"
            tr = out["history"]["train_risk"]
            if not (len(tr) >= 1 and math.isfinite(tr[0])):
                return False, f"bad train risk: {tr}"
            # Basic forward sanity check
            import torch
            S = 100 * torch.exp(0.01 * torch.randn(10, 101, dtype=torch.float64))
            S[:, 0] = 100.0
            with torch.no_grad():
                deltas = hedger.hedge_paths(S, T=1.0, S0=100.0)
            valid = deltas.shape == (10, 100) and bool(
                (deltas >= 0).all() and (deltas <= 1).all()
            )
            return valid, f"train_risk[0]={tr[0]:.3f}, shape={tuple(deltas.shape)}"
        finally:
            mod.GBM_PATH = orig_gbm_path


# -----------------------------------------------------------------------
# Test 2: Fine-tune(n_ft=0) is a no-op
# -----------------------------------------------------------------------

def test_finetune_n_ft_zero_noop() -> Tuple[bool, str]:
    hedger = _make_fresh_hedger(seed=7)
    state_before = copy.deepcopy(hedger.state_dict())

    # Minimal data stub (no training happens at n_ft=0)
    data = {"S_ft_full": None, "S_val": None, "p0": 0.0, "S_test": None}
    exp = _make_exp_tiny()
    out = exp.fine_tune(pretrained_state=state_before, n_ft=0, data=data)

    state_after = out["hedger"].state_dict()

    unchanged = all(
        torch.equal(state_before[k], state_after[k]) for k in state_before
    )
    return unchanged and out.get("skipped_training", False), (
        f"no-op check: unchanged={unchanged}, skipped={out.get('skipped_training')}"
    )


# -----------------------------------------------------------------------
# Test 3: Fine-tune does not mutate pretrained state dict
# -----------------------------------------------------------------------

def test_finetune_preserves_pretrained() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        from deep_hedging.experiments import transfer_learning as mod
        orig_gbm_path = mod.GBM_PATH
        mod.GBM_PATH = Path(tmp) / "gbm.pt"
        try:
            exp = _make_exp_tiny()
            pre_out = exp.pretrain_on_gbm(n_train=1500, n_val=300, seed=42)
            pretrained_state = copy.deepcopy(pre_out["hedger"].state_dict())
            pretrained_copy = copy.deepcopy(pretrained_state)

            # Make data
            data = exp.prepare_rbergomi_data(seed=42)

            # Fine-tune with n_ft=500
            ft = exp.fine_tune(pretrained_state, n_ft=500, data=data)

            # Assert the ORIGINAL pretrained dict is unchanged
            unchanged = all(
                torch.equal(pretrained_copy[k], pretrained_state[k])
                for k in pretrained_copy
            )
            # And the fine-tuned hedger has (probably) different weights
            ft_state = ft["hedger"].state_dict()
            any_changed = any(
                not torch.equal(pretrained_copy[k], ft_state[k])
                for k in pretrained_copy
            )
            return unchanged and any_changed, (
                f"pretrained dict unchanged={unchanged}, "
                f"fine-tuned weights differ={any_changed}"
            )
        finally:
            mod.GBM_PATH = orig_gbm_path


# -----------------------------------------------------------------------
# Test 4: Train from scratch with small data runs without error
# -----------------------------------------------------------------------

def test_train_from_scratch_small() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        from deep_hedging.experiments import transfer_learning as mod
        orig_gbm_path = mod.GBM_PATH
        mod.GBM_PATH = Path(tmp) / "gbm.pt"
        try:
            exp = _make_exp_tiny()
            data = exp.prepare_rbergomi_data(seed=42)
            out = exp.train_from_scratch(n_ft=500, data=data)
            if out is None:
                return False, "got None result"
            tr = out["history"]["train_risk"]
            finite = all(math.isfinite(x) for x in tr)
            return finite, f"finite={finite}, len(tr)={len(tr)}, conv={out['converged']}"
        finally:
            mod.GBM_PATH = orig_gbm_path


# -----------------------------------------------------------------------
# Test 5: Deterministic evaluation
# -----------------------------------------------------------------------

def test_deterministic_eval() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        from deep_hedging.experiments import transfer_learning as mod
        orig_gbm_path = mod.GBM_PATH
        mod.GBM_PATH = Path(tmp) / "gbm.pt"
        try:
            exp = _make_exp_tiny()
            data = exp.prepare_rbergomi_data(seed=42)
            hedger = _make_fresh_hedger(seed=11)
            hedger.eval()

            m1 = exp.evaluate(hedger, data)
            m2 = exp.evaluate(hedger, data)
            es1 = m1["metrics"]["es_95"]
            es2 = m2["metrics"]["es_95"]
            return abs(es1 - es2) < 1e-9, f"es1={es1:.6f}, es2={es2:.6f}"
        finally:
            mod.GBM_PATH = orig_gbm_path


# -----------------------------------------------------------------------
# Test 6: Data prefixes are consistent
# -----------------------------------------------------------------------

def test_data_prefixes() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        from deep_hedging.experiments import transfer_learning as mod
        orig_gbm_path = mod.GBM_PATH
        mod.GBM_PATH = Path(tmp) / "gbm.pt"
        try:
            exp = TransferLearningExperiment(
                n_ft_values=[0, 500, 2000], n_test=500,
            )
            data = exp.prepare_rbergomi_data(seed=42)

            # S_ft_full should have shape (max_n_ft, n_steps+1)
            max_n_ft = max(exp.n_ft_values)
            S_ft = data["S_ft_full"]
            if S_ft.shape != (max_n_ft, 101):
                return False, f"S_ft_full shape {tuple(S_ft.shape)}"

            # Prefix consistency: taking prefixes gives sub-datasets
            prefix_500 = S_ft[:500]
            prefix_2000 = S_ft[:2000]
            if not torch.equal(prefix_500, prefix_2000[:500]):
                return False, "prefix 500 not consistent with prefix 2000"

            return True, f"S_ft={tuple(S_ft.shape)}, prefix consistent"
        finally:
            mod.GBM_PATH = orig_gbm_path


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. GBM pretraining works",                  test_pretrain_gbm),
        ("2. fine_tune(n_ft=0) is a no-op",            test_finetune_n_ft_zero_noop),
        ("3. fine-tune preserves pretrained state",    test_finetune_preserves_pretrained),
        ("4. train_from_scratch with small data",      test_train_from_scratch_small),
        ("5. Evaluation is deterministic",             test_deterministic_eval),
        ("6. Data prefixes consistent",                test_data_prefixes),
    ]

    print("=" * 65, flush=True)
    print(" Transfer Learning — Smoke Tests", flush=True)
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
