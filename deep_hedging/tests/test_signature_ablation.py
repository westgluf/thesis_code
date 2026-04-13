#!/usr/bin/env python
"""
Smoke tests for the signature ablation and H-sweep experiments.

    python -m deep_hedging.tests.test_signature_ablation
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.experiments.signature_ablation import (
    SignatureAblationExperiment, TwoTowerHedger, StandardisedSignatureHedger,
)
from deep_hedging.experiments.signature_h_sweep import SignatureHSweepExperiment
from deep_hedging.hedging.signature_hedger import SignatureDeepHedger
from deep_hedging.hedging.deep_hedger import train_deep_hedger
from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.objectives.pnl import compute_payoff


# -----------------------------------------------------------------------
# Test 1: run_stage_1 returns proper structure
# -----------------------------------------------------------------------

def test_stage_1_structure() -> Tuple[bool, str]:
    """Tiny stage 1 run; check shapes/types only, not performance."""
    exp = SignatureAblationExperiment()
    r = exp.run_stage_1(
        H=0.1, n_train=3000, n_val=500, n_test=1000,
        epochs=8, seed=42,
    )
    expected = {"H", "p0", "bs", "flat", "sig3", "sigfull",
                "gamma_flat", "gamma_sig3", "gamma_sigfull",
                "gate_passed", "gate_threshold"}
    keys_ok = expected.issubset(r.keys())
    metrics_ok = "es_95" in r["bs"]["metrics"] and "es_95" in r["flat"]["metrics"]
    pnl_len_ok = (r["bs"]["pnl"].shape[0] == 1000
                  and r["flat"]["pnl"].shape[0] == 1000
                  and r["sigfull"]["pnl"].shape[0] == 1000)
    finite_ok = all(
        math.isfinite(r[k]) for k in ["gamma_flat", "gamma_sig3", "gamma_sigfull"]
    )
    passed = keys_ok and metrics_ok and pnl_len_ok and finite_ok
    return passed, (
        f"keys_ok={keys_ok}, metrics_ok={metrics_ok}, "
        f"pnl_len_ok={pnl_len_ok}, finite_ok={finite_ok}, "
        f"gate={r['gate_passed']}"
    )


# -----------------------------------------------------------------------
# Test 2: Permutation importance works
# -----------------------------------------------------------------------

def test_feature_importance() -> Tuple[bool, str]:
    """Train a tiny sig-full hedger; check permutation importance returns 12 features."""
    sim = DifferentiableRoughBergomi(n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235**2)
    S, _, _ = sim.simulate(n_paths=4000, S0=100.0, seed=11)
    S_tr, S_va, S_te = S[:3000], S[3000:3500], S[3500:]
    p0 = float(compute_payoff(S_tr, 100.0, "call").mean())

    hedger = SignatureDeepHedger(feature_set="sig-full", hidden_dim=32, n_res_blocks=1, T=1.0)
    train_deep_hedger(
        hedger, S_tr, S_va, K=100.0, T=1.0, S0=100.0, p0=p0,
        epochs=5, batch_size=512, patience=10, verbose=False,
    )

    exp = SignatureAblationExperiment()
    diag = exp._diag_feature_importance(hedger, S_te, p0)

    n_feats = len(diag["importances"])
    finite_all = all(math.isfinite(v) for v in diag["importances"].values())
    has_path_features = any(name in diag["importances"]
                            for name in ["rv5", "rv15", "R", "Q"])
    passed = n_feats == 12 and finite_all and has_path_features
    return passed, (
        f"n_feats={n_feats}, all_finite={finite_all}, "
        f"most_important={diag['most_important']}"
    )


# -----------------------------------------------------------------------
# Test 3: TwoTowerHedger trains
# -----------------------------------------------------------------------

def test_two_tower() -> Tuple[bool, str]:
    sim = DifferentiableRoughBergomi(n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235**2)
    S, _, _ = sim.simulate(n_paths=3000, S0=100.0, seed=22)
    S_tr, S_va = S[:2500], S[2500:]
    p0 = float(compute_payoff(S_tr, 100.0, "call").mean())

    hedger = TwoTowerHedger(n_flat=4, n_path=8, hidden_dim=32, n_res_blocks=1)
    history = train_deep_hedger(
        hedger, S_tr, S_va, K=100.0, T=1.0, S0=100.0, p0=p0,
        epochs=10, batch_size=512, patience=15, verbose=False,
    )
    tr = history["train_risk"]
    no_nan = all(x == x for x in tr)
    decreased = tr[-1] < tr[0] + 0.5  # tolerant check
    passed = no_nan and decreased
    return passed, f"risk: {tr[0]:.3f} -> {tr[-1]:.3f}, no_nan={no_nan}"


# -----------------------------------------------------------------------
# Test 4: SignatureHSweepExperiment.run_single_H works
# -----------------------------------------------------------------------

def test_h_sweep_single() -> Tuple[bool, str]:
    exp = SignatureHSweepExperiment(
        H_values=[0.1], n_train=3000, n_val=500, n_test=1000, epochs=8,
    )
    r = exp.run_single_H(H=0.1, seed=42)
    keys = {"bs_metrics", "flat_metrics", "sig3_metrics", "sigfull_metrics",
            "gamma_flat", "gamma_sig3", "gamma_sigfull", "roughness_advantage"}
    keys_ok = keys.issubset(r.keys())
    finite_ok = all(math.isfinite(r[k]) for k in
                    ["gamma_flat", "gamma_sig3", "gamma_sigfull", "roughness_advantage"])
    passed = keys_ok and finite_ok
    return passed, (
        f"gamma_flat={r['gamma_flat']:+.3f}, "
        f"gamma_sig3={r['gamma_sig3']:+.3f}, "
        f"gamma_sigfull={r['gamma_sigfull']:+.3f}, "
        f"roughness={r['roughness_advantage']:+.3f}"
    )


# -----------------------------------------------------------------------
# Test 5: JSON save/load round-trip
# -----------------------------------------------------------------------

def test_json_roundtrip() -> Tuple[bool, str]:
    exp = SignatureHSweepExperiment(
        H_values=[0.1, 0.3], n_train=2000, n_val=500, n_test=500, epochs=5,
    )
    results = exp.run_full_sweep()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    exp.save_results(results, path)
    with open(path) as f:
        loaded = json.load(f)
    path.unlink()

    ok = (len(loaded) == 2
          and abs(loaded[0]["gamma_flat"] - results[0]["gamma_flat"]) < 1e-6
          and "sigfull_metrics" in loaded[1])
    return ok, f"loaded {len(loaded)} records, keys ok"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. run_stage_1 structure",        test_stage_1_structure),
        ("2. Feature importance",            test_feature_importance),
        ("3. TwoTowerHedger trains",         test_two_tower),
        ("4. run_single_H works",           test_h_sweep_single),
        ("5. JSON save/load round-trip",     test_json_roundtrip),
    ]

    print("=" * 65, flush=True)
    print(" Signature Ablation — Smoke Tests", flush=True)
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
