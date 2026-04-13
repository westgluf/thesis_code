#!/usr/bin/env python
"""
Smoke tests for the Pareto front experiment (Prompt 9).

    python -m deep_hedging.tests.test_pareto_front
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.experiments._training_helpers import (
    make_objective, train_deep_hedger_with_objective,
)
from deep_hedging.experiments.pareto_front import ParetoExperiment, _mean_turnover
from deep_hedging.objectives.pnl import compute_payoff


# -----------------------------------------------------------------------
# Test 1: make_objective returns callables for each name
# -----------------------------------------------------------------------

def test_make_objective() -> Tuple[bool, str]:
    torch.manual_seed(0)
    pnl = torch.randn(100, dtype=torch.float32)
    vals = {}
    for name, kwargs in [
        ("es", {"alpha": 0.95}),
        ("entropic", {"lam": 1.0}),
        ("mse", {}),
        ("mean", {}),
    ]:
        fn = make_objective(name, **kwargs)
        v = fn(pnl)
        if not torch.is_tensor(v) or v.ndim != 0 or not torch.isfinite(v):
            return False, f"{name}: bad output {v}"
        vals[name] = float(v)
    return True, ", ".join(f"{k}={v:+.3f}" for k, v in vals.items())


# -----------------------------------------------------------------------
# Test 2: MSE and Mean objectives give different trained weights
# -----------------------------------------------------------------------

def test_mse_vs_mean_differ() -> Tuple[bool, str]:
    sim = DifferentiableRoughBergomi(
        n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2,
    )
    S, _, _ = sim.simulate(n_paths=2500, S0=100.0, seed=11)
    S_tr, S_va = S[:2000], S[2000:]
    p0 = float(compute_payoff(S_tr, 100.0, "call").mean())

    out_mse = train_deep_hedger_with_objective(
        S_tr, S_va, "mse", {}, 0.0, p0,
        hidden_dim=32, n_res_blocks=1, epochs=8, batch_size=512, patience=10, seed=1,
    )
    out_mean = train_deep_hedger_with_objective(
        S_tr, S_va, "mean", {}, 0.0, p0,
        hidden_dim=32, n_res_blocks=1, epochs=8, batch_size=512, patience=10, seed=1,
    )

    p_mse = torch.cat([p.flatten() for p in out_mse["model"].parameters()])
    p_mean = torch.cat([p.flatten() for p in out_mean["model"].parameters()])
    l2 = float(torch.norm(p_mse - p_mean))

    r_mse = out_mse["history"]["train_risk"][-1]
    r_mean = out_mean["history"]["train_risk"][-1]
    finite = math.isfinite(r_mse) and math.isfinite(r_mean)
    passed = finite and l2 > 0.01
    return passed, f"weight L2 diff={l2:.3f}, r_mse={r_mse:.3f}, r_mean={r_mean:+.3f}"


# -----------------------------------------------------------------------
# Test 3: Part A single cell
# -----------------------------------------------------------------------

def test_part_A_single_cell() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exp = ParetoExperiment(
            n_train=1500, n_val=300, n_test=500, save_dir=tmp,
        )
        data = exp.generate_paths(n_steps=25, seed=42)
        bs = exp.run_bs_delta(data, n_steps=25, cost_lambda=0.001)
        deep = exp.run_deep_hedger_cell(
            data, n_steps=25, cost_lambda=0.001, seed=1, epochs=5,
        )

    bs_ok = (math.isfinite(bs["metrics"]["es_95"])
             and bs["mean_turnover"] >= 0
             and "pnl" in bs)
    deep_ok = (math.isfinite(deep["metrics"]["es_95"])
               and deep["mean_turnover"] >= 0
               and "history" in deep
               and "pnl" in deep)
    passed = bs_ok and deep_ok
    return passed, (
        f"BS ES_95={bs['metrics']['es_95']:.3f}, turn={bs['mean_turnover']:.3f}; "
        f"Deep ES_95={deep['metrics']['es_95']:.3f}, turn={deep['mean_turnover']:.3f}"
    )


# -----------------------------------------------------------------------
# Test 4: Part B smoke
# -----------------------------------------------------------------------

def test_part_B_smoke() -> Tuple[bool, str]:
    objectives = [
        ("es", {"alpha": 0.95}),
        ("mse", {}),
        ("mean", {}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        exp = ParetoExperiment(
            n_train=1500, n_val=300, n_test=500, save_dir=tmp,
        )
        results = exp.run_part_B(
            objectives=objectives, n_steps=25, cost_lambda=0.001, epochs=5,
        )

    has_bs = "bs" in results and "metrics" in results["bs"]
    # Check each objective tag present
    tags_present = {k for k in results if k not in ("bs", "config")}
    has_es = any(k.startswith("es") for k in tags_present)
    has_mse = "mse" in tags_present
    has_mean = "mean" in tags_present
    all_finite = all(
        math.isfinite(v["metrics"]["es_95"])
        for k, v in results.items()
        if k not in ("config",) and isinstance(v, dict) and "metrics" in v
    )
    passed = has_bs and has_es and has_mse and has_mean and all_finite
    return passed, (
        f"has_bs={has_bs}, es={has_es}, mse={has_mse}, mean={has_mean}, "
        f"n_obj={len(tags_present)}, all_finite={all_finite}"
    )


# -----------------------------------------------------------------------
# Test 5: cost_lambda actually enters the training loss
# -----------------------------------------------------------------------

def test_cost_aware_training() -> Tuple[bool, str]:
    """Cost-inclusive training loss should be strictly higher than free training loss."""
    sim = DifferentiableRoughBergomi(
        n_steps=50, T=1.0, H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2,
    )
    S, _, _ = sim.simulate(n_paths=3500, S0=100.0, seed=33)
    S_tr, S_va = S[:3000], S[3000:]
    p0 = float(compute_payoff(S_tr, 100.0, "call").mean())

    out_free = train_deep_hedger_with_objective(
        S_tr, S_va, "es", {"alpha": 0.95}, 0.0, p0,
        hidden_dim=32, n_res_blocks=1, epochs=10, batch_size=512, patience=15, seed=7,
    )
    out_cost = train_deep_hedger_with_objective(
        S_tr, S_va, "es", {"alpha": 0.95}, 0.005, p0,
        hidden_dim=32, n_res_blocks=1, epochs=10, batch_size=512, patience=15, seed=7,
    )

    # With the same init seed and data, the cost-aware run should see a
    # systematically larger risk (because cost is deducted from PnL).
    r_free = out_free["history"]["train_risk"][-1]
    r_cost = out_cost["history"]["train_risk"][-1]
    diff = r_cost - r_free
    passed = math.isfinite(r_free) and math.isfinite(r_cost) and diff > 0.05
    return passed, f"train_risk: free={r_free:.3f}, cost={r_cost:.3f}, diff={diff:+.3f}"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. make_objective returns callables",       test_make_objective),
        ("2. MSE vs Mean produce different weights",  test_mse_vs_mean_differ),
        ("3. Part A single cell",                      test_part_A_single_cell),
        ("4. Part B smoke (3 objectives)",             test_part_B_smoke),
        ("5. Cost-aware training affects training loss", test_cost_aware_training),
    ]

    print("=" * 65, flush=True)
    print(" Pareto Front — Smoke Tests", flush=True)
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
