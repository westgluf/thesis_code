#!/usr/bin/env python
"""
Smoke tests for the H2 grid extension (Prompt 9.5).

    python -m deep_hedging.tests.test_h2_extension
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch

from deep_hedging.experiments.h2_grid_extension import (
    FREQ_VALUES, COST_VALUES, PROMPT_9_FREQ, PROMPT_9_COST,
    PROMPT_9_JSON, N_TEST,
    load_prompt_9_cells,
    generate_paths_for_freq,
    evaluate_bs_delta_at_costs,
    detect_reversal,
)


# -----------------------------------------------------------------------
# Test 1: Grid parameters
# -----------------------------------------------------------------------

def test_grid_parameters() -> Tuple[bool, str]:
    expected_freq = [25, 50, 100, 200, 400, 800]
    expected_cost = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010]
    expected_p9_freq = {50, 100, 200}
    expected_p9_cost = {0.0, 0.001, 0.002}

    if FREQ_VALUES != expected_freq:
        return False, f"FREQ_VALUES mismatch: {FREQ_VALUES}"
    if COST_VALUES != expected_cost:
        return False, f"COST_VALUES mismatch: {COST_VALUES}"
    if PROMPT_9_FREQ != expected_p9_freq:
        return False, f"PROMPT_9_FREQ mismatch: {PROMPT_9_FREQ}"
    if PROMPT_9_COST != expected_p9_cost:
        return False, f"PROMPT_9_COST mismatch: {PROMPT_9_COST}"

    return True, f"grid {len(FREQ_VALUES)}x{len(COST_VALUES)} = {len(FREQ_VALUES)*len(COST_VALUES)} cells"


# -----------------------------------------------------------------------
# Test 2: Path generation at n_steps=400
# -----------------------------------------------------------------------

def test_path_generation_400() -> Tuple[bool, str]:
    data = generate_paths_for_freq(n_steps=400, device=torch.device("cpu"), n_paths=500)
    S = data["S"]
    payoff = data["payoff"]

    shape_ok = S.shape == (500, 401)
    finite_S = bool(torch.isfinite(S).all())
    finite_payoff = bool(torch.isfinite(payoff).all())
    p0_finite = math.isfinite(data["p0"])

    passed = shape_ok and finite_S and finite_payoff and p0_finite
    return passed, f"shape={tuple(S.shape)}, finite_S={finite_S}, p0={data['p0']:.3f}"


# -----------------------------------------------------------------------
# Test 3: n_steps=800 dry run
# -----------------------------------------------------------------------

def test_n_steps_800_dry_run() -> Tuple[bool, str]:
    data = generate_paths_for_freq(n_steps=800, device=torch.device("cpu"), n_paths=100)
    S = data["S"]
    shape_ok = S.shape == (100, 801)
    finite = bool(torch.isfinite(S).all())
    positive = bool((S > 0).all())
    passed = shape_ok and finite and positive
    return passed, f"shape={tuple(S.shape)}, finite={finite}, positive={positive}"


# -----------------------------------------------------------------------
# Test 4: BS delta at multiple costs
# -----------------------------------------------------------------------

def test_bs_delta_cost_monotonic() -> Tuple[bool, str]:
    data = generate_paths_for_freq(n_steps=100, device=torch.device("cpu"), n_paths=1000)
    cost_values = [0.0, 0.001, 0.002]
    cells = evaluate_bs_delta_at_costs(data, cost_values)

    if len(cells) != 3:
        return False, f"expected 3 cells, got {len(cells)}"

    es_values = [cells[c]["metrics"]["es_95"] for c in cost_values]
    monotonic = es_values[0] < es_values[1] < es_values[2]

    # Turnover should be identical across costs (cost-independent)
    turnovers = [cells[c]["mean_turnover"] for c in cost_values]
    turnover_constant = abs(turnovers[0] - turnovers[1]) < 1e-9 and abs(turnovers[1] - turnovers[2]) < 1e-9

    passed = monotonic and turnover_constant
    return passed, (
        f"ES_95: {es_values[0]:.3f} < {es_values[1]:.3f} < {es_values[2]:.3f}, "
        f"turnover constant={turnover_constant}"
    )


# -----------------------------------------------------------------------
# Test 5: Prompt 9 loader returns expected cells
# -----------------------------------------------------------------------

def test_prompt_9_loader() -> Tuple[bool, str]:
    cells = load_prompt_9_cells(PROMPT_9_JSON)

    # Should have 3 frequencies
    if set(cells.keys()) != {50, 100, 200}:
        return False, f"expected freq keys {{50,100,200}}, got {set(cells.keys())}"

    # Each should have 3 costs
    for n in (50, 100, 200):
        if set(cells[n].keys()) != {0.0, 0.001, 0.002}:
            return False, f"n={n}: cost keys {set(cells[n].keys())}"

    # Exact values from Prompt 9 run
    expected = {
        (50, 0.0): 12.751,
        (100, 0.001): 11.491,
        (200, 0.002): 11.697,
    }
    for (n, c), expected_es in expected.items():
        actual_es = cells[n][c]["metrics"]["es_95"]
        if abs(actual_es - expected_es) > 0.001:
            return False, f"(n={n},c={c}): expected {expected_es}, got {actual_es}"

    return True, (
        f"loaded 9 cells, "
        f"(50,0.0)={cells[50][0.0]['metrics']['es_95']:.3f}, "
        f"(100,0.001)={cells[100][0.001]['metrics']['es_95']:.3f}, "
        f"(200,0.002)={cells[200][0.002]['metrics']['es_95']:.3f}"
    )


# -----------------------------------------------------------------------
# Test 6: Reversal detection logic
# -----------------------------------------------------------------------

def test_reversal_detection() -> Tuple[bool, str]:
    """Synthetic grid: weak H2 at low cost, strong H2 at high cost."""
    # Build synthetic results dict matching the expected structure
    synthetic = {}
    # λ=0: strictly decreasing in n_steps (weak H2 — no reversal)
    # λ=0.005: has minimum at n=200, with higher values at 400 and 800 (strong H2)
    # λ=0.010: minimum at n=100 (strong H2)
    data_by_cost = {
        0.0: {25: 14.0, 50: 13.0, 100: 12.0, 200: 11.5, 400: 11.2, 800: 11.0},
        0.0005: {25: 14.1, 50: 13.2, 100: 12.3, 200: 11.8, 400: 11.5, 800: 11.3},
        0.001: {25: 14.2, 50: 13.3, 100: 12.5, 200: 12.0, 400: 11.7, 800: 11.5},
        0.002: {25: 14.3, 50: 13.4, 100: 12.6, 200: 12.2, 400: 12.0, 800: 12.0},
        0.003: {25: 14.4, 50: 13.5, 100: 12.8, 200: 12.5, 400: 12.5, 800: 12.6},
        0.005: {25: 14.6, 50: 13.7, 100: 13.0, 200: 12.8, 400: 13.0, 800: 13.2},
        0.010: {25: 15.0, 50: 14.0, 100: 13.2, 200: 13.4, 400: 13.8, 800: 14.2},
    }

    for n in [25, 50, 100, 200, 400, 800]:
        synthetic[n] = {}
        for c, es_by_n in data_by_cost.items():
            synthetic[n][c] = {
                "metrics": {"es_95": es_by_n[n]},
                "mean_turnover": float(n) / 25.0,
            }

    detection = detect_reversal(
        synthetic,
        freq_values=[25, 50, 100, 200, 400, 800],
        cost_values=list(data_by_cost.keys()),
    )

    # Check verdict
    if detection["verdict"] != "strong H2":
        return False, f"expected strong H2, got {detection['verdict']}"

    # Reversal threshold should be <= 0.005 (reversal visible at 0.005 and 0.010)
    if detection["reversal_cost_threshold"] is None:
        return False, "reversal threshold should be finite"
    if detection["reversal_cost_threshold"] > 0.005 + 1e-9:
        return False, f"threshold too high: {detection['reversal_cost_threshold']}"

    # At λ=0.010, minimum should be at n=100 (strong reversal)
    if detection["min_freq_by_cost"][0.010] != 100:
        return False, f"λ=0.01 min freq: {detection['min_freq_by_cost'][0.010]}"

    return True, (
        f"verdict={detection['verdict']}, "
        f"threshold={detection['reversal_cost_threshold']}, "
        f"min@0.010=n={detection['min_freq_by_cost'][0.010]}"
    )


# -----------------------------------------------------------------------
# Leland tests
# -----------------------------------------------------------------------

def test_leland_delta_zero_cost_matches_bs() -> Tuple[bool, str]:
    """Leland at lam=0 must be identical to BS."""
    from deep_hedging.hedging.delta_hedger import BlackScholesDelta, LelandDelta

    sigma, K, T = 0.235, 100.0, 1.0
    n_steps = 50

    bs = BlackScholesDelta(sigma=sigma, K=K, T=T)
    ld = LelandDelta(sigma=sigma, K=K, T=T, lam=0.0, n_steps=n_steps)

    sigma_ok = ld.sigma_leland == sigma
    torch.manual_seed(0)
    S = torch.rand(16, n_steps + 1) * 10 + 100.0
    delta_ok = bool(torch.allclose(bs.hedge_paths(S), ld.hedge_paths(S)))
    passed = sigma_ok and delta_ok
    return passed, f"sigma_match={sigma_ok}, delta_match={delta_ok}"


def test_leland_delta_monotone_in_lambda() -> Tuple[bool, str]:
    """sigma_Leland must grow monotonically with lam at fixed n."""
    from deep_hedging.hedging.delta_hedger import LelandDelta
    sigma, K, T = 0.235, 100.0, 1.0
    n_steps = 100

    prev = sigma
    all_mono = True
    for lam in [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        ld = LelandDelta(sigma=sigma, K=K, T=T, lam=lam, n_steps=n_steps)
        if ld.sigma_leland < prev - 1e-12:
            all_mono = False
        prev = ld.sigma_leland
    return all_mono, f"monotone={all_mono}, final_sigma={prev:.4f}"


def test_leland_formula_matches_literature() -> Tuple[bool, str]:
    """Hand-computed Leland sigma for a specific tuple."""
    from deep_hedging.hedging.delta_hedger import LelandDelta
    sigma, lam, T, n_steps = 0.2, 0.001, 1.0, 100
    dt = T / n_steps
    expected = sigma * math.sqrt(
        1.0 + math.sqrt(8.0 / math.pi) * lam / (sigma * math.sqrt(dt))
    )
    ld = LelandDelta(sigma=sigma, K=100.0, T=T, lam=lam, n_steps=n_steps)
    ok = abs(ld.sigma_leland - expected) < 1e-14
    return ok, f"computed={ld.sigma_leland:.8f}, expected={expected:.8f}"


def test_leland_rejects_invalid_inputs() -> Tuple[bool, str]:
    """LelandDelta must reject negative lam, zero sigma, zero n_steps."""
    from deep_hedging.hedging.delta_hedger import LelandDelta
    errors = []
    try:
        LelandDelta(sigma=0.2, K=100, T=1, lam=-0.001, n_steps=100)
        errors.append("accepted negative lam")
    except ValueError:
        pass
    try:
        LelandDelta(sigma=0.0, K=100, T=1, lam=0.001, n_steps=100)
        errors.append("accepted zero sigma")
    except ValueError:
        pass
    try:
        LelandDelta(sigma=0.2, K=100, T=1, lam=0.001, n_steps=0)
        errors.append("accepted zero n_steps")
    except ValueError:
        pass
    passed = len(errors) == 0
    return passed, f"errors={errors}" if errors else "All invalid inputs rejected"


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Grid parameters",                  test_grid_parameters),
        ("2. Path generation at n=400",          test_path_generation_400),
        ("3. n_steps=800 dry run",               test_n_steps_800_dry_run),
        ("4. BS delta cost monotonicity",        test_bs_delta_cost_monotonic),
        ("5. Prompt 9 loader",                   test_prompt_9_loader),
        ("6. Reversal detection logic",          test_reversal_detection),
        ("7. Leland lam=0 matches BS",           test_leland_delta_zero_cost_matches_bs),
        ("8. Leland monotone in lambda",         test_leland_delta_monotone_in_lambda),
        ("9. Leland formula vs literature",      test_leland_formula_matches_literature),
        ("10. Leland rejects invalid inputs",    test_leland_rejects_invalid_inputs),
    ]

    print("=" * 65, flush=True)
    print(" H2 Grid Extension — Smoke Tests", flush=True)
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
        print(f" All {len(tests)} tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
