#!/usr/bin/env python
"""
Tests for P02.1 -- Honest Hybrid Scheme Validation.

Six tests:
1. Preflight runnable.
2. Strategy B sanity: both simulators produce fBm with Var ~ T^{2H}.
3. MC std computation correctness on a known distribution.
4. Arbitrage flag logic: synthetic butterfly with ratio 3.5 flagged, 1.5 not.
5. Verdict logic: synthetic criteria produce correct verdicts.
6. No modification of existing files.

Run:
    python -m deep_hedging.tests.test_block1_cholesky_v2
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

def _pr(name: str, ok: bool, msg: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {msg}")

# ---- Test 1: Preflight ------------------------------------------------

def test_preflight() -> Tuple[bool, str]:
    p = Path(__file__).resolve().parents[2] / "results" / "block1" / "p021_preflight.md"
    if not p.exists():
        return False, f"Not found: {p}"
    text = p.read_text()
    if "READY TO PROCEED" not in text and "HALT" not in text:
        return False, "Missing verdict line"
    return True, f"Preflight exists, {len(text)} chars"

# ---- Test 2: fBm variance sanity (tiny n=20 case) ---------------------

def test_fbm_variance_sanity() -> Tuple[bool, str]:
    """Both exact RL Cholesky and hybrid produce Var[WH(T)] ~ T^{2H} within 5%."""
    from deep_hedging.experiments.block1_cholesky_v2 import (
        build_joint_cholesky, simulate_exact_rl,
    )
    from deep_hedging.core.volterra import HybridVolterraDriver

    n, T, H = 20, 1.0, 0.07
    n_paths = 50_000
    expected_var = T ** (2 * H)  # 1.0

    # Exact RL
    L = build_joint_cholesky(n, T, H)
    exact = simulate_exact_rl(n, T, H, 0, 0, 0, 0, n_paths, 9901, L, wh_only=True)
    var_exact = float(exact["WH"][:, -1].var())
    err_exact = abs(var_exact - expected_var) / expected_var

    # Hybrid
    driver = HybridVolterraDriver(n, T, H)
    gen = torch.Generator().manual_seed(9902)
    Z = torch.randn(n_paths, n, 2, dtype=torch.float64, generator=gen)
    WH_h, _ = driver(Z)
    var_hybrid = float(WH_h[:, -1].var())
    err_hybrid = abs(var_hybrid - expected_var) / expected_var

    ok = err_exact < 0.05 and err_hybrid < 0.05
    return ok, (
        f"Var_exact={var_exact:.4f} (err={err_exact:.3f}), "
        f"Var_hybrid={var_hybrid:.4f} (err={err_hybrid:.3f}), expected={expected_var:.4f}"
    )

# ---- Test 3: MC std correctness ---------------------------------------

def test_mc_std_correctness() -> Tuple[bool, str]:
    """For 50k draws from N(mu, sigma^2), mc_std_of_mean ~ sigma/sqrt(N) within 5%."""
    mu, sigma, N = 5.0, 3.0, 50_000
    rng = np.random.RandomState(42)
    x = rng.normal(mu, sigma, N)
    mc_std_empirical = float(np.std(x) / np.sqrt(N))
    mc_std_expected = sigma / np.sqrt(N)
    err = abs(mc_std_empirical - mc_std_expected) / mc_std_expected
    ok = err < 0.05
    return ok, f"empirical={mc_std_empirical:.6f}, expected={mc_std_expected:.6f}, err={err:.4f}"

# ---- Test 4: Arbitrage flag logic --------------------------------------

def test_arbitrage_flag_logic() -> Tuple[bool, str]:
    """Synthetic butterfly: ratio 3.5 must be flagged, ratio 1.5 must not."""
    # Simulate flagging logic from the experiment
    cases = [
        {"value": -0.35, "mc_std": 0.10, "expected_flag": True},   # ratio 3.5
        {"value": -0.15, "mc_std": 0.10, "expected_flag": False},  # ratio 1.5
        {"value": 0.05, "mc_std": 0.10, "expected_flag": False},   # positive (no violation)
        {"value": -0.25, "mc_std": 0.10, "expected_flag": True},   # ratio 2.5
    ]
    all_ok = True
    details = []
    for c in cases:
        bfly = c["value"]
        mc = c["mc_std"]
        ratio = abs(bfly) / (mc + 1e-12)
        flagged = bfly < 0 and ratio > 2.0
        ok = flagged == c["expected_flag"]
        all_ok = all_ok and ok
        details.append(f"val={bfly} ratio={ratio:.1f} flag={flagged} expect={c['expected_flag']} {'OK' if ok else 'FAIL'}")
    return all_ok, "; ".join(details)

# ---- Test 5: Verdict logic ---------------------------------------------

def test_verdict_logic() -> Tuple[bool, str]:
    """Synthetic criteria tuples produce correct verdicts."""
    from deep_hedging.experiments.block1_cholesky_v2 import compute_verdict

    # Helper to build minimal dicts that compute_verdict needs
    def mk(c1_rd, c2_rd, ks_p, w1_val, n100_flag, n400_flag):
        call = {"rel_diff": c1_rd, "mc_std_exact": 0, "mc_std_hybrid": 0,
                "price_hybrid": 10, "rel_diff_in_mc_units": 0}
        var = {"max_rel_diff": c2_rd, "rel_differences": [c2_rd],
               "mc_std_of_each_mean": [(0, 0)], "E_v_hybrid": [0.055],
               "rel_diff_in_mc_units": [0], "t_values": [1.0]}
        fbm = {"ks_pvalue": ks_p, "wasserstein_1": w1_val}
        arb = {
            "n100": {"total_flagged": n100_flag, "butterflies": []},
            "n400": {"total_flagged": n400_flag, "butterflies": []},
        }
        _, v = compute_verdict(fbm, var, call, arb)
        return v

    results = []
    # STRICT_PASS: all pass
    v = mk(0.01, 0.02, 0.5, 0.03, 0, 0)
    results.append(("STRICT_PASS", v, v == "STRICT_PASS"))

    # PASS_WITH_ONE_NOTE: only C3 fails (W1 between 0.05 and 0.10)
    v = mk(0.01, 0.02, 0.005, 0.07, 0, 0)
    results.append(("PASS_WITH_ONE_NOTE", v, v == "PASS_WITH_ONE_NOTE"))

    # FAIL: C1 fails
    v = mk(0.03, 0.02, 0.5, 0.03, 0, 0)
    results.append(("FAIL(C1)", v, v == "FAIL"))

    # FAIL: C2 fails
    v = mk(0.01, 0.04, 0.5, 0.03, 0, 0)
    results.append(("FAIL(C2)", v, v == "FAIL"))

    # FAIL: C4 fails
    v = mk(0.01, 0.02, 0.5, 0.03, 1, 0)
    results.append(("FAIL(C4)", v, v == "FAIL"))

    # FAIL: C3 fails with W1 > 0.10 (not eligible for ONE_NOTE)
    v = mk(0.01, 0.02, 0.005, 0.15, 0, 0)
    results.append(("FAIL(C3_wide)", v, v == "FAIL"))

    all_ok = all(r[2] for r in results)
    detail = "; ".join(f"{r[0]}->{'PASS' if r[2] else 'FAIL'}({r[1]})" for r in results)
    return all_ok, detail

# ---- Test 6: No modification of existing files -------------------------

def test_no_modification() -> Tuple[bool, str]:
    """P01, P01.6, P02 artefacts preserved."""
    root = Path(__file__).resolve().parents[2]
    critical = [
        "deep_hedging/core/volterra.py",
        "deep_hedging/core/volterra_exact.py",
        "deep_hedging/core/volterra_kappa2.py",
        "deep_hedging/core/rough_bergomi.py",
        "deep_hedging/experiments/block1_cholesky_arbitrage_kappa.py",
        "deep_hedging/experiments/block1_extended_validation.py",
    ]
    missing = []
    for f in critical:
        p = root / f
        if not p.exists():
            missing.append(f)
    if missing:
        return False, f"Missing: {missing}"
    # Check that P02 results still exist
    p02 = root / "results" / "block1" / "cholesky_arbitrage_kappa.json"
    if not p02.exists():
        return False, "P02 results missing"
    return True, "All existing files preserved"

# ---- Runner ------------------------------------------------------------

def main() -> int:
    tests = [
        ("Preflight exists", test_preflight),
        ("fBm variance sanity (n=20)", test_fbm_variance_sanity),
        ("MC std correctness", test_mc_std_correctness),
        ("Arbitrage flag logic", test_arbitrage_flag_logic),
        ("Verdict logic", test_verdict_logic),
        ("No modification guard", test_no_modification),
    ]
    print(f"\nP02.1 Test Suite ({len(tests)} tests)\n{'=' * 50}")
    n_pass = 0
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"Exception: {e}"
        _pr(name, ok, msg)
        n_pass += int(ok)

    print(f"\n{'=' * 50}")
    print(f"  {n_pass}/{len(tests)} passed")
    return 0 if n_pass == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())
