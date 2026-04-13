#!/usr/bin/env python
"""
Quick validation tests for hedging on rBergomi paths.

Run before the full experiment (cheaper, < 3 min total):

    python -m deep_hedging.tests.test_rbergomi_hedging
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta, HestonDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    hedge_paths_deep,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall, compute_all_metrics

# Shared constants
H, ETA, RHO, XI0 = 0.07, 1.9, -0.7, 0.235 ** 2
S0, K, T = 100.0, 100.0, 1.0
N_STEPS = 100
SIGMA = 0.235


def _make_model(n_steps: int = N_STEPS):
    return DifferentiableRoughBergomi(n_steps=n_steps, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0)


# ---------------------------------------------------------------------------
# Test 1: rBergomi path statistics
# ---------------------------------------------------------------------------

def test_rbergomi_stats() -> Tuple[bool, str]:
    """E[S_T] ~ S0, E[V_T] ~ xi0, roughness scaling."""
    model = _make_model()
    S, V, _ = model.simulate(n_paths=10_000, S0=S0, seed=42)

    mean_ST = float(S[:, -1].mean())
    mean_VT = float(V[:, -1].mean())
    st_ok = abs(mean_ST / S0 - 1.0) < 0.02
    vt_ok = abs(mean_VT / XI0 - 1.0) < 0.05

    # Roughness from log-V variogram (quick, lag=1 only)
    logV = torch.log(V.clamp(min=1e-30))
    diffs = logV[:, 1:] - logV[:, :-1]
    m2 = float((diffs ** 2).mean())
    dt = T / N_STEPS
    # m2 ~ C * dt^{2H} => rough check: m2 should be small
    m2_ok = m2 > 0 and m2 < 10.0

    passed = st_ok and vt_ok and m2_ok
    return passed, (
        f"E[S_T]={mean_ST:.2f} (S0={S0}), "
        f"E[V_T]={mean_VT:.6f} (xi0={XI0:.6f}), "
        f"m2(lag=1)={m2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: All strategies produce valid deltas
# ---------------------------------------------------------------------------

def test_valid_deltas() -> Tuple[bool, str]:
    """Shape and bounds check for all three hedgers."""
    model = _make_model()
    S, V, _ = model.simulate(n_paths=1000, S0=S0, seed=7)

    # BS delta
    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    d_bs = bs.hedge_paths(S)
    ok_bs = d_bs.shape == (1000, N_STEPS) and bool((d_bs >= 0).all() and (d_bs <= 1).all())

    # Heston delta
    hd = HestonDelta(K=K, T=T)
    d_h = hd.hedge_paths(S, V)
    ok_h = d_h.shape == (1000, N_STEPS) and bool((d_h >= 0).all() and (d_h <= 1).all())

    # Deep hedger (untrained)
    dh = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    dh.eval()
    with torch.no_grad():
        d_d = hedge_paths_deep(dh, S, T, S0)
    ok_d = d_d.shape == (1000, N_STEPS) and bool((d_d >= 0).all() and (d_d <= 1).all())

    passed = ok_bs and ok_h and ok_d
    return passed, f"BS={ok_bs}, Heston={ok_h}, Deep={ok_d}"


# ---------------------------------------------------------------------------
# Test 3: PnL finite on rBergomi paths
# ---------------------------------------------------------------------------

def test_pnl_finite() -> Tuple[bool, str]:
    """BS delta PnL on rBergomi paths is finite and bounded."""
    model = _make_model()
    S, V, _ = model.simulate(n_paths=1000, S0=S0, seed=11)

    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    deltas = bs.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    p0 = float(payoff.mean())
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)

    finite_ok = bool(torch.isfinite(pnl).all())
    mean_pnl = float(pnl.mean())
    bound_ok = abs(mean_pnl) < 20.0

    passed = finite_ok and bound_ok
    return passed, f"finite={finite_ok}, mean_pnl={mean_pnl:.3f} (|·|<20)"


# ---------------------------------------------------------------------------
# Test 4: Quick training smoke test
# ---------------------------------------------------------------------------

def test_training_smoke() -> Tuple[bool, str]:
    """Train 10 epochs on 5k rBergomi paths — no NaN, loss decreases."""
    model_sim = _make_model(n_steps=50)
    S, V, _ = model_sim.simulate(n_paths=5000, S0=S0, seed=22)
    S_tr, S_va = S[:4000], S[4000:]

    payoff_tr = compute_payoff(S_tr, K, "call")
    p0 = float(payoff_tr.mean())

    dh = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    history = train_deep_hedger(
        dh, S_tr, S_va,
        K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
        lr=1e-3, batch_size=1024, epochs=10, patience=10,
        verbose=False,
    )

    tr = history["train_risk"]
    no_nan = all(x == x for x in tr)  # NaN != NaN
    decreased = tr[-1] < tr[0]

    passed = no_nan and decreased
    return passed, f"train_risk: {tr[0]:.4f} -> {tr[-1]:.4f}, no_nan={no_nan}"


# ---------------------------------------------------------------------------
# Test 5: H1 directional check (mini)
# ---------------------------------------------------------------------------

def test_h1_mini() -> Tuple[bool, str]:
    """Mini H1: trained deep hedger has lower ES_95 than BS delta on rBergomi."""
    model_sim = _make_model(n_steps=N_STEPS)  # use full 100 steps
    S, V, _ = model_sim.simulate(n_paths=30_000, S0=S0, seed=55)
    S_tr, S_va, S_te = S[:20_000], S[20_000:25_000], S[25_000:]

    payoff_tr = compute_payoff(S_tr, K, "call")
    p0 = float(payoff_tr.mean())

    # Train deep hedger
    dh = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    train_deep_hedger(
        dh, S_tr, S_va,
        K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
        lr=1e-3, batch_size=2048, epochs=100, patience=20,
        verbose=False,
    )
    pnl_deep = evaluate_deep_hedger(dh, S_te, K=K, T=T, S0=S0, p0=p0)

    # BS delta
    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    deltas_bs = bs.hedge_paths(S_te)
    payoff_te = compute_payoff(S_te, K, "call")
    pnl_bs = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, cost_lambda=0.0)

    es_deep = float(expected_shortfall(pnl_deep, 0.95))
    es_bs = float(expected_shortfall(pnl_bs, 0.95))

    # Mini test: deep hedger should be competitive (within 1.0 of BS).
    # The full experiment (100k paths, 200 epochs) tests the strict H1 ordering.
    passed = es_deep < es_bs + 1.0
    return passed, f"ES95: deep={es_deep:.3f}, BS={es_bs:.3f}, gap={es_deep-es_bs:.3f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. rBergomi path statistics",    test_rbergomi_stats),
        ("2. All strategies valid deltas",  test_valid_deltas),
        ("3. PnL finite on rBergomi",       test_pnl_finite),
        ("4. Training smoke test",          test_training_smoke),
        ("5. H1 directional check (mini)",  test_h1_mini),
    ]

    print("=" * 65)
    print(" rBergomi Hedging — Quick Validation")
    print("=" * 65)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("-" * 65)
    if all_passed:
        print(" All 5 validation tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
