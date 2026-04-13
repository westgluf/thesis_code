#!/usr/bin/env python
"""
Validation suite for delta hedging, P&L computation, and risk measures.

Six tests plus three diagnostic figures.  Run with:

    python -m deep_hedging.tests.test_hedging_basics
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.gbm import GBM
from deep_hedging.core.heston import Heston
from deep_hedging.hedging.delta_hedger import BlackScholesDelta, HestonDelta
from deep_hedging.objectives.pnl import (
    compute_payoff,
    compute_hedging_pnl,
)
from deep_hedging.objectives.risk_measures import (
    expected_shortfall,
    entropic_risk,
    value_at_risk,
    compute_all_metrics,
)

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"


# ---------------------------------------------------------------------------
# Shared simulation helpers
# ---------------------------------------------------------------------------

def _simulate_gbm(n_paths: int = 100_000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate GBM paths and return (S, V, dt).  sigma=0.235, S0=100, T=1, n=100."""
    model = GBM(n_steps=100, T=1.0, sigma=0.235)
    S, V, t_grid = model.simulate(n_paths=n_paths, S0=100.0, seed=seed)
    return S, V, float(model.dt)


# ---------------------------------------------------------------------------
# Test 1: BS delta on GBM → near-zero mean PnL
# ---------------------------------------------------------------------------

def test_bs_delta_gbm_replication() -> Tuple[bool, str]:
    """Under correctly-specified GBM, BS delta replicates the call."""
    S, V, _ = _simulate_gbm(100_000, seed=42)

    sigma, K, T = 0.235, 100.0, 1.0
    hedger = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas = hedger.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    p0 = BlackScholesDelta.bs_call_price(100.0, K, T, sigma)
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)

    mu = float(pnl.mean())
    sd = float(pnl.std())
    passed = abs(mu) < 0.5 and sd < 3.0
    return passed, f"mean={mu:.4f} (|·|<0.5), std={sd:.4f} (<3.0), p0={p0:.4f}"


# ---------------------------------------------------------------------------
# Test 2: Misspecified sigma → systematic bias
# ---------------------------------------------------------------------------

def test_bs_delta_misspecified() -> Tuple[bool, str]:
    """With sigma_assumed != sigma_true, there is a PnL bias."""
    S, V, _ = _simulate_gbm(100_000, seed=42)

    K, T = 100.0, 1.0
    hedger = BlackScholesDelta(sigma=0.30, K=K, T=T)      # overestimated
    deltas = hedger.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    # p0 set at the WRONG vol
    p0 = BlackScholesDelta.bs_call_price(100.0, K, T, 0.30)
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)

    mu = float(pnl.mean())
    passed = abs(mu) > 0.1
    return passed, f"mean={mu:.4f} (|·|>0.1, showing bias from misspecification)"


# ---------------------------------------------------------------------------
# Test 3: Transaction costs increase dispersion
# ---------------------------------------------------------------------------

def test_costs_increase_dispersion() -> Tuple[bool, str]:
    """Adding proportional costs increases std and ES."""
    S, V, _ = _simulate_gbm(100_000, seed=42)

    sigma, K, T = 0.235, 100.0, 1.0
    hedger = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas = hedger.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    p0 = BlackScholesDelta.bs_call_price(100.0, K, T, sigma)

    pnl_no_cost = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)
    pnl_cost = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.005)

    std_nc = float(pnl_no_cost.std())
    std_c = float(pnl_cost.std())
    es_nc = float(expected_shortfall(pnl_no_cost, 0.95))
    es_c = float(expected_shortfall(pnl_cost, 0.95))

    passed = std_c > std_nc and es_c > es_nc
    return passed, (
        f"std: {std_nc:.4f} -> {std_c:.4f}, "
        f"ES95: {es_nc:.4f} -> {es_c:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Risk measure basic properties
# ---------------------------------------------------------------------------

def test_risk_measure_properties() -> Tuple[bool, str]:
    """Verify monotonicity, translation-invariance, and key inequality."""
    torch.manual_seed(12345)
    pnl = torch.randn(10_000, dtype=torch.float64)

    # ES monotonicity: ES_0.99 >= ES_0.95 >= ES_0.50
    es99 = float(expected_shortfall(pnl, 0.99))
    es95 = float(expected_shortfall(pnl, 0.95))
    es50 = float(expected_shortfall(pnl, 0.50))
    mono_ok = es99 >= es95 >= es50

    # Translation invariance: entropic(c * ones) ≈ -c
    c = 3.0
    pnl_const = c * torch.ones(10_000, dtype=torch.float64)
    ent_const = float(entropic_risk(pnl_const, lam=1.0))
    transl_ok = abs(ent_const - (-c)) < 0.01

    # Standard inequality: VaR_alpha <= ES_alpha
    var95 = float(value_at_risk(pnl, 0.95))
    ineq_ok = var95 <= es95 + 1e-6

    # compute_all_metrics returns expected keys
    m = compute_all_metrics(pnl)
    keys_expected = {
        "mean_pnl", "std_pnl", "var_95", "es_95", "es_99",
        "entropic_1", "max_loss", "min_pnl", "skewness", "kurtosis",
    }
    keys_ok = keys_expected.issubset(m.keys())

    passed = mono_ok and transl_ok and ineq_ok and keys_ok
    details = (
        f"mono={mono_ok} (ES99={es99:.3f}>=ES95={es95:.3f}>=ES50={es50:.3f}), "
        f"transl={transl_ok} (ent(3)={ent_const:.4f}≈-3), "
        f"ineq={ineq_ok} (VaR95={var95:.3f}<=ES95), "
        f"keys={keys_ok}"
    )
    return passed, details


# ---------------------------------------------------------------------------
# Test 5: BS call price sanity vs MC
# ---------------------------------------------------------------------------

def test_bs_price_sanity() -> Tuple[bool, str]:
    """Compare analytical BS price with MC estimate from 200k GBM paths."""
    S0, K, T, sigma = 100.0, 100.0, 1.0, 0.235

    C_analytical = BlackScholesDelta.bs_call_price(S0, K, T, sigma)

    model = GBM(n_steps=100, T=T, sigma=sigma)
    S, _, _ = model.simulate(n_paths=200_000, S0=S0, seed=999)
    C_mc = float(torch.relu(S[:, -1] - K).mean())

    err = abs(C_analytical - C_mc)
    passed = err < 0.3
    return passed, f"C_BS={C_analytical:.4f}, C_MC={C_mc:.4f}, |err|={err:.4f} (<0.3)"


# ---------------------------------------------------------------------------
# Test 6: Delta shapes and bounds
# ---------------------------------------------------------------------------

def test_delta_shapes_and_bounds() -> Tuple[bool, str]:
    """Check shapes, bounds [0,1], and ATM delta ≈ 0.5."""
    n_paths = 1_000
    K, T = 100.0, 1.0

    # BS delta on GBM
    gbm = GBM(n_steps=100, T=T, sigma=0.235)
    S_gbm, V_gbm, _ = gbm.simulate(n_paths=n_paths, S0=100.0, seed=7)
    bs_hedger = BlackScholesDelta(sigma=0.235, K=K, T=T)
    d_bs = bs_hedger.hedge_paths(S_gbm)
    shape_ok_bs = d_bs.shape == (n_paths, 100)
    bounds_ok_bs = bool((d_bs >= 0.0).all() and (d_bs <= 1.0).all())

    # Heston delta on Heston paths
    hest = Heston(n_steps=100, T=T)
    S_h, V_h, _ = hest.simulate(n_paths=n_paths, S0=100.0, seed=8)
    hest_hedger = HestonDelta(K=K, T=T)
    d_h = hest_hedger.hedge_paths(S_h, V_h)
    shape_ok_h = d_h.shape == (n_paths, 100)
    bounds_ok_h = bool((d_h >= 0.0).all() and (d_h <= 1.0).all())

    # ATM check: at t=0, S0=100, K=100 → delta ≈ 0.5
    t0 = torch.tensor(0.0, dtype=torch.float64)
    S0_t = torch.tensor([100.0], dtype=torch.float64)
    atm_delta = float(bs_hedger.compute_delta(t0, S0_t)[0])
    atm_ok = abs(atm_delta - 0.5) < 0.1

    passed = shape_ok_bs and bounds_ok_bs and shape_ok_h and bounds_ok_h and atm_ok
    return passed, (
        f"BS: shape={shape_ok_bs}, bounds={bounds_ok_bs}; "
        f"Heston: shape={shape_ok_h}, bounds={bounds_ok_h}; "
        f"ATM delta={atm_delta:.4f} (≈0.5)"
    )


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

def _make_figures() -> None:
    """Generate three diagnostic plots and save to figures/."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    S, V, _ = _simulate_gbm(100_000, seed=42)
    sigma, K, T, S0 = 0.235, 100.0, 1.0, 100.0
    hedger = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas = hedger.hedge_paths(S)
    payoff = compute_payoff(S, K, "call")
    p0 = BlackScholesDelta.bs_call_price(S0, K, T, sigma)

    pnl_nc = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)
    pnl_c = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.005)

    # --- Figure 1: PnL histograms ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    pnl_nc_np = pnl_nc.detach().numpy()
    pnl_c_np = pnl_c.detach().numpy()

    bins = 150
    ax.hist(pnl_nc_np, bins=bins, alpha=0.6, density=True, label="No costs")
    ax.hist(pnl_c_np, bins=bins, alpha=0.6, density=True, label=r"With costs $\lambda=0.005$")

    es_nc = float(expected_shortfall(pnl_nc, 0.95))
    es_c = float(expected_shortfall(pnl_c, 0.95))
    ax.axvline(float(pnl_nc.mean()), color="C0", ls="--", lw=1.5, label=f"Mean (no cost) = {float(pnl_nc.mean()):.2f}")
    ax.axvline(-es_nc, color="C0", ls=":", lw=1.5, label=f"-ES95 (no cost) = {-es_nc:.2f}")
    ax.axvline(float(pnl_c.mean()), color="C1", ls="--", lw=1.5, label=f"Mean (cost) = {float(pnl_c.mean()):.2f}")
    ax.axvline(-es_c, color="C1", ls=":", lw=1.5, label=f"-ES95 (cost) = {-es_c:.2f}")

    ax.set_xlabel("P&L")
    ax.set_ylabel("Density")
    ax.set_title("GBM Benchmark: BS Delta P&L Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(-15, 10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_pnl_gbm_baseline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_pnl_gbm_baseline.png'}")

    # --- Figure 2: Delta surface ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    t_grid = torch.linspace(0.0, T - 0.01, 100, dtype=torch.float64)
    S_grid = torch.linspace(80.0, 120.0, 100, dtype=torch.float64)
    TT, SS = torch.meshgrid(t_grid, S_grid, indexing="ij")

    # Compute delta over the grid
    tau = T - TT
    sigma_t = torch.tensor(sigma, dtype=torch.float64)
    tau_safe = torch.clamp(tau, min=1e-8)
    d1 = (torch.log(SS / K) + 0.5 * sigma ** 2 * tau_safe) / (sigma * torch.sqrt(tau_safe))
    delta_surface = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))

    c = ax.contourf(
        TT.numpy(), SS.numpy(), delta_surface.numpy(),
        levels=30, cmap="RdYlGn",
    )
    fig.colorbar(c, ax=ax, label="Delta")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spot price")
    ax.set_title(f"BS Delta Surface (K={K}, σ={sigma}, T={T})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_delta_surface.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_delta_surface.png'}")

    # --- Figure 3: Risk measures bar chart ---
    m_nc = compute_all_metrics(pnl_nc)
    m_c = compute_all_metrics(pnl_c)

    metric_keys = ["std_pnl", "var_95", "es_95", "es_99", "entropic_1"]
    labels = ["Std", "VaR95", "ES95", "ES99", "Entropic(1)"]

    vals_nc = [m_nc[k] for k in metric_keys]
    vals_c = [m_c[k] for k in metric_keys]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], vals_nc, width, label="No costs", color="steelblue")
    ax.bar([i + width / 2 for i in x], vals_c, width, label=r"With costs ($\lambda=0.005$)", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Risk metric value")
    ax.set_title("GBM + BS Delta: Risk Measures Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_risk_measures_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_risk_measures_bar.png'}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. BS delta on GBM → near-zero PnL",       test_bs_delta_gbm_replication),
        ("2. Misspecified sigma → bias",              test_bs_delta_misspecified),
        ("3. Costs increase dispersion",              test_costs_increase_dispersion),
        ("4. Risk measure properties",                test_risk_measure_properties),
        ("5. BS call price vs MC",                    test_bs_price_sanity),
        ("6. Delta shapes and bounds",                test_delta_shapes_and_bounds),
    ]

    print("=" * 70)
    print(" Delta Hedging & Risk Measures — Validation Suite")
    print("=" * 70)

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

    print("-" * 70)

    # Generate figures
    print("\n  Generating diagnostic figures...")
    try:
        _make_figures()
    except Exception as exc:
        print(f"  WARNING: Figure generation failed: {exc}")

    print("-" * 70)
    if all_passed:
        print(" All 6 tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
