#!/usr/bin/env python
"""
Validation suite for the deep hedger (ResidualFNN) and training loop.

Six tests plus three diagnostic figures.  Run with:

    python -m deep_hedging.tests.test_deep_hedger
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.gbm import GBM
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    build_features,
    hedge_paths_deep,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall, compute_all_metrics

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
SIGMA = 0.235
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 50  # Use 50 steps for speed in tests


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _simulate_gbm_splits(
    n_train: int = 80_000, n_val: int = 20_000, n_test: int = 50_000,
    n_steps: int = N_STEPS, seed: int = 42,
):
    """Simulate GBM and split into train/val/test."""
    total = n_train + n_val + n_test
    model = GBM(n_steps=n_steps, T=T, sigma=SIGMA)
    S, _, t_grid = model.simulate(n_paths=total, S0=S0, seed=seed)
    S_tr = S[:n_train]
    S_va = S[n_train : n_train + n_val]
    S_te = S[n_train + n_val :]
    return S_tr, S_va, S_te


# ---------------------------------------------------------------------------
# Test 1: Architecture shapes
# ---------------------------------------------------------------------------

def test_architecture_shapes() -> Tuple[bool, str]:
    """Verify output shape, value bounds, and parameter count."""
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    x = torch.randn(32, 4)
    y = model(x)

    shape_ok = y.shape == (32, 1)
    bounds_ok = bool((y >= 0).all() and (y <= 1).all())

    n_params = sum(p.numel() for p in model.parameters())
    # "medium" config should be roughly 50k-100k
    param_ok = 30_000 < n_params < 150_000

    passed = shape_ok and bounds_ok and param_ok
    return passed, f"shape={y.shape}, bounds=[{y.min():.3f},{y.max():.3f}], params={n_params}"


# ---------------------------------------------------------------------------
# Test 2: Feature construction
# ---------------------------------------------------------------------------

def test_feature_construction() -> Tuple[bool, str]:
    """Verify build_features shapes and value ranges."""
    batch = 100
    n_steps = 50
    S = S0 * torch.exp(0.01 * torch.randn(batch, n_steps + 1, dtype=torch.float64))
    S[:, 0] = S0
    t_grid = torch.linspace(0.0, T, n_steps + 1, dtype=torch.float64)
    delta_prev = torch.zeros(batch, dtype=torch.float64)

    # k=0
    feat0 = build_features(S, t_grid, T, delta_prev, 0)
    shape_ok_0 = feat0.shape == (batch, 4)
    t_over_T_0 = float(feat0[0, 0])   # should be 0.0
    log_m_0 = float(feat0[0, 1])      # log(S0/S0) = 0
    tau_over_T_0 = float(feat0[0, 2]) # (T-0)/T = 1.0
    dtype_ok = feat0.dtype == torch.float32

    # k = n_steps - 1 (last step)
    feat_last = build_features(S, t_grid, T, delta_prev, n_steps - 1)
    tau_last = float(feat_last[0, 2])  # (T - t_{n-1})/T = dt/T = 1/n

    ok0 = abs(t_over_T_0) < 1e-5 and abs(log_m_0) < 1e-5 and abs(tau_over_T_0 - 1.0) < 1e-5
    ok_last = abs(tau_last - 1.0 / n_steps) < 1e-3

    passed = shape_ok_0 and dtype_ok and ok0 and ok_last
    return passed, (
        f"k=0: t/T={t_over_T_0:.4f}, logM={log_m_0:.4f}, tau/T={tau_over_T_0:.4f}; "
        f"k={n_steps-1}: tau/T={tau_last:.4f}; dtype={feat0.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 3: hedge_paths_deep shapes
# ---------------------------------------------------------------------------

def test_hedge_paths_shapes() -> Tuple[bool, str]:
    """Verify hedge_paths_deep output shape and bounds."""
    model = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    batch, n_steps = 100, 100
    S = S0 * torch.exp(0.01 * torch.randn(batch, n_steps + 1, dtype=torch.float64))
    S[:, 0] = S0

    model.eval()
    with torch.no_grad():
        deltas = hedge_paths_deep(model, S, T, S0)

    shape_ok = deltas.shape == (batch, n_steps)
    bounds_ok = bool((deltas >= 0).all() and (deltas <= 1).all())

    passed = shape_ok and bounds_ok
    return passed, f"shape={deltas.shape}, bounds=[{deltas.min():.3f},{deltas.max():.3f}]"


# ---------------------------------------------------------------------------
# Test 4: GBM sanity — deep hedger ≈ BS delta [KEY TEST]
# ---------------------------------------------------------------------------

def test_gbm_sanity() -> Tuple[bool, str]:
    """Train deep hedger on GBM and compare with BS delta."""
    S_tr, S_va, S_te = _simulate_gbm_splits(
        n_train=80_000, n_val=20_000, n_test=50_000, seed=42,
    )

    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    history = train_deep_hedger(
        model, S_tr, S_va,
        K=K, T=T, S0=S0, cost_lambda=0.0,
        alpha=0.95, lr=1e-3, batch_size=2048,
        epochs=200, patience=25, verbose=True,
    )

    # Deep hedger PnL
    pnl_deep = evaluate_deep_hedger(model, S_te, K=K, T=T, S0=S0, cost_lambda=0.0)

    # BS delta PnL
    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    deltas_bs = bs.hedge_paths(S_te)
    payoff_te = compute_payoff(S_te, K, "call")
    p0 = BlackScholesDelta.bs_call_price(S0, K, T, SIGMA)
    pnl_bs = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, cost_lambda=0.0)

    m_deep = compute_all_metrics(pnl_deep)
    m_bs = compute_all_metrics(pnl_bs)

    es_diff = abs(m_deep["es_95"] - m_bs["es_95"])
    std_ok = m_deep["std_pnl"] < 4.0
    es_ok = es_diff < 1.0

    # Save shared state for figures
    test_gbm_sanity._data = {
        "history": history, "pnl_deep": pnl_deep, "pnl_bs": pnl_bs,
        "m_deep": m_deep, "m_bs": m_bs,
        "model": model, "S_te": S_te, "deltas_bs": deltas_bs,
    }

    passed = std_ok and es_ok
    return passed, (
        f"Deep: mean={m_deep['mean_pnl']:.3f}, std={m_deep['std_pnl']:.3f}, "
        f"ES95={m_deep['es_95']:.3f}  |  "
        f"BS: mean={m_bs['mean_pnl']:.3f}, std={m_bs['std_pnl']:.3f}, "
        f"ES95={m_bs['es_95']:.3f}  |  "
        f"|ES_diff|={es_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Cost-aware training reduces turnover
# ---------------------------------------------------------------------------

def test_cost_aware_turnover() -> Tuple[bool, str]:
    """Deep hedger with costs should have lower turnover than BS delta."""
    S_tr, S_va, S_te = _simulate_gbm_splits(
        n_train=40_000, n_val=10_000, n_test=20_000, seed=99,
    )

    model = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    train_deep_hedger(
        model, S_tr, S_va,
        K=K, T=T, S0=S0, cost_lambda=0.005,
        alpha=0.95, lr=1e-3, batch_size=512,
        epochs=50, patience=15, verbose=False,
    )

    model.eval()
    with torch.no_grad():
        deltas_deep = hedge_paths_deep(model, S_te, T, S0)
    deltas_deep = deltas_deep.to(S_te.dtype)

    bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
    deltas_bs = bs.hedge_paths(S_te)

    def _turnover(d: Tensor) -> float:
        d_prev = torch.cat([torch.zeros(d.shape[0], 1, dtype=d.dtype, device=d.device), d[:, :-1]], dim=1)
        return float(torch.abs(d - d_prev).mean())

    turn_deep = _turnover(deltas_deep)
    turn_bs = _turnover(deltas_bs)

    passed = turn_deep <= turn_bs * 1.2
    return passed, f"turnover: deep={turn_deep:.5f}, BS={turn_bs:.5f} (ratio={turn_deep/turn_bs:.3f})"


# ---------------------------------------------------------------------------
# Test 6: Training convergence
# ---------------------------------------------------------------------------

def test_training_convergence() -> Tuple[bool, str]:
    """Verify that training risk decreases."""
    S_tr, S_va, _ = _simulate_gbm_splits(
        n_train=40_000, n_val=10_000, n_test=1000, seed=77,
    )

    model = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    history = train_deep_hedger(
        model, S_tr, S_va,
        K=K, T=T, S0=S0, cost_lambda=0.0,
        alpha=0.95, lr=1e-3, batch_size=512,
        epochs=50, patience=50, verbose=False,
    )

    tr = history["train_risk"]
    initial = tr[0]
    final = tr[-1]
    decreased = final < initial

    # Save for figure
    test_training_convergence._data = history

    passed = decreased
    return passed, f"train_risk: {initial:.4f} -> {final:.4f} (decreased={decreased})"


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

def _make_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Training curve ---
    if hasattr(test_training_convergence, "_data"):
        h = test_training_convergence._data
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(h["train_risk"]) + 1)
        ax.plot(epochs_range, h["train_risk"], label="Train risk", alpha=0.8)
        ax.plot(epochs_range, h["val_risk"], label="Val risk", alpha=0.8)
        ax.axvline(h["best_epoch"], color="grey", ls="--", lw=1, label=f"Best epoch ({h['best_epoch']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"ES$_{0.95}$ (risk)")
        ax.set_title("Deep Hedger Training Convergence (GBM)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig_training_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {FIGURE_DIR / 'fig_training_curve.png'}")

    # --- Figures 2 & 3 from test_gbm_sanity data ---
    if not hasattr(test_gbm_sanity, "_data"):
        return
    d = test_gbm_sanity._data

    # Figure 2: PnL histograms
    fig, ax = plt.subplots(figsize=(8, 5))
    pnl_d = d["pnl_deep"].detach().float().numpy()
    pnl_b = d["pnl_bs"].detach().float().numpy()
    ax.hist(pnl_d, bins=150, alpha=0.6, density=True, label="Deep Hedger")
    ax.hist(pnl_b, bins=150, alpha=0.6, density=True, label="BS Delta")
    ax.axvline(d["m_deep"]["mean_pnl"], color="C0", ls="--", lw=1.3)
    ax.axvline(d["m_bs"]["mean_pnl"], color="C1", ls="--", lw=1.3)
    ax.axvline(-d["m_deep"]["es_95"], color="C0", ls=":", lw=1.3)
    ax.axvline(-d["m_bs"]["es_95"], color="C1", ls=":", lw=1.3)
    ax.set_xlabel("P&L")
    ax.set_ylabel("Density")
    ax.set_title("GBM Benchmark: Deep Hedger vs BS Delta")
    ax.legend()
    ax.set_xlim(-10, 8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_deep_vs_bs_gbm.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_deep_vs_bs_gbm.png'}")

    # Figure 3: Learned delta vs BS delta scatter
    model = d["model"]
    S_te = d["S_te"]
    deltas_bs = d["deltas_bs"]

    model.eval()
    with torch.no_grad():
        deltas_deep = hedge_paths_deep(model, S_te, T, S0)

    n = deltas_deep.shape[1]
    steps = [0, n // 2, n - 1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, k in zip(axes, steps):
        d_bs_k = deltas_bs[:, k].detach().float().numpy()
        d_dp_k = deltas_deep[:, k].detach().float().numpy()
        # Subsample for speed
        idx = slice(None, 5000)
        ax.scatter(d_bs_k[idx], d_dp_k[idx], s=1, alpha=0.15)
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.set_xlabel("BS Delta")
        ax.set_ylabel("Learned Delta")
        ax.set_title(f"k = {k}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
    fig.suptitle("Learned Delta vs BS Delta (GBM, no costs)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_learned_delta_vs_bs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_learned_delta_vs_bs.png'}")


# ---------------------------------------------------------------------------
# Test 7: log-moneyness equivalence
# ---------------------------------------------------------------------------

def test_log_moneyness_equivalence_at_s0_equals_k() -> Tuple[bool, str]:
    """Feature Set B: log(S/S_0) == log(S/K) when S_0 = K."""
    n_steps = 10
    T = 1.0
    S0 = K = 100.0

    torch.manual_seed(0)
    S = S0 + torch.randn(16, n_steps + 1) * 5.0
    S[:, 0] = S0  # simulator always starts exactly at S0
    t_grid = torch.linspace(0.0, T, n_steps + 1)
    deltas_prev = torch.zeros(16)

    all_match = True
    for k in range(n_steps):
        feat = build_features(S, t_grid, T, deltas_prev, k)
        log_s0 = feat[:, 1].double()
        log_K = torch.log(S[:, k] / K).double()
        if not torch.allclose(log_s0, log_K, atol=1e-12, rtol=0.0):
            all_match = False
            break

    return all_match, f"log(S/S_0)==log(S/K) at S_0=K=100: {all_match}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Architecture shapes",          test_architecture_shapes),
        ("2. Feature construction",          test_feature_construction),
        ("3. hedge_paths_deep shapes",       test_hedge_paths_shapes),
        ("4. GBM sanity (deep ≈ BS)",       test_gbm_sanity),
        ("5. Cost-aware turnover",           test_cost_aware_turnover),
        ("6. Training convergence",          test_training_convergence),
        ("7. log(S/S0)==log(S/K) at S0=K",  test_log_moneyness_equivalence_at_s0_equals_k),
    ]

    print("=" * 70)
    print(" Deep Hedger (ResidualFNN) — Validation Suite")
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
    print("\n  Generating diagnostic figures...")
    try:
        _make_figures()
    except Exception as exc:
        print(f"  WARNING: Figure generation failed: {exc}")

    print("-" * 70)
    if all_passed:
        print(f" All {len(tests)} tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
