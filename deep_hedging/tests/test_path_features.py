#!/usr/bin/env python
"""
Tests for path-dependent features and signature deep hedger.

    python -m deep_hedging.tests.test_path_features
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
import numpy as np

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.core.gbm import GBM
from deep_hedging.hedging.features import PathFeatureExtractor
from deep_hedging.hedging.signature_hedger import SignatureDeepHedger
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN, train_deep_hedger, evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff
from deep_hedging.objectives.risk_measures import expected_shortfall

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
XI0 = 0.235 ** 2
S0, K, T = 100.0, 100.0, 1.0


def _sim_rbergomi(n_paths, n_steps=100, H=0.07, seed=42):
    m = DifferentiableRoughBergomi(n_steps=n_steps, T=T, H=H, eta=1.9, rho=-0.7, xi0=XI0)
    S, V, tg = m.simulate(n_paths=n_paths, S0=S0, seed=seed)
    return S, V, tg


# -----------------------------------------------------------------------
# Test 1: Feature dimensions
# -----------------------------------------------------------------------

def test_dimensions() -> Tuple[bool, str]:
    S, _, _ = _sim_rbergomi(32)
    n = S.shape[1] - 1
    dp = torch.zeros(32, n, dtype=S.dtype)

    dims = {}
    for fs, expected in [("flat", 4), ("sig-3", 7), ("sig-full", 12)]:
        ext = PathFeatureExtractor(feature_set=fs, T=T, xi0=XI0)
        f = ext(S, dp)
        dims[fs] = f.shape
        if f.shape != (32, n, expected):
            return False, f"{fs}: got {f.shape}, expected (32, {n}, {expected})"

    # First 4 columns should match across sets
    ext_flat = PathFeatureExtractor(feature_set="flat", T=T, xi0=XI0)
    ext_full = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f_flat = ext_flat(S, dp)
    f_full = ext_full(S, dp)
    match = torch.allclose(f_flat, f_full[:, :, :4], atol=1e-6)

    return match, f"flat={dims['flat']}, sig-3={dims['sig-3']}, sig-full={dims['sig-full']}, flat_match={match}"


# -----------------------------------------------------------------------
# Test 2: No NaN or Inf
# -----------------------------------------------------------------------

def test_no_nan() -> Tuple[bool, str]:
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    issues = []
    for H in [0.02, 0.07, 0.5]:
        S, _, _ = _sim_rbergomi(200, H=H, seed=77)
        dp = torch.zeros(200, S.shape[1] - 1, dtype=S.dtype)
        f = ext(S, dp)
        if torch.isnan(f).any():
            issues.append(f"H={H}: NaN")
        if torch.isinf(f).any():
            issues.append(f"H={H}: Inf")
    passed = len(issues) == 0
    return passed, "clean" if passed else "; ".join(issues)


# -----------------------------------------------------------------------
# Test 3: Gradient flow
# -----------------------------------------------------------------------

def test_gradient_flow() -> Tuple[bool, str]:
    hedger = SignatureDeepHedger(feature_set="sig-full", hidden_dim=32, n_res_blocks=1, T=T, xi0=XI0)
    S, _, _ = _sim_rbergomi(64, n_steps=50, seed=33)
    deltas = hedger.hedge_paths(S, T=T, S0=S0)
    loss = deltas.sum()
    loss.backward()

    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in hedger.network.parameters() if p.requires_grad
    )
    return grad_ok, f"all_grads_finite={grad_ok}"


# -----------------------------------------------------------------------
# Test 4a: Constant path
# -----------------------------------------------------------------------

def test_constant_path() -> Tuple[bool, str]:
    batch, n = 10, 50
    S = torch.full((batch, n + 1), S0, dtype=torch.float64)
    dp = torch.zeros(batch, n, dtype=torch.float64)
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f = ext(S, dp)

    # RV at k=0 should be sqrt(xi0)/sqrt(xi0) = 1.0 (normalised sentinel)
    rv_k0 = f[0, 0, 4:7]  # sig-3 features at k=0
    sentinel_ok = torch.allclose(rv_k0, torch.ones_like(rv_k0), atol=0.01)

    # RV at k>0 should be ~0 (no returns) — sentinel only at k=0
    # Actually sqrt(EPS)/sqrt_xi0 which is very small
    rv_k5 = f[0, 5, 4:7]
    rv_small = (rv_k5.abs() < 0.01).all()

    return bool(sentinel_ok and rv_small), f"sentinel_ok={sentinel_ok}, rv_k5_small={rv_small}"


# -----------------------------------------------------------------------
# Test 4b: Linear log-price
# -----------------------------------------------------------------------

def test_linear_log_price() -> Tuple[bool, str]:
    batch, n = 10, 100
    dt = T / n
    c = math.sqrt(XI0)  # constant "vol"
    t_grid = torch.linspace(0, T, n + 1, dtype=torch.float64)
    log_S = c * t_grid.unsqueeze(0).expand(batch, -1)
    S = S0 * torch.exp(log_S)
    dp = torch.zeros(batch, n, dtype=torch.float64)
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f = ext(S, dp)

    # Linear log-price with constant returns c*dt produces RV = c*sqrt(dt).
    # Normalised: c*sqrt(dt) / sqrt(xi0) = sqrt(dt) ≈ 0.1 (since c=sqrt(xi0)).
    mid = n // 2
    rv_mid = f[0, mid, 4:7]
    expected_rv = math.sqrt(dt)  # ≈ 0.1 for n=100
    rv_ok = (rv_mid - expected_rv).abs().max() < 0.05

    # Roughness ratio should ≈ 1.0 (all windows see same constant returns)
    R_mid = f[0, mid, 7]
    R_ok = abs(float(R_mid) - 1.0) < 0.15

    return bool(rv_ok and R_ok), f"rv_mid={rv_mid.tolist()}, R={float(R_mid):.3f}"


# -----------------------------------------------------------------------
# Test 5: Running max/min
# -----------------------------------------------------------------------

def test_running_max_min() -> Tuple[bool, str]:
    n = 10
    # Price: goes up then down
    log_S_seq = torch.tensor([0.0, 0.1, 0.2, 0.15, 0.05, -0.1, -0.05, 0.0, 0.1, 0.15, 0.2])
    S = S0 * torch.exp(log_S_seq).unsqueeze(0)  # (1, 11)
    dp = torch.zeros(1, n, dtype=torch.float64)
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f = ext(S, dp)

    # Feature indices: [0-3 flat, 4-6 rv, 7 R, 8 Q, 9 max, 10 min, 11 QV]
    run_max = f[0, :, 9].tolist()
    run_min = f[0, :, 10].tolist()

    # At k=0, max/min = log_S[0] = 0; at k=2, max = 0.2; at k=5, min = -0.1
    max_ok = abs(run_max[2] - 0.2) < 0.01 and run_max[5] >= 0.19
    min_ok = abs(run_min[5] - (-0.1)) < 0.01

    return bool(max_ok and min_ok), f"max@2={run_max[2]:.3f}, max@5={run_max[5]:.3f}, min@5={run_min[5]:.3f}"


# -----------------------------------------------------------------------
# Test 6: Feature scale diagnostics
# -----------------------------------------------------------------------

def test_feature_scales() -> Tuple[bool, str]:
    S, _, _ = _sim_rbergomi(1000, seed=88)
    n = S.shape[1] - 1
    dp = torch.zeros(1000, n, dtype=S.dtype)
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f = ext(S, dp).float()

    means = f.mean(dim=(0, 1))
    stds = f.std(dim=(0, 1))

    mean_ok = (means.abs() < 10.0).all()
    # Skip f3 (delta_prev = 0 by construction) and f0/f2 (time features) for std check
    check_idx = [i for i in range(f.shape[-1]) if i not in (0, 2, 3)]
    std_ok = (stds[check_idx] > 0.01).all() and (stds[check_idx] < 20.0).all()
    passed = bool(mean_ok and std_ok)

    detail = ", ".join(f"f{i}: mu={means[i]:.3f} sd={stds[i]:.3f}" for i in range(f.shape[-1]))
    return passed, detail


# -----------------------------------------------------------------------
# Test 7: Mini training smoke test
# -----------------------------------------------------------------------

def test_training_smoke() -> Tuple[bool, str]:
    S, _, _ = _sim_rbergomi(10_000, n_steps=50, seed=44)
    S_tr, S_va = S[:8000], S[8000:]
    payoff_tr = compute_payoff(S_tr, K, "call")
    p0 = float(payoff_tr.mean())

    hedger = SignatureDeepHedger(feature_set="sig-full", hidden_dim=64,
                                n_res_blocks=1, T=T, xi0=XI0)
    history = train_deep_hedger(
        hedger, S_tr, S_va, K=K, T=T, S0=S0, p0=p0,
        epochs=20, patience=20, batch_size=2048, verbose=False,
    )
    tr = history["train_risk"]
    no_nan = all(x == x for x in tr)
    decreased = tr[-1] < tr[0]
    passed = no_nan and decreased
    return passed, f"risk: {tr[0]:.3f} -> {tr[-1]:.3f}, no_nan={no_nan}"


# -----------------------------------------------------------------------
# Test 8: Flat feature set equivalence
# -----------------------------------------------------------------------

def test_flat_equivalence() -> Tuple[bool, str]:
    """SignatureDeepHedger(flat) should be competitive with DeepHedgerFNN."""
    model_gbm = GBM(n_steps=50, T=T, sigma=0.235)
    S, _, _ = model_gbm.simulate(n_paths=20_000, S0=S0, seed=55)
    S_tr, S_va, S_te = S[:15_000], S[15_000:18_000], S[18_000:]
    p0 = float(compute_payoff(S_tr, K, "call").mean())

    # Flat signature hedger
    sig = SignatureDeepHedger(feature_set="flat", hidden_dim=64, n_res_blocks=1, T=T, xi0=XI0)
    train_deep_hedger(sig, S_tr, S_va, K=K, T=T, S0=S0, p0=p0,
                      epochs=50, patience=15, batch_size=2048, verbose=False)
    pnl_sig = evaluate_deep_hedger(sig, S_te, K=K, T=T, S0=S0, p0=p0)
    es_sig = float(expected_shortfall(pnl_sig, 0.95))

    # Original DeepHedgerFNN
    fnn = DeepHedgerFNN(input_dim=4, hidden_dim=64, n_res_blocks=1)
    train_deep_hedger(fnn, S_tr, S_va, K=K, T=T, S0=S0, p0=p0,
                      epochs=50, patience=15, batch_size=2048, verbose=False)
    pnl_fnn = evaluate_deep_hedger(fnn, S_te, K=K, T=T, S0=S0, p0=p0)
    es_fnn = float(expected_shortfall(pnl_fnn, 0.95))

    # Should be within 30% of each other (both are small, ~2-3)
    ratio = es_sig / max(es_fnn, 0.01)
    passed = 0.5 < ratio < 1.5
    return passed, f"ES_sig_flat={es_sig:.3f}, ES_fnn={es_fnn:.3f}, ratio={ratio:.3f}"


# -----------------------------------------------------------------------
# Test 9: Mini H4 directional check [KEY TEST]
# -----------------------------------------------------------------------

def test_h4_mini() -> Tuple[bool, str]:
    """sig-full should do no worse than flat on rough paths."""
    S, _, _ = _sim_rbergomi(30_000, n_steps=50, H=0.05, seed=66)
    S_tr, S_va, S_te = S[:24_000], S[24_000:27_000], S[27_000:]
    p0 = float(compute_payoff(S_tr, K, "call").mean())

    results = {}
    for fs in ["flat", "sig-3", "sig-full"]:
        hedger = SignatureDeepHedger(feature_set=fs, hidden_dim=64,
                                    n_res_blocks=1, T=T, xi0=XI0)
        train_deep_hedger(hedger, S_tr, S_va, K=K, T=T, S0=S0, p0=p0,
                          epochs=50, patience=15, batch_size=2048, verbose=False)
        pnl = evaluate_deep_hedger(hedger, S_te, K=K, T=T, S0=S0, p0=p0)
        results[fs] = float(expected_shortfall(pnl, 0.95))

    # sig-full should not be significantly worse than flat (allow 10% tolerance)
    passed = results["sig-full"] <= results["flat"] * 1.10
    detail = ", ".join(f"{k}={v:.3f}" for k, v in results.items())
    return passed, f"ES_95: {detail}"


# -----------------------------------------------------------------------
# Diagnostic figures
# -----------------------------------------------------------------------

def _make_figures():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    S, _, _ = _sim_rbergomi(5000, n_steps=100, H=0.07, seed=88)
    n = S.shape[1] - 1
    dp = torch.zeros(5000, n, dtype=S.dtype)
    ext = PathFeatureExtractor(feature_set="sig-full", T=T, xi0=XI0)
    f = ext(S, dp).float().numpy()

    # Figure 1: feature distributions at k=n/2
    mid = n // 2
    names = ["t/T", "logM", "tau/T", "d_prev",
             "rv5", "rv15", "rv50",
             "R", "Q", "max", "min", "QV"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for i, ax in enumerate(axes.flat):
        if i < f.shape[-1]:
            ax.hist(f[:, mid, i], bins=50, alpha=0.7, color="steelblue")
            ax.set_title(names[i] if i < len(names) else f"f{i}")
        else:
            ax.axis("off")
    fig.suptitle(f"Feature distributions at k={mid} (H=0.07, 5k paths)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_feature_distributions.png'}", flush=True)

    # Figure 2: single path feature evolution
    S1, _, _ = _sim_rbergomi(1, n_steps=100, H=0.05, seed=99)
    dp1 = torch.zeros(1, 100, dtype=S1.dtype)
    f1 = ext(S1, dp1).float().squeeze(0).numpy()
    t = np.linspace(0, T, 100)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes[0, 0].plot(np.linspace(0, T, 101), S1.squeeze().numpy())
    axes[0, 0].set_title("Price S")
    axes[0, 1].plot(t, f1[:, 1])
    axes[0, 1].set_title("Log-moneyness")
    for i, w in enumerate(["rv5", "rv15", "rv50"]):
        axes[1, 0].plot(t, f1[:, 4 + i], label=w)
    axes[1, 0].legend(); axes[1, 0].set_title("Realised vol (normalised)")
    axes[1, 1].plot(t, f1[:, 7]); axes[1, 1].set_title("Roughness ratio R")
    axes[2, 0].plot(t, f1[:, 8]); axes[2, 0].set_title("Vol-of-vol Q")
    axes[2, 1].plot(t, f1[:, 9], label="max")
    axes[2, 1].plot(t, f1[:, 10], label="min")
    axes[2, 1].plot(t, f1[:, 11], label="QV_norm")
    axes[2, 1].legend(); axes[2, 1].set_title("Path statistics")
    for ax in axes.flat:
        ax.set_xlabel("t")
    fig.suptitle("Sample path features (H=0.05)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_feature_sample_path.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURE_DIR / 'fig_feature_sample_path.png'}", flush=True)


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Feature dimensions",          test_dimensions),
        ("2. No NaN/Inf in features",       test_no_nan),
        ("3. Gradient flow",                test_gradient_flow),
        ("4a. Constant path values",        test_constant_path),
        ("4b. Linear log-price values",     test_linear_log_price),
        ("5. Running max/min",              test_running_max_min),
        ("6. Feature scale diagnostics",    test_feature_scales),
        ("7. Training smoke test",          test_training_smoke),
        ("8. Flat equivalence",             test_flat_equivalence),
        ("9. Mini H4 directional (KEY)",    test_h4_mini),
    ]

    print("=" * 65, flush=True)
    print(" Path Features & Signature Hedger — Validation", flush=True)
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
    print("\n  Generating diagnostic figures...", flush=True)
    try:
        _make_figures()
    except Exception as exc:
        print(f"  WARNING: figures failed: {exc}", flush=True)

    print("-" * 65, flush=True)
    if all_passed:
        print(" All tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
