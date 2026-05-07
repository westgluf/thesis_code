#!/usr/bin/env python
"""
P02 — Exact Cholesky Validation + Static Arbitrage + Kappa-2 Variant.

Three independent validation checks for the hybrid BLP simulator:
1. Distributional comparison of hybrid vs exact Cholesky fBm at n=100.
2. Static arbitrage verification on MC-priced call surface at n=100, n=400.
3. κ=1 vs κ=2 near-field comparison at n=100.

Run:
    python -u -m deep_hedging.experiments.block1_cholesky_arbitrage_kappa --preflight-only
    python -u -m deep_hedging.experiments.block1_cholesky_arbitrage_kappa
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats as sp_stats
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.core.volterra import HybridVolterraDriver
from deep_hedging.core.volterra_exact import ExactCholeskyVolterra
from deep_hedging.core.volterra_kappa2 import HybridVolterraDriverKappa2
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import expected_shortfall
from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

# ─── Constants ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "block1"
FIGURES_DIR = REPO_ROOT / "figures" / "block1"
BASELINE = {"H": 0.07, "eta": 1.9, "rho": -0.7, "xi0": 0.235**2,
            "S0": 100.0, "K": 100.0, "T": 1.0, "sigma_bs": 0.235}

C_EXACT = "#4CAF50"
C_HYBRID = "#2196F3"
C_KAPPA2 = "#FF9800"

OUTPUT_FILES_R = ["cholesky_arbitrage_kappa.json", "p02_readme.md"]
OUTPUT_FILES_F = ["hybrid_vs_exact_terminal.png", "hybrid_vs_exact_qq.png",
                  "arbitrage_violation_heatmap.png", "kappa1_vs_kappa2_comparison.png"]


def _check_no_collision() -> None:
    for f in OUTPUT_FILES_R:
        p = RESULTS_DIR / f
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite: {p}")
    for f in OUTPUT_FILES_F:
        p = FIGURES_DIR / f
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite: {p}")


# =====================================================================
# Part 1: Hybrid vs Exact Cholesky
# =====================================================================

def run_cholesky_comparison(n: int = 100, n_paths: int = 2000,
                             seed_hybrid: int = 8001, seed_exact: int = 8002,
                             ) -> dict[str, Any]:
    """Compare hybrid and exact Cholesky fBm at distributional level."""
    print("\n  [Part 1] Hybrid vs Exact Cholesky comparison", flush=True)
    H, T = BASELINE["H"], BASELINE["T"]

    # --- Generate fBm via both methods ---
    hybrid_driver = HybridVolterraDriver(n, T, H)
    exact_driver = ExactCholeskyVolterra(n, T, H)

    gen_h = torch.Generator().manual_seed(seed_hybrid)
    Z_hybrid = torch.randn(n_paths, n, 2, dtype=torch.float64, generator=gen_h)
    WH_hybrid, dW_hybrid = hybrid_driver(Z_hybrid)

    WH_exact, dWH_exact = exact_driver.simulate(n_paths, seed=seed_exact)

    # --- Terminal WH comparison ---
    wh_term_h = WH_hybrid[:, -1].numpy()
    wh_term_e = WH_exact[:, -1].numpy()

    ks_stat, ks_pval = sp_stats.ks_2samp(wh_term_h, wh_term_e)
    w1_terminal = wasserstein_distance(wh_term_h, wh_term_e)

    terminal_comp = {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "wasserstein_1": float(w1_terminal),
        "moments": {
            "mean_exact": float(np.mean(wh_term_e)),
            "mean_hybrid": float(np.mean(wh_term_h)),
            "std_exact": float(np.std(wh_term_e)),
            "std_hybrid": float(np.std(wh_term_h)),
            "skew_exact": float(sp_stats.skew(wh_term_e)),
            "skew_hybrid": float(sp_stats.skew(wh_term_h)),
            "kurt_exact": float(sp_stats.kurtosis(wh_term_e)),
            "kurt_hybrid": float(sp_stats.kurtosis(wh_term_h)),
        },
    }
    print(f"    Terminal WH: KS p={ks_pval:.4f}, W1={w1_terminal:.6f}", flush=True)

    # --- One-step increment comparison ---
    dWH_hybrid = WH_hybrid[:, 1:] - WH_hybrid[:, :-1]  # [n_paths, n]
    dWH_h_np = dWH_hybrid.numpy()
    dWH_e_np = dWH_exact.numpy()

    ks_pvals_steps = []
    w1_steps = []
    for k in range(n):
        ks_s, ks_p = sp_stats.ks_2samp(dWH_h_np[:, k], dWH_e_np[:, k])
        w1_k = wasserstein_distance(dWH_h_np[:, k], dWH_e_np[:, k])
        ks_pvals_steps.append(float(ks_p))
        w1_steps.append(float(w1_k))

    ks_arr = np.array(ks_pvals_steps)
    worst_3_idx = np.argsort(ks_arr)[:3]
    increment_comp = {
        "ks_pvalues_summary": {
            "min": float(ks_arr.min()),
            "max": float(ks_arr.max()),
            "median": float(np.median(ks_arr)),
            "fraction_below_0.01": float(np.mean(ks_arr < 0.01)),
            "worst_3_steps": [int(i) for i in worst_3_idx],
            "worst_3_pvalues": [float(ks_arr[i]) for i in worst_3_idx],
        },
        "avg_wasserstein": float(np.mean(w1_steps)),
    }
    print(f"    Increments: median KS p={np.median(ks_arr):.4f}, "
          f"fraction<0.01={np.mean(ks_arr<0.01):.3f}", flush=True)

    # --- Variance path match (via rBergomi) ---
    eta, rho, xi0 = BASELINE["eta"], BASELINE["rho"], BASELINE["xi0"]
    t_grid = torch.linspace(0, T, n + 1, dtype=torch.float64)
    t_2H = t_grid ** (2.0 * H)

    V_hybrid = xi0 * torch.exp(eta * WH_hybrid - 0.5 * eta**2 * t_2H.unsqueeze(0))
    V_exact = xi0 * torch.exp(eta * WH_exact - 0.5 * eta**2 * t_2H.unsqueeze(0))

    check_times = [0.25, 0.5, 0.75, 1.0]
    E_v_exact, E_v_hybrid, rel_diffs_v = [], [], []
    for t_val in check_times:
        idx = int(t_val * n)
        ev_e = float(V_exact[:, idx].mean())
        ev_h = float(V_hybrid[:, idx].mean())
        rd = abs(ev_e - ev_h) / (abs(ev_e) + 1e-12)
        E_v_exact.append(ev_e)
        E_v_hybrid.append(ev_h)
        rel_diffs_v.append(rd)

    variance_match = {
        "t_values": check_times,
        "E_v_exact": E_v_exact,
        "E_v_hybrid": E_v_hybrid,
        "rel_differences": rel_diffs_v,
    }
    print(f"    Variance match: max rel diff = {max(rel_diffs_v):.4f}", flush=True)

    # --- Call price comparison ---
    n_price = 20000
    sim_h = DifferentiableRoughBergomi(n_steps=n, T=T, H=H, eta=eta, rho=rho, xi0=xi0)
    S_h, _, _ = sim_h.simulate(n_price, S0=BASELINE["S0"], seed=seed_hybrid)
    payoff_h = compute_payoff(S_h, BASELINE["K"], "call")
    price_h = float(payoff_h.mean())

    # Exact Cholesky rBergomi: generate WH then build S manually
    WH_ex_price, _ = exact_driver.simulate(n_price, seed=seed_exact)
    # Need independent Brownian for price
    gen_p = torch.Generator().manual_seed(seed_exact + 100)
    Z_price_ex = torch.randn(n_price, n, dtype=torch.float64, generator=gen_p)
    dW_ex = torch.sqrt(torch.tensor(T / n, dtype=torch.float64)) * Z_price_ex

    V_ex_price = xi0 * torch.exp(eta * WH_ex_price - 0.5 * eta**2 * t_2H.unsqueeze(0))
    V_hedge = torch.clamp(V_ex_price[:, :-1], min=1e-12)
    sqrt_V = torch.sqrt(V_hedge)
    dt_val = T / n
    rho_perp = np.sqrt(1 - rho**2)
    gen_p2 = torch.Generator().manual_seed(seed_exact + 200)
    Z_orth = torch.randn(n_price, n, dtype=torch.float64, generator=gen_p2)
    dB = rho * dW_ex + rho_perp * np.sqrt(dt_val) * Z_orth
    log_inc = -0.5 * V_hedge * dt_val + sqrt_V * dB
    log_S = torch.cumsum(log_inc, dim=-1)
    log_S = torch.cat([torch.zeros(n_price, 1, dtype=torch.float64), log_S], dim=-1)
    S_ex = BASELINE["S0"] * torch.exp(log_S)
    payoff_ex = compute_payoff(S_ex, BASELINE["K"], "call")
    price_ex = float(payoff_ex.mean())

    price_rel_diff = abs(price_h - price_ex) / (abs(price_ex) + 1e-12)
    call_price = {
        "price_exact": price_ex,
        "price_hybrid": price_h,
        "rel_diff": float(price_rel_diff),
    }
    print(f"    Call prices: exact={price_ex:.4f}, hybrid={price_h:.4f}, "
          f"rel_diff={price_rel_diff:.4f}", flush=True)

    return {
        "n": n, "n_paths": n_paths,
        "terminal_WH_comparison": terminal_comp,
        "one_step_increment_comparison": increment_comp,
        "variance_path_match": variance_match,
        "call_price_comparison": call_price,
        "_wh_term_h": wh_term_h, "_wh_term_e": wh_term_e,  # for plotting
    }


# =====================================================================
# Part 2: Static Arbitrage Check
# =====================================================================

def run_arbitrage_check(n_steps_list: list[int] = [100, 400],
                         n_paths: int = 50000,
                         seed_base: int = 8003,
                         ) -> dict[str, Any]:
    """Check static arbitrage conditions on MC call surface."""
    print("\n  [Part 2] Static arbitrage check", flush=True)
    H, eta, rho, xi0 = BASELINE["H"], BASELINE["eta"], BASELINE["rho"], BASELINE["xi0"]
    S0, K_base = BASELINE["S0"], BASELINE["K"]

    K_grid = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    K_abs = [k * S0 for k in K_grid]
    T_grid = [0.25, 0.50, 1.00]

    results = {}
    for n in n_steps_list:
        print(f"    n={n}:", flush=True)
        prices = []
        mc_stds = []
        for T_val in T_grid:
            sim = DifferentiableRoughBergomi(n_steps=n, T=T_val, H=H, eta=eta, rho=rho, xi0=xi0)
            seed_t = seed_base + int(T_val * 100) + n
            S, _, _ = sim.simulate(n_paths, S0=S0, seed=seed_t)
            row_prices = []
            row_stds = []
            for K_val in K_abs:
                payoff = torch.relu(S[:, -1] - K_val)
                price = float(payoff.mean())
                std = float(payoff.std() / np.sqrt(n_paths))
                row_prices.append(price)
                row_stds.append(std)
            prices.append(row_prices)
            mc_stds.append(row_stds)
            print(f"      T={T_val}: prices = {[f'{p:.2f}' for p in row_prices]}", flush=True)

        # Check monotonicity in K (at each T)
        mono_violations = []
        for t_idx, T_val in enumerate(T_grid):
            for k_idx in range(len(K_abs) - 1):
                if prices[t_idx][k_idx] < prices[t_idx][k_idx + 1]:
                    diff = prices[t_idx][k_idx + 1] - prices[t_idx][k_idx]
                    threshold = max(0.01 * prices[t_idx][k_idx], 2 * mc_stds[t_idx][k_idx])
                    if diff > threshold:
                        mono_violations.append({
                            "T": T_val, "K_lo": K_abs[k_idx], "K_hi": K_abs[k_idx + 1],
                            "price_lo": prices[t_idx][k_idx], "price_hi": prices[t_idx][k_idx + 1],
                            "diff": diff, "threshold": threshold,
                        })

        # Check convexity (butterfly)
        conv_violations = []
        for t_idx, T_val in enumerate(T_grid):
            for k_idx in range(1, len(K_abs) - 1):
                butterfly = prices[t_idx][k_idx - 1] - 2 * prices[t_idx][k_idx] + prices[t_idx][k_idx + 1]
                if butterfly < 0:
                    threshold = max(0.01 * prices[t_idx][k_idx], 2 * mc_stds[t_idx][k_idx])
                    if abs(butterfly) > threshold:
                        conv_violations.append({
                            "T": T_val, "K": K_abs[k_idx],
                            "butterfly": butterfly, "threshold": threshold,
                        })

        # Check calendar monotonicity
        cal_violations = []
        for k_idx in range(len(K_abs)):
            for t_idx in range(len(T_grid) - 1):
                if prices[t_idx + 1][k_idx] < prices[t_idx][k_idx]:
                    diff = prices[t_idx][k_idx] - prices[t_idx + 1][k_idx]
                    threshold = max(0.01 * prices[t_idx][k_idx],
                                     2 * max(mc_stds[t_idx][k_idx], mc_stds[t_idx + 1][k_idx]))
                    if diff > threshold:
                        cal_violations.append({
                            "T_lo": T_grid[t_idx], "T_hi": T_grid[t_idx + 1],
                            "K": K_abs[k_idx],
                            "price_lo_T": prices[t_idx][k_idx],
                            "price_hi_T": prices[t_idx + 1][k_idx],
                        })

        total = len(mono_violations) + len(conv_violations) + len(cal_violations)
        verdict = "PASS" if total == 0 else "FAIL"
        print(f"      Violations: mono={len(mono_violations)}, conv={len(conv_violations)}, "
              f"cal={len(cal_violations)} → {verdict}", flush=True)

        results[f"n{n}"] = {
            "K_grid": K_abs, "T_grid": T_grid,
            "prices": prices, "mc_std": mc_stds,
            "monotonicity_violations_K": mono_violations,
            "convexity_violations_K": conv_violations,
            "monotonicity_violations_T": cal_violations,
            "total_violations": total,
            "verdict": verdict,
        }

    return results


# =====================================================================
# Part 3: Kappa-1 vs Kappa-2
# =====================================================================

def run_kappa_comparison(n: int = 100, n_paths: int = 10000,
                          seed: int = 8101) -> dict[str, Any]:
    """Compare κ=1 and κ=2 hybrid schemes via BS-hedge ES."""
    print("\n  [Part 3] κ=1 vs κ=2 comparison", flush=True)
    H, T = BASELINE["H"], BASELINE["T"]
    eta, rho, xi0 = BASELINE["eta"], BASELINE["rho"], BASELINE["xi0"]
    S0, K = BASELINE["S0"], BASELINE["K"]

    # --- κ=1 ---
    sim1 = DifferentiableRoughBergomi(n_steps=n, T=T, H=H, eta=eta, rho=rho, xi0=xi0)
    S1, _, _ = sim1.simulate(n_paths, S0=S0, seed=seed)
    hedger = BlackScholesDelta(sigma=BASELINE["sigma_bs"], K=K, T=T)
    deltas1 = hedger.hedge_paths(S1)
    payoff1 = compute_payoff(S1, K, "call")
    p0_1 = float(payoff1.mean())
    pnl1 = compute_hedging_pnl(S1, deltas1, payoff1, p0_1, 0.0)
    es1 = float(expected_shortfall(pnl1.float(), 0.95))

    # --- κ=2: build rBergomi manually ---
    driver_k2 = HybridVolterraDriverKappa2(n, T, H)
    gen = torch.Generator().manual_seed(seed)
    Z_k2 = torch.randn(n_paths, n, 3, dtype=torch.float64, generator=gen)
    WH_k2, dW_k2 = driver_k2(Z_k2)

    # Price noise (independent)
    gen2 = torch.Generator().manual_seed(seed + 1000)
    Z_price = torch.randn(n_paths, n, dtype=torch.float64, generator=gen2)

    t_grid = torch.linspace(0, T, n + 1, dtype=torch.float64)
    t_2H = t_grid ** (2.0 * H)
    V_k2 = xi0 * torch.exp(eta * WH_k2 - 0.5 * eta**2 * t_2H.unsqueeze(0))
    V_hedge = torch.clamp(V_k2[:, :-1], min=1e-12)
    sqrt_V = torch.sqrt(V_hedge)
    dt_val = T / n
    rho_perp = np.sqrt(1 - rho**2)
    dB = rho * dW_k2 + rho_perp * np.sqrt(dt_val) * Z_price
    log_inc = -0.5 * V_hedge * dt_val + sqrt_V * dB
    log_S = torch.cumsum(log_inc, dim=-1)
    log_S = torch.cat([torch.zeros(n_paths, 1, dtype=torch.float64), log_S], dim=-1)
    S2 = S0 * torch.exp(log_S)

    deltas2 = hedger.hedge_paths(S2)
    payoff2 = compute_payoff(S2, K, "call")
    p0_2 = float(payoff2.mean())
    pnl2 = compute_hedging_pnl(S2, deltas2, payoff2, p0_2, 0.0)
    es2 = float(expected_shortfall(pnl2.float(), 0.95))

    diff_abs = es2 - es1
    diff_rel = diff_abs / (abs(es1) + 1e-12)
    if abs(diff_rel) < 0.01:
        verdict = "NEGLIGIBLE (<1%)"
    elif abs(diff_rel) < 0.02:
        verdict = "ACCEPTABLE (<2%)"
    elif abs(diff_rel) < 0.05:
        verdict = "NOTABLE (2-5%)"
    else:
        verdict = "SIGNIFICANT (>5%)"

    print(f"    κ=1 ES_95={es1:.4f}, κ=2 ES_95={es2:.4f}, "
          f"diff={diff_abs:+.4f} ({diff_rel:+.1%}) → {verdict}", flush=True)

    return {
        "n_paths": n_paths,
        "es95_kappa1": es1,
        "es95_kappa2": es2,
        "diff_abs": diff_abs,
        "diff_rel": float(diff_rel),
        "verdict": verdict,
    }


# =====================================================================
# Figures
# =====================================================================

def _fig_style():
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300,
                          "savefig.bbox": "tight"})

def plot_terminal_histograms(wh_h, wh_e, out):
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(wh_h.min(), wh_e.min()), max(wh_h.max(), wh_e.max()), 60)
    ax.hist(wh_h, bins=bins, alpha=0.5, color=C_HYBRID, label="Hybrid", density=True)
    ax.hist(wh_e, bins=bins, alpha=0.5, color=C_EXACT, label="Exact Cholesky", density=True)
    ax.set_xlabel("$W^H(T)$")
    ax.set_ylabel("Density")
    ax.set_title("Terminal fBm: Hybrid vs Exact Cholesky")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

def plot_qq(wh_h, wh_e, out):
    _fig_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    q = np.linspace(0.5, 99.5, 200)
    qh = np.percentile(wh_h, q)
    qe = np.percentile(wh_e, q)
    ax.scatter(qe, qh, s=8, alpha=0.6, color=C_HYBRID)
    lims = [min(qe.min(), qh.min()), max(qe.max(), qh.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="y=x")
    ax.set_xlabel("Exact Cholesky quantiles")
    ax.set_ylabel("Hybrid quantiles")
    ax.set_title("QQ Plot: $W^H(T)$")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

def plot_arbitrage_heatmap(arb_results, out):
    _fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (n_key, data) in enumerate(arb_results.items()):
        ax = axes[ax_idx]
        prices = np.array(data["prices"])
        K_grid = data["K_grid"]
        T_grid = data["T_grid"]
        im = ax.imshow(prices, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(K_grid)))
        ax.set_xticklabels([f"{k:.0f}" for k in K_grid], rotation=45)
        ax.set_yticks(range(len(T_grid)))
        ax.set_yticklabels([f"{t:.2f}" for t in T_grid])
        ax.set_xlabel("Strike $K$")
        ax.set_ylabel("Maturity $T$")
        ax.set_title(f"MC Call Prices ({n_key})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        v = data["total_violations"]
        ax.annotate(f"Violations: {v}", (0.02, 0.02), xycoords="axes fraction",
                     fontsize=10, color="red" if v > 0 else "green",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Static Arbitrage Check — MC Call Surface", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

def plot_kappa_comparison(kappa_results, out):
    _fig_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    es1 = kappa_results["es95_kappa1"]
    es2 = kappa_results["es95_kappa2"]
    bars = ax.bar(["$\\kappa=1$ (2×2)", "$\\kappa=2$ (3×3)"],
                   [es1, es2], color=[C_HYBRID, C_KAPPA2], width=0.5,
                   edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, [es1, es2]):
        ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    diff = kappa_results["diff_rel"]
    ax.set_ylabel(r"$\mathrm{ES}_{0.95}$ (BS hedge)")
    ax.set_title(f"Near-Field Precision: $\\kappa=1$ vs $\\kappa=2$ "
                  f"(diff: {diff:+.2%})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)


# =====================================================================
# README
# =====================================================================

def generate_readme(output, out_path):
    ec = output["exact_cholesky_validation"]
    arb = output["arbitrage_check"]
    kap = output["kappa_variant"]
    v = output["global_verdict"]

    lines = [
        "# P02 — Simulator Validation Results\n",
        f"Generated: {output['meta']['created_at_utc']}",
        f"Git commit: {output['meta']['git_commit']}\n",
        f"## Global Verdict: **{v}**\n",
        "## 1. Hybrid vs Exact Cholesky (n=100, 2000 paths)\n",
        f"- Terminal WH KS p-value: **{ec['terminal_WH_comparison']['ks_pvalue']:.4f}** "
        f"(pass if >0.01)",
        f"- Terminal WH Wasserstein-1: **{ec['terminal_WH_comparison']['wasserstein_1']:.6f}**",
        f"- Increment KS: median p = {ec['one_step_increment_comparison']['ks_pvalues_summary']['median']:.4f}, "
        f"fraction<0.01 = {ec['one_step_increment_comparison']['ks_pvalues_summary']['fraction_below_0.01']:.3f}",
        f"- Variance path: max rel diff = {max(ec['variance_path_match']['rel_differences']):.4f}",
        f"- Call price: exact={ec['call_price_comparison']['price_exact']:.4f}, "
        f"hybrid={ec['call_price_comparison']['price_hybrid']:.4f}, "
        f"rel_diff={ec['call_price_comparison']['rel_diff']:.4f}\n",
        "## 2. Static Arbitrage Check\n",
    ]
    for nk, data in arb.items():
        lines.append(f"- {nk}: **{data['verdict']}** ({data['total_violations']} violations)")
    lines.extend([
        f"\n## 3. Kappa Variant (n=100, {kap['n_paths']} paths)\n",
        f"- κ=1 ES_0.95: **{kap['es95_kappa1']:.4f}**",
        f"- κ=2 ES_0.95: **{kap['es95_kappa2']:.4f}**",
        f"- Difference: {kap['diff_abs']:+.4f} ({kap['diff_rel']:+.2%})",
        f"- Verdict: **{kap['verdict']}**\n",
        "## Next Action\n",
    ])
    if v == "STRICT_PASS":
        lines.append("Hybrid scheme fully validated. No simulator concerns. Continue to Block 2.")
    elif v == "PASS_WITH_NOTES":
        lines.append("Hybrid scheme validated with minor notes. Include in Section 5.4. Continue.")
    else:
        lines.append("Hybrid scheme issues detected. Investigate before proceeding.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}", flush=True)


# =====================================================================
# Main
# =====================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _check_no_collision()

    hardware = record_hardware_and_versions()
    t_total = time.perf_counter()

    print("=" * 60)
    print(" P02 — Simulator Validation Suite")
    print("=" * 60, flush=True)

    # Part 1
    chol = run_cholesky_comparison()
    wh_h = chol.pop("_wh_term_h")
    wh_e = chol.pop("_wh_term_e")

    # Part 2
    arb = run_arbitrage_check()

    # Part 3
    kap = run_kappa_comparison()

    # Global verdict
    notes = []
    chol_ok = (chol["terminal_WH_comparison"]["ks_pvalue"] > 0.01 and
               chol["call_price_comparison"]["rel_diff"] < 0.02)
    arb_ok = all(d["total_violations"] == 0 for d in arb.values())
    kap_ok = abs(kap["diff_rel"]) < 0.02

    if chol_ok and arb_ok and kap_ok:
        verdict = "STRICT_PASS"
    elif not chol_ok:
        if chol["terminal_WH_comparison"]["ks_pvalue"] > 0.001:
            verdict = "PASS_WITH_NOTES"
            notes.append(f"Cholesky KS p-value borderline: {chol['terminal_WH_comparison']['ks_pvalue']:.4f}")
        else:
            verdict = "FAIL"
            notes.append("Cholesky comparison failed")
    elif not arb_ok:
        total_v = sum(d["total_violations"] for d in arb.values())
        if total_v <= 2:
            verdict = "PASS_WITH_NOTES"
            notes.append(f"Minor arbitrage violations: {total_v}")
        else:
            verdict = "FAIL"
            notes.append(f"Significant arbitrage violations: {total_v}")
    elif not kap_ok:
        if abs(kap["diff_rel"]) < 0.05:
            verdict = "PASS_WITH_NOTES"
            notes.append(f"κ=2 difference notable: {kap['diff_rel']:+.2%}")
        else:
            verdict = "FAIL"
            notes.append(f"κ=2 difference significant: {kap['diff_rel']:+.2%}")
    else:
        verdict = "PASS_WITH_NOTES"

    total_time = time.perf_counter() - t_total

    output = {
        "meta": {
            "script": "deep_hedging/experiments/block1_cholesky_arbitrage_kappa.py",
            "git_commit": hardware.get("git_commit", "unknown"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "total_wall_clock_s": round(total_time, 2),
        },
        "exact_cholesky_validation": chol,
        "arbitrage_check": arb,
        "kappa_variant": kap,
        "global_verdict": verdict,
        "notes": notes,
    }

    json_path = RESULTS_DIR / "cholesky_arbitrage_kappa.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {json_path}", flush=True)

    print("\n  Generating figures...", flush=True)
    plot_terminal_histograms(wh_h, wh_e, FIGURES_DIR / "hybrid_vs_exact_terminal.png")
    plot_qq(wh_h, wh_e, FIGURES_DIR / "hybrid_vs_exact_qq.png")
    plot_arbitrage_heatmap(arb, FIGURES_DIR / "arbitrage_violation_heatmap.png")
    plot_kappa_comparison(kap, FIGURES_DIR / "kappa1_vs_kappa2_comparison.png")

    generate_readme(output, RESULTS_DIR / "p02_readme.md")

    print(f"\n{'='*60}")
    print(f" VERDICT: {verdict}")
    if notes:
        for n in notes:
            print(f"  Note: {n}")
    print(f"{'='*60}", flush=True)

    return output


def run_preflight():
    p = RESULTS_DIR / "p02_preflight.md"
    if p.exists():
        print(f"Preflight exists: {p}")
    else:
        print("Preflight not found — should have been created before this script.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P02 — Simulator Validation")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        run_preflight()
    else:
        main()
