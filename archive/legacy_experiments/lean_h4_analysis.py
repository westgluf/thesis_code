#!/usr/bin/env python
"""
Analysis and figures for the lean H4 sweep.

Loads ``lean_h4_sweep_summary.json`` and the per-H PnL tensors, then
produces four publication figures plus a LaTeX table.

Run AFTER ``run_lean_h4_sweep.py`` has finished:

    python -u -m deep_hedging.experiments.lean_h4_analysis
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.objectives.risk_measures import expected_shortfall


# Colours
C_BS = "#2196F3"
C_FLAT = "#FF9800"
C_SIG3 = "#9C27B0"
C_SIGFULL = "#4CAF50"
C_FIT = "#F44336"

OUT_DIR = Path(__file__).resolve().parents[2] / "figures"
SUMMARY_PATH = OUT_DIR / "lean_h4_sweep_summary.json"


# =======================================================================
# Data loading
# =======================================================================

def load_summary() -> dict[float, dict]:
    """Load the per-H summary JSON and convert keys back to float."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} not found. Run run_lean_h4_sweep.py first.")

    with open(SUMMARY_PATH) as f:
        raw = json.load(f)

    return {float(k): v for k, v in raw.items()}


def load_pnl_tensors(H_values: list[float]) -> dict[float, dict]:
    """Load per-H PnL .pt files (may be missing for Stage 1 H=0.05)."""
    tensors: dict[float, dict] = {}
    for H in H_values:
        pt_path = OUT_DIR / f"lean_h4_H{H:.2f}_pnl.pt"
        if pt_path.exists():
            tensors[H] = torch.load(pt_path, weights_only=True)
        else:
            tensors[H] = {}  # e.g. Stage 1 H=0.05 — no tensors saved
    return tensors


# =======================================================================
# Bootstrap CIs for ES
# =======================================================================

def bootstrap_es95_ci(
    pnl: torch.Tensor, n_boot: int = 100, alpha: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Return (low, high) 95% CI for ES_95 via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = pnl.shape[0]
    vals = []
    p = pnl.detach().float().numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(expected_shortfall(torch.tensor(p[idx]), alpha=alpha)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =======================================================================
# Figures
# =======================================================================

def fig_trend(results: dict[float, dict], tensors: dict[float, dict]) -> None:
    """Figure A: Γ_flat, Γ_sig3, Γ_sig-full vs H (the key figure)."""
    H_sorted = sorted(results.keys())
    H_arr = np.array(H_sorted)
    g_flat = np.array([results[H]["gamma_flat"] for H in H_sorted])
    g_sig3 = np.array([results[H]["gamma_sig3"] for H in H_sorted])
    g_sigfull = np.array([results[H]["gamma_sigfull"] for H in H_sorted])

    # Bootstrap CIs using PnL tensors if available
    def ci_gamma(H: float, strategy_pnl_key: str) -> tuple[float, float] | None:
        t = tensors.get(H, {})
        if strategy_pnl_key not in t or "bs_pnl" not in t:
            return None
        bs = t["bs_pnl"]
        dh = t[strategy_pnl_key]
        rng = np.random.default_rng(42 + hash(strategy_pnl_key) % 1000)
        n = min(bs.shape[0], dh.shape[0])
        gammas = []
        for _ in range(100):
            idx = rng.integers(0, n, size=n)
            es_bs = float(expected_shortfall(bs[idx], 0.95))
            es_dh = float(expected_shortfall(dh[idx], 0.95))
            gammas.append(es_bs - es_dh)
        return float(np.percentile(gammas, 2.5)), float(np.percentile(gammas, 97.5))

    err_flat = [[], []]
    err_sig3 = [[], []]
    err_sigfull = [[], []]
    for H in H_sorted:
        for key, err, val in [
            ("flat_pnl", err_flat, results[H]["gamma_flat"]),
            ("sig3_pnl", err_sig3, results[H]["gamma_sig3"]),
            ("sigfull_pnl", err_sigfull, results[H]["gamma_sigfull"]),
        ]:
            ci = ci_gamma(H, key)
            if ci is None:
                err[0].append(0); err[1].append(0)
            else:
                err[0].append(val - ci[0])
                err[1].append(ci[1] - val)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.errorbar(H_arr, g_flat, yerr=err_flat, fmt="o-", color=C_FLAT, lw=2, ms=9,
                capsize=4, label="Flat (4d)")
    ax.errorbar(H_arr, g_sig3, yerr=err_sig3, fmt="s-", color=C_SIG3, lw=2, ms=9,
                capsize=4, label="Sig-3 (7d)")
    ax.errorbar(H_arr, g_sigfull, yerr=err_sigfull, fmt="D-", color=C_SIGFULL, lw=2, ms=9,
                capsize=4, label="Sig-full (12d)")

    # Annotate values
    for H, g in zip(H_arr, g_flat):
        ax.annotate(f"{g:+.2f}", (H, g), textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color=C_FLAT)
    for H, g in zip(H_arr, g_sigfull):
        ax.annotate(f"{g:+.2f}", (H, g), textcoords="offset points", xytext=(8, -14),
                    fontsize=8, color=C_SIGFULL)

    ax.set_xlabel("Hurst $H$", fontsize=12)
    ax.set_ylabel(r"$\Gamma = \mathrm{ES}_{95}^{\mathrm{BS}} - \mathrm{ES}_{95}^{\mathrm{DH}}$",
                  fontsize=12)
    ax.set_title("Hedging Advantage vs Hurst Parameter — Full Budget", fontsize=13)
    ax.text(0.5, 1.01, "80k train, 200 epochs, $n_{\\rm steps}$=100",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "fig_lean_h4_trend.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved {path.name}", flush=True)


def fig_roughness_advantage(results: dict[float, dict], tensors: dict[float, dict]) -> None:
    """Figure B: Γ_sig-full − Γ_flat vs H."""
    H_sorted = sorted(results.keys())
    H_arr = np.array(H_sorted)
    rough_adv = np.array([results[H]["roughness_adv"] for H in H_sorted])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.plot(H_arr, rough_adv, "D-", color=C_SIGFULL, lw=2.2, ms=11,
            markeredgecolor="k", markeredgewidth=0.5)
    for H, v in zip(H_arr, rough_adv):
        ax.annotate(f"{v:+.3f}", (H, v), textcoords="offset points", xytext=(10, 0),
                    fontsize=9)

    ax.set_xlabel("Hurst $H$", fontsize=12)
    ax.set_ylabel(r"$\Gamma^{\rm sig\text{-}full} - \Gamma^{\rm flat}$", fontsize=12)
    ax.set_title("Path-Feature Contribution to Roughness Exploitation", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "fig_lean_h4_roughness_advantage.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved {path.name}", flush=True)


def fig_pnl_tails(results: dict[float, dict], tensors: dict[float, dict]) -> None:
    """Figure C: 3×1 left-tail zoom per H value."""
    H_sorted = sorted(results.keys())
    fig, axes = plt.subplots(1, len(H_sorted), figsize=(5 * len(H_sorted), 4.5), sharey=False)
    if len(H_sorted) == 1:
        axes = [axes]

    for ax, H in zip(axes, H_sorted):
        t = tensors.get(H, {})
        if not t:
            ax.text(0.5, 0.5, f"H={H}\n(tensors not saved)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"$H = {H:.2f}$")
            continue

        bs = t["bs_pnl"].numpy()
        flat = t["flat_pnl"].numpy()
        sigfull = t["sigfull_pnl"].numpy()

        # Left-tail zoom: bottom 10% of each distribution
        cutoff = min(np.percentile(bs, 10), np.percentile(flat, 10), np.percentile(sigfull, 10))
        bins = np.linspace(cutoff - 3, cutoff + 3, 50)

        ax.hist(bs[bs < cutoff + 3], bins=bins, alpha=0.4, color=C_BS, label="BS")
        ax.hist(flat[flat < cutoff + 3], bins=bins, alpha=0.4, color=C_FLAT, label="Flat")
        ax.hist(sigfull[sigfull < cutoff + 3], bins=bins, alpha=0.4,
                color=C_SIGFULL, label="Sig-full")

        ax.axvline(-results[H]["bs_metrics"]["es_95"], color=C_BS, ls=":", lw=1.5)
        ax.axvline(-results[H]["flat_metrics"]["es_95"], color=C_FLAT, ls=":", lw=1.5)
        ax.axvline(-results[H]["sigfull_metrics"]["es_95"], color=C_SIGFULL, ls=":", lw=1.5)

        ax.set_title(f"$H = {H:.2f}$", fontsize=12)
        ax.set_xlabel("P&L (left tail)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle("Left-Tail PnL Distributions (bottom 10%)", y=1.02, fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "fig_lean_h4_pnl_tails.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}", flush=True)


def fig_summary(results: dict[float, dict], tensors: dict[float, dict]) -> None:
    """Figure D: 2x2 one-page summary."""
    H_sorted = sorted(results.keys())
    H_arr = np.array(H_sorted)
    g_flat = np.array([results[H]["gamma_flat"] for H in H_sorted])
    g_sig3 = np.array([results[H]["gamma_sig3"] for H in H_sorted])
    g_sigfull = np.array([results[H]["gamma_sigfull"] for H in H_sorted])
    rough_adv = g_sigfull - g_flat

    es_bs = np.array([results[H]["bs_metrics"]["es_95"] for H in H_sorted])
    rel_flat = 100 * g_flat / es_bs
    rel_sig3 = 100 * g_sig3 / es_bs
    rel_sigfull = 100 * g_sigfull / es_bs

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (A) Gamma curves
    ax = axes[0, 0]
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.plot(H_arr, g_flat, "o-", color=C_FLAT, lw=2, ms=8, label="Flat")
    ax.plot(H_arr, g_sig3, "s-", color=C_SIG3, lw=2, ms=8, label="Sig-3")
    ax.plot(H_arr, g_sigfull, "D-", color=C_SIGFULL, lw=2, ms=8, label="Sig-full")
    ax.set_xlabel("$H$"); ax.set_ylabel("$\\Gamma$")
    ax.set_title("(A) Advantage gap vs $H$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (B) Roughness advantage
    ax = axes[0, 1]
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.plot(H_arr, rough_adv, "D-", color=C_SIGFULL, lw=2.2, ms=10,
            markeredgecolor="k", markeredgewidth=0.5)
    for H, v in zip(H_arr, rough_adv):
        ax.annotate(f"{v:+.3f}", (H, v), textcoords="offset points", xytext=(8, 6),
                    fontsize=9)
    ax.set_xlabel("$H$"); ax.set_ylabel(r"$\Gamma^{\rm sig\text{-}full}-\Gamma^{\rm flat}$")
    ax.set_title("(B) Roughness-specific advantage"); ax.grid(True, alpha=0.3)

    # (C) Percentage improvement over BS
    ax = axes[1, 0]
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.plot(H_arr, rel_flat, "o-", color=C_FLAT, lw=2, ms=7, label="Flat")
    ax.plot(H_arr, rel_sig3, "s-", color=C_SIG3, lw=2, ms=7, label="Sig-3")
    ax.plot(H_arr, rel_sigfull, "D-", color=C_SIGFULL, lw=2, ms=7, label="Sig-full")
    ax.set_xlabel("$H$"); ax.set_ylabel("% improvement over BS")
    ax.set_title("(C) Relative improvement")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (D) Text summary
    ax = axes[1, 1]; ax.axis("off")
    best_by_H = {}
    for H in H_sorted:
        vals = {"flat": g_flat[H_sorted.index(H)],
                "sig-3": g_sig3[H_sorted.index(H)],
                "sig-full": g_sigfull[H_sorted.index(H)]}
        best_by_H[H] = max(vals, key=vals.get)

    max_ra = float(rough_adv.max())
    max_ra_H = H_sorted[int(np.argmax(rough_adv))]
    if any(v > 0.2 for v in rough_adv) and max_ra_H <= 0.1:
        verdict = "weak H4 support"
    else:
        verdict = "H4 refuted"

    lines = ["Summary", "", "Best strategy per H:"]
    for H in H_sorted:
        lines.append(f"  H={H:.2f}: {best_by_H[H]}")
    lines.append("")
    lines.append(f"Max roughness adv:  {max_ra:+.3f}")
    lines.append(f"  at H={max_ra_H:.2f}")
    lines.append("")
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append("Stage 1 reference:")
    lines.append("  H=0.05, 80k train, 200 epochs")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(D) Key numbers")

    fig.suptitle("Lean H4 Sweep — Full-Budget Three-Point Trend Check",
                 y=1.00, fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "fig_lean_h4_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}", flush=True)


# =======================================================================
# LaTeX table
# =======================================================================

def write_latex_table(results: dict[float, dict]) -> None:
    H_sorted = sorted(results.keys())
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Full-budget three-point H-sweep with path-feature ablation.",
        r"         Budget: 80{,}000 training paths, 200 epochs, $n_{\rm steps}=100$.}",
        r"\label{tab:lean_h4}",
        r"\begin{tabular}{c|rrrr|rrr}",
        r"\hline",
        r"$H$ & $\mathrm{ES}_{95}^{\text{BS}}$"
        r" & $\mathrm{ES}_{95}^{\text{flat}}$"
        r" & $\mathrm{ES}_{95}^{\text{sig-3}}$"
        r" & $\mathrm{ES}_{95}^{\text{sig-full}}$"
        r" & $\Gamma^{\text{flat}}$"
        r" & $\Gamma^{\text{sig-full}}$"
        r" & $\Gamma^{\text{sig-full}}-\Gamma^{\text{flat}}$ \\",
        r"\hline",
    ]
    for H in H_sorted:
        r = results[H]
        es_bs = r["bs_metrics"]["es_95"]
        es_flat = r["flat_metrics"]["es_95"]
        es_sig3 = r["sig3_metrics"]["es_95"]
        es_sigfull = r["sigfull_metrics"]["es_95"]
        lines.append(
            f"{H:.2f} & {es_bs:.2f} & {es_flat:.2f} & {es_sig3:.2f} & "
            f"{es_sigfull:.2f} & {r['gamma_flat']:+.3f} & "
            f"{r['gamma_sigfull']:+.3f} & {r['roughness_adv']:+.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    path = OUT_DIR / "lean_h4_table.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved {path.name}", flush=True)


# =======================================================================
# Main
# =======================================================================

def main() -> None:
    print("=" * 65, flush=True)
    print("  Lean H4 Analysis", flush=True)
    print("=" * 65, flush=True)

    results = load_summary()
    tensors = load_pnl_tensors(list(results.keys()))

    print(f"\n  Loaded {len(results)} H values: {sorted(results.keys())}", flush=True)
    for H in sorted(results.keys()):
        r = results[H]
        print(
            f"    H={H:.2f}  Γ_flat={r['gamma_flat']:+.3f}  "
            f"Γ_sig-3={r['gamma_sig3']:+.3f}  Γ_sig-full={r['gamma_sigfull']:+.3f}  "
            f"rough_adv={r['roughness_adv']:+.3f}",
            flush=True,
        )

    print("\n  Generating figures ...", flush=True)
    fig_trend(results, tensors)
    fig_roughness_advantage(results, tensors)
    fig_pnl_tails(results, tensors)
    fig_summary(results, tensors)
    write_latex_table(results)

    # Verdict
    H_sorted = sorted(results.keys())
    rough_advs = [results[H]["roughness_adv"] for H in H_sorted]
    max_ra = max(rough_advs)
    max_ra_H = H_sorted[rough_advs.index(max_ra)]
    any_positive = any(v > 0.2 for v in rough_advs)

    print("\n" + "=" * 65, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 65, flush=True)
    for H in H_sorted:
        print(f"  H={H:.2f}: roughness_adv = {results[H]['roughness_adv']:+.3f}", flush=True)
    print("", flush=True)

    if any_positive and max_ra_H <= 0.1:
        print("  weak H4 support — roughness advantage appears at small H", flush=True)
        print(f"  max adv = {max_ra:+.3f} at H={max_ra_H}", flush=True)
        print("  RECOMMEND: extend to full 9-point sweep (Prompt 8.6) before Prompt 9", flush=True)
    else:
        print("  H4 refuted at all three points", flush=True)
        print(f"  max adv = {max_ra:+.3f}  (not meaningfully positive)", flush=True)
        print("  RECOMMEND: proceed to Prompt 9 (Pareto front) with Scenario C narrative",
              flush=True)


if __name__ == "__main__":
    main()
