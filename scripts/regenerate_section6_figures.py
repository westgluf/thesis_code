"""Regenerate Section 6 main-text figures with the corrected three-strategy
composition (Black-Scholes Delta, True Heston PDE Delta, Deep Hedger; no
Plug-in Delta) and standalone panels for M.5, L.1, L.4, M.1.

Outputs (in ``latex_package/figures/``):
  6_3_1_pnl_histograms.png       — three-strategy P&L histograms (seed 2024)
  6_3_1_qq_plots.png             — three-panel Q-Q plots
  6_3_1_metrics_bar.png          — three-strategy 5-seed metrics bar chart
  6_3_1_strategy_comparison.png  — three-strategy 5-seed ES_0.95 bars
  6_3_2_objective_robustness.png — five-objective M.5 panel
  6_3_4_multi_source.png         — three-source L.1 zero-shot bars
  6_3_4_reverse_transfer.png     — two-target L.4 reverse transfer bars
  6_3_5_extended_radius.png      — DH vs BS worst-case eta+ over radii

Run from repo root::

    python scripts/regenerate_section6_figures.py [--no-multi-source]

The ``--no-multi-source`` flag skips the L.1 figure (used while the L.1
Heston 5-seed run is still in progress).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = REPO_ROOT / "latex_package" / "figures"
MIRROR = REPO_ROOT / "figures" / "canonical_v2"

DPI = 200

# Strategy color palette
COL_BS = "#4F81BD"        # steelblue
COL_HESTON = "#E0853A"    # darkorange
COL_DH = "#3A8C3A"        # forestgreen

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_pnl_arrays() -> dict[str, np.ndarray]:
    """Load BS, True Heston PDE, and DH per-path P&L arrays for seed 2024."""
    base = REPO_ROOT / "results" / "canonical_v2"
    return {
        "BS Delta": np.load(base / "baseline_seed2024_pnl_bs.npy"),
        "True Heston PDE Delta": np.load(base / "heston_pde_pnl_seed2024.npy"),
        "Deep Hedger": np.load(base / "baseline_seed2024_pnl_dh.npy"),
    }

def load_5seed_metrics() -> dict[str, dict[str, float]]:
    """Load 5-seed aggregate metrics (mean, std) for each strategy."""
    canon = json.load(open(REPO_ROOT / "results" / "canonical_v2"
                            / "baseline_5seeds.json"))
    hpde = json.load(open(REPO_ROOT / "results" / "heston_pde"
                           / "heston_pde_5seeds.json"))
    bs = canon["aggregated"]["0.0"]
    dh = canon["aggregated"]["0.0"]
    h = hpde["aggregated"]["heston_pde"]
    bs_h = hpde["aggregated"]["bs"]   # cross-check BS from heston file (different seeds)
    return {
        "BS Delta": {
            "es_95_mean": bs["es95_bs"]["mean"],
            "es_95_std":  bs["es95_bs"]["std"],
            "es_99_mean": bs["es99_bs"]["mean"],
            "es_99_std":  bs["es99_bs"]["std"],
            "std_pnl_mean": bs["std_pl_bs"]["mean"],
            "std_pnl_std":  bs["std_pl_bs"]["std"],
        },
        "True Heston PDE Delta": {
            "es_95_mean": h["es_95"]["mean"],
            "es_95_std":  h["es_95"]["std"],
            "es_99_mean": h["es_99"]["mean"],
            "es_99_std":  h["es_99"]["std"],
            "std_pnl_mean": h["std_pnl"]["mean"],
            "std_pnl_std":  h["std_pnl"]["std"],
        },
        "Deep Hedger": {
            "es_95_mean": dh["es95_dh"]["mean"],
            "es_95_std":  dh["es95_dh"]["std"],
            "es_99_mean": dh["es99_dh"]["mean"],
            "es_99_std":  dh["es99_dh"]["std"],
            "std_pnl_mean": dh["std_pl_dh"]["mean"],
            "std_pnl_std":  dh["std_pl_dh"]["std"],
        },
    }

# --------------------------------------------------------------------------
# 6.3.1 — P&L histograms (three strategies, seed 2024)
# --------------------------------------------------------------------------

def plot_pnl_histograms(pnls: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {
        "BS Delta": COL_BS,
        "True Heston PDE Delta": COL_HESTON,
        "Deep Hedger": COL_DH,
    }

    all_vals = np.concatenate(list(pnls.values()))
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(lo, hi, 90)

    for name, arr in pnls.items():
        loss = -arr
        es_95 = float(loss[loss >= np.quantile(loss, 0.95)].mean())
        ax.hist(arr, bins=bins, alpha=0.45, color=colors[name],
                label=f"{name}  (ES$_{{0.95}}$ = {es_95:.2f})",
                density=True, edgecolor="none")
        # Vertical line at -ES_0.95 quantile
        ax.axvline(-es_95, color=colors[name], linestyle="--",
                   linewidth=1.0, alpha=0.85)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    ax.set_title("Terminal P&L distributions (seed 2024, canonical rough Bergomi)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.1 — Q-Q plots (three panels)
# --------------------------------------------------------------------------

def plot_qq_plots(pnls: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    palette = {
        "BS Delta": COL_BS,
        "True Heston PDE Delta": COL_HESTON,
        "Deep Hedger": COL_DH,
    }

    # Determine common y range from standardised data
    standardised = {n: (a - a.mean()) / a.std() for n, a in pnls.items()}
    global_lo = min(s.min() for s in standardised.values())
    global_hi = max(s.max() for s in standardised.values())

    for ax, (name, std) in zip(axes, standardised.items()):
        sstats.probplot(std, dist="norm", plot=ax)
        # Customize the points (line 0) and reference line (line 1)
        lines = ax.get_lines()
        lines[0].set_color(palette[name])
        lines[0].set_markersize(2.0)
        lines[0].set_alpha(0.5)
        lines[1].set_color("red")
        lines[1].set_linestyle("--")
        lines[1].set_linewidth(1.2)
        ax.set_title(name)
        ax.set_ylim(global_lo - 1, global_hi + 1)
        ax.set_xlabel("Standard normal quantile")
        ax.set_ylabel("Standardised P&L quantile")
        ax.grid(alpha=0.3)

    fig.suptitle("Q-Q plots vs. standard normal "
                 "(seed 2024, canonical rough Bergomi)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.1 — Metrics bar chart (three strategies × three metrics)
# --------------------------------------------------------------------------

def plot_metrics_bar(metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    strategies = ["BS Delta", "True Heston PDE Delta", "Deep Hedger"]
    palette = [COL_BS, COL_HESTON, COL_DH]
    metric_keys = [("ES$_{0.95}$", "es_95"),
                   ("ES$_{0.99}$", "es_99"),
                   ("Std P&L", "std_pnl")]

    n_metrics = len(metric_keys)
    n_strats = len(strategies)
    width = 0.25
    xpos = np.arange(n_metrics)

    for i, strat in enumerate(strategies):
        means = [metrics[strat][f"{k}_mean"] for _, k in metric_keys]
        stds  = [metrics[strat][f"{k}_std"]  for _, k in metric_keys]
        offset = (i - (n_strats - 1) / 2) * width
        bars = ax.bar(xpos + offset, means, width, yerr=stds,
                      color=palette[i], label=strat, capsize=4,
                      edgecolor="black", linewidth=0.5,
                      error_kw={"alpha": 0.7, "linewidth": 1.0})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(xpos)
    ax.set_xticklabels([lbl for lbl, _ in metric_keys])
    ax.set_ylabel("Value")
    ax.set_title("5-seed risk metrics (canonical rough Bergomi, $\\lambda = 0$)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.1 — Strategy comparison (single bar group, ES_0.95)
# --------------------------------------------------------------------------

def plot_strategy_comparison(metrics: dict[str, dict[str, float]],
                              out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))

    strategies = ["BS Delta", "True Heston PDE Delta", "Deep Hedger"]
    palette = [COL_BS, COL_HESTON, COL_DH]

    means = [metrics[s]["es_95_mean"] for s in strategies]
    stds  = [metrics[s]["es_95_std"]  for s in strategies]
    xpos = np.arange(len(strategies))

    bars = ax.bar(xpos, means, yerr=stds, color=palette, capsize=6,
                  edgecolor="black", linewidth=0.7,
                  error_kw={"linewidth": 1.2})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=10)

    # Reference line: BS mean
    ax.axhline(metrics["BS Delta"]["es_95_mean"], color="grey",
               linestyle=":", linewidth=1.0, alpha=0.7,
               label="BS Delta reference")

    ax.set_xticks(xpos)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("ES$_{0.95}$ (5-seed mean)")
    ax.set_title("ES$_{0.95}$ across hedging strategies "
                 "(canonical rough Bergomi, 5 seeds)")
    ax.set_ylim(bottom=min(means) * 0.85, top=max(means) * 1.10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.2 — Objective robustness (M.5) panel
# --------------------------------------------------------------------------

def plot_objective_robustness(out_path: Path) -> None:
    M5 = json.load(open(REPO_ROOT / "results" / "perturbation_v2"
                         / "M5_objective_robustness.json"))

    objectives = ["es_090", "es_095", "es_099", "mse", "entropic"]
    obj_labels = ["ES$_{0.90}$", "ES$_{0.95}$", "ES$_{0.99}$", "MSE", "Entropic"]
    radii = ["1", "2", "3"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    xpos = np.arange(len(objectives))

    radius_colors = ["#A6CEE3", "#1F78B4", "#08306B"]  # light → dark blue
    for i, r in enumerate(radii):
        means = [M5["results"][o]["aggregate_per_radius"][r]["mean"]
                 for o in objectives]
        stds  = [M5["results"][o]["aggregate_per_radius"][r]["std"]
                 for o in objectives]
        offset = (i - 1) * width
        bars = ax.bar(xpos + offset, means, width, yerr=stds,
                      color=radius_colors[i], label=f"$r = {r}$",
                      capsize=3, edgecolor="black", linewidth=0.4,
                      error_kw={"alpha": 0.7, "linewidth": 0.8})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(xpos)
    ax.set_xticklabels(obj_labels, fontsize=10)
    ax.set_ylabel("Worst-case ES$_{0.95}$ (5 seeds)")
    ax.set_title("Worst-case tail risk by training objective "
                 "(5 seeds × 6 axis-aligned PGD directions)")
    ax.legend(title="PGD radius", loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.4 — Reverse transfer (L.4) panel
# --------------------------------------------------------------------------

def plot_reverse_transfer(out_path: Path) -> None:
    L4 = json.load(open(REPO_ROOT / "results" / "transfer_v2"
                         / "L4_reverse_transfer.json"))

    targets = ["gbm", "heston"]
    target_labels = ["Target = GBM\n(reference: BS Delta)",
                     "Target = Heston\n(reference: Heston PDE)"]

    dh_means = []
    dh_stds  = []
    ref_means = []
    ref_stds  = []
    gaps = []
    gap_stds = []
    for t in targets:
        agg = L4["results"]["per_target"][t]["aggregate"]
        dh_means.append(agg["dh_es95"]["mean"])
        dh_stds.append(agg["dh_es95"]["std"])
        ref_means.append(agg["ref_es95"]["mean"])
        ref_stds.append(agg["ref_es95"]["std"])
        gaps.append(agg["gap_dh_minus_ref"]["mean"])
        gap_stds.append(agg["gap_dh_minus_ref"]["std"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xpos = np.arange(len(targets))
    width = 0.36

    bars_dh  = ax.bar(xpos - width/2, dh_means, width, yerr=dh_stds,
                      color=COL_DH, label="rB-trained Deep Hedger",
                      capsize=5, edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 1.0, "alpha": 0.8})
    bars_ref = ax.bar(xpos + width/2, ref_means, width, yerr=ref_stds,
                      color="#888888", label="Reference (BS / Heston PDE)",
                      capsize=5, edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 1.0, "alpha": 0.8})

    for b, m in zip(bars_dh, dh_means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.10,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    for b, m in zip(bars_ref, ref_means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.10,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    # Gap annotations
    for i, (g, gs) in enumerate(zip(gaps, gap_stds)):
        ymax = max(dh_means[i], ref_means[i]) + max(dh_stds[i], ref_stds[i])
        sign = "+" if g >= 0 else ""
        ax.text(i, ymax + 0.6,
                f"gap = {sign}{g:.2f} ± {gs:.2f}",
                ha="center", fontsize=10, fontweight="bold",
                color="#B22222" if g >= 0 else "#3A8C3A")

    ax.set_xticks(xpos)
    ax.set_xticklabels(target_labels, fontsize=10)
    ax.set_ylabel("ES$_{0.95}$ (3 seeds)")
    ax.set_title("Reverse transfer: rough-Bergomi-trained DH on Markovian targets")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=max(max(dh_means), max(ref_means)) * 1.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# 6.3.4 — Multi-source (L.1) panel  (requires L1_heston_5seeds.json)
# --------------------------------------------------------------------------

def plot_multi_source(out_path: Path) -> None:
    """Three-source zero-shot bars: GBM (L.2 N=160k), Heston (new L.1 5-seed),
    rBergomi H=0.3 (L.2 N=160k). References: BS Delta (canonical 5-seed),
    canonical DH (5-seed)."""
    L2 = json.load(open(REPO_ROOT / "results" / "transfer_v2"
                         / "L2_budget_sweep.json"))
    canon = json.load(open(REPO_ROOT / "results" / "canonical_v2"
                            / "baseline_5seeds.json"))

    L1H_path = REPO_ROOT / "results" / "transfer_v2" / "L1_heston_5seeds.json"
    if L1H_path.exists():
        L1H = json.load(open(L1H_path))
        heston_agg = L1H["results"]["heston"]["aggregate"]["es_95"]
        heston_mean = heston_agg["mean"]
        heston_std = heston_agg["std"]
        heston_label = "Heston source\n(5 seeds)"
    else:
        # Fallback: L.2 N=160k for Heston (3 seeds)
        agg = L2["results"]["heston"]["160000"]["aggregate"]["es_95"]
        heston_mean = agg["mean"]
        heston_std = agg["std"]
        heston_label = "Heston source\n(L.2 N=160k, 3 seeds)"

    # GBM source — L.2 N=160k 3 seeds
    agg_gbm = L2["results"]["gbm"]["160000"]["aggregate"]["es_95"]
    gbm_mean = agg_gbm["mean"]; gbm_std = agg_gbm["std"]

    # rBergomi H=0.3 source — L.2 N=160k 3 seeds
    agg_rb = L2["results"]["rbergomi_H03"]["160000"]["aggregate"]["es_95"]
    rb_mean = agg_rb["mean"]; rb_std = agg_rb["std"]

    # References
    bs_mean = canon["aggregated"]["0.0"]["es95_bs"]["mean"]
    bs_std  = canon["aggregated"]["0.0"]["es95_bs"]["std"]
    dh_mean = canon["aggregated"]["0.0"]["es95_dh"]["mean"]
    dh_std  = canon["aggregated"]["0.0"]["es95_dh"]["std"]

    sources = ["GBM source\n(L.2 N=160k, 3 seeds)",
               heston_label,
               "rBergomi H=0.3 source\n(L.2 N=160k, 3 seeds)"]
    means = [gbm_mean, heston_mean, rb_mean]
    stds  = [gbm_std, heston_std, rb_std]
    palette = [COL_BS, COL_HESTON, "#7A4FC4"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xpos = np.arange(len(sources))
    bars = ax.bar(xpos, means, yerr=stds, color=palette,
                  edgecolor="black", linewidth=0.5, capsize=5,
                  error_kw={"linewidth": 1.0})
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)

    # Reference lines
    ax.axhline(bs_mean, color="#444444", linestyle=":", linewidth=1.2,
               alpha=0.85, label=f"BS Delta reference: {bs_mean:.3f} ± {bs_std:.3f}")
    ax.axhline(dh_mean, color=COL_DH, linestyle="--", linewidth=1.2,
               alpha=0.85,
               label=f"canonical Deep Hedger: {dh_mean:.3f} ± {dh_std:.3f}")

    ax.set_xticks(xpos)
    ax.set_xticklabels(sources, fontsize=9)
    ax.set_ylabel("ES$_{0.95}$ (zero-shot on canonical rough Bergomi target)")
    ax.set_title("Multi-source zero-shot transfer to canonical rough Bergomi")
    ax.set_ylim(bottom=min(min(means) - 0.5, dh_mean - 0.4),
                top=max(max(means) + max(stds) + 0.5, bs_mean + 0.5))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}  (Heston source = {heston_label.splitlines()[1]})")

# --------------------------------------------------------------------------
# 6.3.5 — Extended radius (M.1) panel
# --------------------------------------------------------------------------

def plot_extended_radius(out_path: Path) -> None:
    M1 = json.load(open(REPO_ROOT / "results" / "perturbation_v2"
                         / "M1_extended_radius.json"))

    radii_str = ["0.5", "1", "1.5", "2", "3", "4", "5"]
    radii = [float(r) for r in radii_str]

    # The "worst direction" at each radius is eta+
    dh_means = []; dh_stds = []
    bs_means = []; bs_stds = []
    for r in radii_str:
        agg = M1["results"]["eta"]["+"][r]["aggregate"]
        dh_means.append(agg["dh_es95"]["mean"])
        dh_stds.append(agg["dh_es95"]["std"])
        bs_means.append(agg["bs_es95"]["mean"])
        bs_stds.append(agg["bs_es95"]["std"])

    dh_means = np.array(dh_means); dh_stds = np.array(dh_stds)
    bs_means = np.array(bs_means); bs_stds = np.array(bs_stds)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(radii, dh_means, "o-", color=COL_DH, linewidth=1.5,
            markersize=6, label="Deep Hedger")
    ax.fill_between(radii, dh_means - dh_stds, dh_means + dh_stds,
                     color=COL_DH, alpha=0.18)
    ax.plot(radii, bs_means, "s-", color=COL_BS, linewidth=1.5,
            markersize=6, label="Black-Scholes Delta")
    ax.fill_between(radii, bs_means - bs_stds, bs_means + bs_stds,
                     color=COL_BS, alpha=0.18)

    # Vertical reference lines
    ymin, ymax = bs_means.min() - 1.0, bs_means.max() + 1.5
    ax.set_ylim(ymin, ymax)
    ax.axvline(2.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(2.05, ymin + 0.4, "$r = 2$\n(initial study limit)",
            fontsize=9, color="grey", ha="left")
    r_star = M1["crossover_analysis"]["r_star"]
    ax.axvline(r_star, color="firebrick", linestyle="--", linewidth=1.0,
               alpha=0.85)
    ax.text(r_star + 0.05, ymin + 0.4,
            f"$r^* = {r_star}$\n(eta$-$ crossover)",
            fontsize=9, color="firebrick", ha="left")

    ax.set_xlabel("PGD radius $r$ (axis-scaled units)")
    ax.set_ylabel("Worst-case ES$_{0.95}$ (eta+ direction)")
    ax.set_title("Worst-case tail risk under axis-aligned PGD perturbations")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")

# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-multi-source", action="store_true",
                        help="Skip 6_3_4_multi_source.png "
                             "(use when L1_heston_5seeds.json is not yet ready)")
    parser.add_argument("--only-multi-source", action="store_true",
                        help="Only generate 6_3_4_multi_source.png")
    args = parser.parse_args()

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {FIG_OUT}")

    if args.only_multi_source:
        plot_multi_source(FIG_OUT / "6_3_4_multi_source.png")
        return

    print("\n[1/8] Loading per-path P&L arrays...")
    pnls = load_pnl_arrays()
    print("\n[2/8] Loading 5-seed metrics...")
    metrics = load_5seed_metrics()

    print("\n[3/8] Generating 6_3_1_pnl_histograms.png ...")
    plot_pnl_histograms(pnls, FIG_OUT / "6_3_1_pnl_histograms.png")

    print("\n[4/8] Generating 6_3_1_qq_plots.png ...")
    plot_qq_plots(pnls, FIG_OUT / "6_3_1_qq_plots.png")

    print("\n[5/8] Generating 6_3_1_metrics_bar.png ...")
    plot_metrics_bar(metrics, FIG_OUT / "6_3_1_metrics_bar.png")

    print("\n[6/8] Generating 6_3_1_strategy_comparison.png ...")
    plot_strategy_comparison(metrics, FIG_OUT / "6_3_1_strategy_comparison.png")

    print("\n[7/8] Generating 6_3_2_objective_robustness.png ...")
    plot_objective_robustness(FIG_OUT / "6_3_2_objective_robustness.png")

    print("\n[8/8] Generating 6_3_4_reverse_transfer.png ...")
    plot_reverse_transfer(FIG_OUT / "6_3_4_reverse_transfer.png")

    print("\n[bonus] Generating 6_3_5_extended_radius.png ...")
    plot_extended_radius(FIG_OUT / "6_3_5_extended_radius.png")

    if not args.no_multi_source:
        print("\n[9/9] Generating 6_3_4_multi_source.png ...")
        plot_multi_source(FIG_OUT / "6_3_4_multi_source.png")
    else:
        print("\n[skip] 6_3_4_multi_source.png — use --only-multi-source after L.1 done")

    print("\nAll figures written.")

if __name__ == "__main__":
    main()
