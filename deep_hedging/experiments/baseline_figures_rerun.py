#!/usr/bin/env python
"""
Phase D — Regenerate Section 6.3.1 figures on seed 2024 with fixed seeding.

Rerun the canonical Section 6.3 baseline (rough Bergomi H=0.07, eta=1.9,
rho=-0.7, xi0=0.235^2) for a single seed (2024) with the seeding protocol
fixed by Phase B. Produce three figures matching the dissertation's 6.3.1
structure:
  - 6_3_1_pnl_histograms_seed2024.png  (Figure 20)
  - 6_3_1_qq_plots_seed2024.png        (Figure 21)
  - 6_3_1_metrics_bar_seed2024.png     (Figure 22)

Also dumps full numerical data (per-path P&L arrays, all metrics, training
history) to `results/canonical_v2/baseline_seed2024_full.json`.

Run:
    python -u -m deep_hedging.experiments.baseline_figures_rerun
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats

from deep_hedging.experiments.run_section_6_3_baseline import Section63Experiment
from deep_hedging.utils.config import DatasetConfig, RoughBergomiParams

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "canonical_v2"
FIGURES_DIR = REPO_ROOT / "figures" / "canonical_v2"

SEED = 2024
COST_LAMBDA = 0.0

# Canonical training budget (same as Phase B baseline)
N_TRAIN = 80_000
N_VAL = 20_000
N_TEST = 50_000
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 2048
LR = 1e-3

# Aggregate Γ (from Phase B baseline_5seeds.json, 5-seed mean ± std)
CANONICAL_5SEED_GAMMA_MEAN = 1.1479
CANONICAL_5SEED_GAMMA_STD = 0.0761

COLORS = {
    "BS Delta": "#2196F3",
    "Heston Delta": "#FF9800",    # "plug-in" / realised-variance BS
    "Deep Hedger": "#4CAF50",
}

def _git_commit_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"

def _mean_turnover(deltas: torch.Tensor) -> float:
    """Mean over paths of sum_k |delta_k - delta_{k-1}|, delta_{-1}=0."""
    batch = deltas.shape[0]
    dtype, device = deltas.dtype, deltas.device
    delta_prev = torch.cat(
        [torch.zeros(batch, 1, dtype=dtype, device=device), deltas[:, :-1]],
        dim=1,
    )
    return float((deltas - delta_prev).abs().sum(dim=1).mean())

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_pnl_histograms(pnls: dict[str, np.ndarray],
                         metrics: dict[str, dict[str, float]],
                         save_path: Path,
                         seed: int,
                         gamma: float) -> None:
    """Figure 20 equivalent: terminal P&L histograms (BS, Heston, DH)."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Use wide bins over the union range
    all_pnl = np.concatenate(list(pnls.values()))
    lo = float(np.quantile(all_pnl, 0.0005))
    hi = float(np.quantile(all_pnl, 0.9995))
    bins = np.linspace(lo, hi, 120)

    for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
        if name not in pnls:
            continue
        ax.hist(pnls[name], bins=bins, alpha=0.42, density=True,
                color=COLORS[name], edgecolor="black", lw=0.3, label=name)

    # ES_0.95 quantiles as vertical dashed lines
    for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
        if name not in metrics:
            continue
        es = metrics[name]["es_95"]
        ax.axvline(-es, color=COLORS[name], ls="--", lw=1.4, alpha=0.9,
                   label=f"−ES$_{{0.95}}$({name}) = {-es:+.2f}")

    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    ax.set_title(f"Section 6.3.1 — Terminal P&L distributions "
                 f"(rBergomi baseline, seed {seed}, λ=0)\n"
                 f"Γ(seed {seed}) = {gamma:+.4f}")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}", flush=True)

def plot_qq_plots(pnls: dict[str, np.ndarray], save_path: Path,
                   seed: int) -> None:
    """Figure 21 equivalent: Q-Q plots of P&L vs Gaussian reference."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, name in zip(axes, ["BS Delta", "Heston Delta", "Deep Hedger"]):
        if name not in pnls:
            ax.set_title(f"{name}\n(unavailable)")
            continue
        pnl = pnls[name]
        # Reference theoretical quantiles
        (osm, osr), (slope, intercept, _) = sp_stats.probplot(pnl, dist="norm")
        ax.scatter(osm, osr, s=2, alpha=0.35, color=COLORS[name], label="samples")
        xlim = ax.get_xlim()
        xs = np.linspace(xlim[0], xlim[1], 50)
        ax.plot(xs, slope * xs + intercept, "k--", lw=1.0,
                label=f"LS fit (slope={slope:.3f})")
        ax.set_title(name)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Section 6.3.1 — Q-Q plots vs Gaussian (rBergomi, seed {seed}, λ=0)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}", flush=True)

def plot_metrics_bar(metrics: dict[str, dict[str, float]],
                      turnovers: dict[str, float],
                      save_path: Path, seed: int) -> None:
    """Figure 22 equivalent: grouped bar chart of ES, std, turnover per strategy."""
    metric_keys = ["es_95", "es_99", "std_pnl"]
    metric_labels = [r"$\mathrm{ES}_{0.95}$", r"$\mathrm{ES}_{0.99}$", "Std P&L"]

    strat_names = ["BS Delta", "Heston Delta", "Deep Hedger"]

    # Top plot: risk metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    n_metrics = len(metric_keys)
    n_strats = len(strat_names)
    width = 0.27
    x = np.arange(n_metrics)
    for i, name in enumerate(strat_names):
        if name not in metrics:
            continue
        vals = [metrics[name][k] for k in metric_keys]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, color=COLORS[name], alpha=0.85,
               edgecolor="black", lw=0.5, label=name)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Risk metric value")
    ax.set_title("Risk metrics")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Right plot: turnover
    ax = axes[1]
    for i, name in enumerate(strat_names):
        if name not in turnovers:
            continue
        ax.bar(i, turnovers[name], color=COLORS[name], alpha=0.85,
               edgecolor="black", lw=0.5, label=name)
        ax.text(i, turnovers[name] + 0.03, f"{turnovers[name]:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(strat_names)))
    ax.set_xticklabels(strat_names, rotation=15)
    ax.set_ylabel("Mean turnover")
    ax.set_title("Turnover")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Section 6.3.1 — Risk metrics & turnover (seed {seed}, λ=0)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"  Phase D — Regenerate Section 6.3.1 figures on seed {SEED}", flush=True)
    print(f"  commit: {_git_commit_sha()}", flush=True)
    print("=" * 70, flush=True)

    params = RoughBergomiParams()  # canonical
    dataset_config = DatasetConfig(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    exp = Section63Experiment(params=params, dataset_config=dataset_config)

    t0 = time.perf_counter()
    exp.generate_data(seed=SEED)
    tgen = time.perf_counter() - t0
    print(f"  Data generation done in {tgen:.1f}s  "
          f"p0 = {exp.p0:.4f}", flush=True)

    # BS delta (σ = √ξ_0 = 0.235 is the assumed-vol used by Section 6.3.1)
    print("\n  Running BS Delta...", flush=True)
    r_bs = exp.run_bs_delta(COST_LAMBDA)
    print(f"  BS Delta ES_0.95 = {r_bs['metrics']['es_95']:.4f}", flush=True)

    # Plug-in / Heston (observes realised variance)
    print("\n  Running Plug-in (Heston-style) delta...", flush=True)
    r_plugin = exp.run_plugin_delta(COST_LAMBDA)
    print(f"  Plug-in ES_0.95 = {r_plugin['metrics']['es_95']:.4f}", flush=True)

    # Deep hedger with fixed seeding
    print("\n  Training Deep Hedger (epochs=200, patience=30)...", flush=True)
    t0 = time.perf_counter()
    r_dh = exp.run_deep_hedger(
        cost_lambda=COST_LAMBDA,
        epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR,
        seed=SEED, verbose=True,
    )
    train_time = time.perf_counter() - t0
    print(f"  Training done in {train_time/60:.1f} min  "
          f"(best_epoch={r_dh['history']['best_epoch']})", flush=True)
    print(f"  DH ES_0.95 = {r_dh['metrics']['es_95']:.4f}", flush=True)

    # Compute Γ for verification
    gamma = r_bs["metrics"]["es_95"] - r_dh["metrics"]["es_95"]
    print(f"\n  Γ (seed {SEED}) = {gamma:+.4f}", flush=True)
    print(f"  Expected: ≈ 1.1844 (from Phase B baseline_5seeds.json)", flush=True)

    # Sanity check: Γ should be close to canonical mean
    if abs(gamma - CANONICAL_5SEED_GAMMA_MEAN) > 3 * CANONICAL_5SEED_GAMMA_STD:
        print(f"  WARNING: |Γ - mean| = "
              f"{abs(gamma - CANONICAL_5SEED_GAMMA_MEAN):.4f} > 3σ", flush=True)

    # --- Extract per-strategy data ---
    strategies = ["BS Delta", "Heston Delta", "Deep Hedger"]
    # Map plug-in to "Heston Delta" label for dissertation consistency
    pnls = {
        "BS Delta": r_bs["pnl"].detach().cpu().numpy(),
        "Heston Delta": r_plugin["pnl"].detach().cpu().numpy(),
        "Deep Hedger": r_dh["pnl"].detach().cpu().numpy(),
    }
    metrics = {
        "BS Delta": r_bs["metrics"],
        "Heston Delta": r_plugin["metrics"],
        "Deep Hedger": r_dh["metrics"],
    }
    # Turnovers
    turnovers = {
        "BS Delta": _mean_turnover(r_bs["deltas"]),
        "Heston Delta": _mean_turnover(r_plugin["deltas"]),
        "Deep Hedger": _mean_turnover(r_dh["deltas"]),
    }

    # --- Figures ---
    print("\n  Generating figures...", flush=True)
    plot_pnl_histograms(
        pnls, metrics,
        FIGURES_DIR / "6_3_1_pnl_histograms_seed2024.png",
        seed=SEED, gamma=gamma,
    )
    plot_qq_plots(
        pnls,
        FIGURES_DIR / "6_3_1_qq_plots_seed2024.png",
        seed=SEED,
    )
    plot_metrics_bar(
        metrics, turnovers,
        FIGURES_DIR / "6_3_1_metrics_bar_seed2024.png",
        seed=SEED,
    )

    # --- Full JSON ---
    output = {
        "meta": {
            "script": "deep_hedging/experiments/baseline_figures_rerun.py",
            "git_commit": _git_commit_sha(),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "seed": SEED,
            "parameters": {
                "H": params.H, "eta": params.eta, "rho": params.rho,
                "xi0": params.xi0, "S0": params.S0, "T": params.T,
                "n_steps": params.n_steps, "K": 100.0,
                "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                "epochs": EPOCHS, "patience": PATIENCE,
                "batch_size": BATCH_SIZE, "lr": LR,
                "cost_lambda": COST_LAMBDA, "alpha": 0.95,
            },
            "p0": exp.p0,
            "canonical_5seed_aggregate": {
                "gamma_mean": CANONICAL_5SEED_GAMMA_MEAN,
                "gamma_std": CANONICAL_5SEED_GAMMA_STD,
                "source": "results/canonical_v2/baseline_5seeds.json",
            },
        },
        "metrics": {
            "bs": {"es95": float(metrics["BS Delta"]["es_95"]),
                    "es99": float(metrics["BS Delta"]["es_99"]),
                    "var95": float(metrics["BS Delta"]["var_95"]),
                    "std_pnl": float(metrics["BS Delta"]["std_pnl"]),
                    "mean_pnl": float(metrics["BS Delta"]["mean_pnl"]),
                    "skewness": float(metrics["BS Delta"]["skewness"]),
                    "kurtosis": float(metrics["BS Delta"]["kurtosis"]),
                    "turnover": turnovers["BS Delta"]},
            "heston": {"es95": float(metrics["Heston Delta"]["es_95"]),
                        "es99": float(metrics["Heston Delta"]["es_99"]),
                        "var95": float(metrics["Heston Delta"]["var_95"]),
                        "std_pnl": float(metrics["Heston Delta"]["std_pnl"]),
                        "mean_pnl": float(metrics["Heston Delta"]["mean_pnl"]),
                        "skewness": float(metrics["Heston Delta"]["skewness"]),
                        "kurtosis": float(metrics["Heston Delta"]["kurtosis"]),
                        "turnover": turnovers["Heston Delta"]},
            "dh": {"es95": float(metrics["Deep Hedger"]["es_95"]),
                    "es99": float(metrics["Deep Hedger"]["es_99"]),
                    "var95": float(metrics["Deep Hedger"]["var_95"]),
                    "std_pnl": float(metrics["Deep Hedger"]["std_pnl"]),
                    "mean_pnl": float(metrics["Deep Hedger"]["mean_pnl"]),
                    "skewness": float(metrics["Deep Hedger"]["skewness"]),
                    "kurtosis": float(metrics["Deep Hedger"]["kurtosis"]),
                    "turnover": turnovers["Deep Hedger"]},
        },
        "gamma": gamma,
        "training_history": {
            "train_risk": [float(v) for v in r_dh["history"]["train_risk"]],
            "val_risk": [float(v) for v in r_dh["history"]["val_risk"]],
            "best_epoch": int(r_dh["history"]["best_epoch"]),
            "best_val_risk": float(r_dh["history"]["best_val_risk"]),
        },
    }

    # Save P&L arrays separately as .npy (keep JSON compact)
    np.save(RESULTS_DIR / "baseline_seed2024_pnl_bs.npy", pnls["BS Delta"])
    np.save(RESULTS_DIR / "baseline_seed2024_pnl_heston.npy", pnls["Heston Delta"])
    np.save(RESULTS_DIR / "baseline_seed2024_pnl_dh.npy", pnls["Deep Hedger"])

    json_path = RESULTS_DIR / "baseline_seed2024_full.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {json_path}", flush=True)

    # --- README ---
    write_readme(output, gamma, FIGURES_DIR / "README.md")

    print("\n" + "=" * 70, flush=True)
    print(f"  HEADLINE: Γ(seed {SEED}) = {gamma:+.4f}  "
          f"(canonical mean: {CANONICAL_5SEED_GAMMA_MEAN:+.4f} "
          f"± {CANONICAL_5SEED_GAMMA_STD:.4f})", flush=True)
    print("=" * 70, flush=True)

def write_readme(output: dict[str, Any], gamma: float, path: Path) -> None:
    ts = output["meta"]["timestamp"]
    commit = output["meta"]["git_commit"]
    es_bs = output["metrics"]["bs"]["es95"]
    es_dh = output["metrics"]["dh"]["es95"]
    es_heston = output["metrics"]["heston"]["es95"]
    lines = [
        "# Canonical baseline figures (Section 6.3.1)",
        "",
        f"Generated: {ts}",
        f"Seed: {SEED}",
        "Script: `deep_hedging/experiments/baseline_figures_rerun.py`",
        f"Git commit: {commit}",
        "",
        "## Figure mapping",
        "",
        "- `6_3_1_pnl_histograms_seed2024.png` → Figure 20 "
        "(terminal P&L histograms; BS, Heston plug-in, Deep Hedger)",
        "- `6_3_1_qq_plots_seed2024.png` → Figure 21 (Q-Q plots vs Gaussian)",
        "- `6_3_1_metrics_bar_seed2024.png` → Figure 22 "
        "(risk metrics bar chart + turnover panel)",
        "",
        "## Per-seed values for this figure",
        "",
        f"- ES_0.95 BS: {es_bs:.4f}",
        f"- ES_0.95 Heston (plug-in): {es_heston:.4f}",
        f"- ES_0.95 DH: {es_dh:.4f}",
        f"- Γ (seed {SEED}): {gamma:+.4f}",
        "",
        "## Aggregate across 5 seeds (from Phase B `baseline_5seeds.json`)",
        "",
        f"- Γ = {CANONICAL_5SEED_GAMMA_MEAN:.4f} ± "
        f"{CANONICAL_5SEED_GAMMA_STD:.4f} (mean ± std, 5 seeds)",
        f"- 95% CI: [{CANONICAL_5SEED_GAMMA_MEAN - 2*CANONICAL_5SEED_GAMMA_STD:.4f}, "
        f"{CANONICAL_5SEED_GAMMA_MEAN + 2*CANONICAL_5SEED_GAMMA_STD:.4f}]",
        "",
        "This figure shows seed 2024 as a representative realisation. Per-seed",
        "variation across 5 seeds is documented in Appendix B Table B.1.",
        "",
        "## Raw P&L arrays",
        "",
        "- `../../results/canonical_v2/baseline_seed2024_pnl_bs.npy`",
        "- `../../results/canonical_v2/baseline_seed2024_pnl_heston.npy`",
        "- `../../results/canonical_v2/baseline_seed2024_pnl_dh.npy`",
        "",
        "Each is 50000 float32 values (1 per test path).",
    ]
    path.write_text("\n".join(lines))
    print(f"  Wrote {path}", flush=True)

if __name__ == "__main__":
    main()
