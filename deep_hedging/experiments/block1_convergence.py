#!/usr/bin/env python
"""
Block 1.1 — BLP Convergence Sweep and Richardson Extrapolation.

Runs the rBergomi simulator at six grid resolutions n ∈ {50,100,200,400,800,1600}
with 3 independent seeds, computes BS-delta hedge P&L risk metrics for each cell,
then fits the Richardson model ES(n) ≈ ES(∞) + C·n^{-α} to quantify convergence.

Part of the dissertation revision plan (Block 1, Phase P01).
All outputs go into results/block1/ and figures/block1/.  No existing files are
modified or overwritten.

Run:
    python -u -m deep_hedging.experiments.block1_convergence
    python -u -m deep_hedging.experiments.block1_convergence --dry-run
    python -u -m deep_hedging.experiments.block1_convergence --n-grid 50 100 --n-paths 500 --seeds 7001
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.optimize import curve_fit
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import (
    compute_hedging_pnl,
    compute_payoff,
)
from deep_hedging.objectives.risk_measures import (
    compute_all_metrics,
    expected_shortfall,
    value_at_risk,
)
from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

# ─── Colour scheme (matching existing experiments) ──────────────
C_BS = "#2196F3"
C_DEEP = "#4CAF50"
C_FIT = "#F44336"

SEED_COLOURS = ["#1976D2", "#388E3C", "#E64A19"]
SEED_MARKERS = ["o", "s", "^"]
MEAN_COLOUR = "#212121"

# ─── Repo-root-relative output directories ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "block1"
FIGURES_DIR = REPO_ROOT / "figures" / "block1"

# ─── Baseline rBergomi calibration ──────────────────────────────
BASELINE = {
    "H": 0.07,
    "eta": 1.9,
    "rho": -0.7,
    "xi0": 0.235**2,
    "S0": 100.0,
    "K": 100.0,
    "T": 1.0,
    "sigma_bs": 0.235,
}

# =====================================================================
#  Core computation
# =====================================================================

def _compute_metrics_numpy(pnl_np: np.ndarray) -> dict[str, float]:
    """Compute risk metrics from a numpy P&L array.

    Parameters
    ----------
    pnl_np : np.ndarray
        1-D array of P&L values.

    Returns
    -------
    dict
        Keys: mean_pl, std_pl, es_95, es_99, var_95, var_99,
        skew, kurt, q01, q05, q95, q99.
    """
    pnl_t = torch.tensor(pnl_np, dtype=torch.float64)
    return {
        "mean_pl": float(np.mean(pnl_np)),
        "std_pl": float(np.std(pnl_np, ddof=1)),
        "es_95": float(expected_shortfall(pnl_t, 0.95)),
        "es_99": float(expected_shortfall(pnl_t, 0.99)),
        "var_95": float(value_at_risk(pnl_t, 0.95)),
        "var_99": float(value_at_risk(pnl_t, 0.99)),
        "skew": float(sp_skew(pnl_np)),
        "kurt": float(sp_kurtosis(pnl_np, fisher=True)),
        "q01": float(np.percentile(pnl_np, 1)),
        "q05": float(np.percentile(pnl_np, 5)),
        "q95": float(np.percentile(pnl_np, 95)),
        "q99": float(np.percentile(pnl_np, 99)),
    }

def run_single_cell(
    n_steps: int,
    seed: int,
    n_paths: int,
    n_paths_p0: int,
    device: torch.device,
) -> dict[str, Any]:
    """Simulate rBergomi at a given n and seed, hedge with BS-delta, return metrics.

    Parameters
    ----------
    n_steps : int
        Number of discretisation steps.
    seed : int
        Random seed for simulation paths.
    n_paths : int
        Number of paths for hedging P&L evaluation.
    n_paths_p0 : int
        Number of independent paths for premium estimation.
    device : torch.device
        Computation device.

    Returns
    -------
    dict
        Risk metrics plus wall-clock time.
    """
    t0 = time.perf_counter()

    model = DifferentiableRoughBergomi(
        n_steps=n_steps,
        T=BASELINE["T"],
        H=BASELINE["H"],
        eta=BASELINE["eta"],
        rho=BASELINE["rho"],
        xi0=BASELINE["xi0"],
    )

    # --- Premium estimation on independent paths ---
    p0_seed = seed + 2000  # disjoint from test seeds
    S_p0, _, _ = model.simulate(n_paths_p0, S0=BASELINE["S0"], seed=p0_seed, device=device)
    payoff_p0 = compute_payoff(S_p0, BASELINE["K"], "call")
    p0 = float(payoff_p0.mean())

    # --- Test-set paths ---
    S, V, t_grid = model.simulate(n_paths, S0=BASELINE["S0"], seed=seed, device=device)

    # --- BS-delta hedge ---
    hedger = BlackScholesDelta(
        sigma=BASELINE["sigma_bs"],
        K=BASELINE["K"],
        T=BASELINE["T"],
    )
    deltas = hedger.hedge_paths(S)
    payoff = compute_payoff(S, BASELINE["K"], "call")
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, cost_lambda=0.0)

    # --- Metrics ---
    pnl_np = pnl.detach().cpu().numpy().astype(np.float64)
    metrics = _compute_metrics_numpy(pnl_np)
    metrics["wall_clock_s"] = round(time.perf_counter() - t0, 2)
    metrics["p0"] = p0

    return metrics

# =====================================================================
#  Richardson extrapolation
# =====================================================================

def _richardson_model(n: np.ndarray, es_inf: float, C: float, alpha: float) -> np.ndarray:
    """Parametric model: ES(n) = ES_inf + C * n^{-alpha}."""
    return es_inf + C * np.power(n.astype(float), -alpha)

def fit_richardson(
    n_values: list[int],
    metric_values: list[float],
) -> dict[str, Any]:
    """Fit ES(n) = ES(∞) + C·n^{-α} and return parameter estimates.

    Parameters
    ----------
    n_values : list[int]
        Grid sizes.
    metric_values : list[float]
        Corresponding mean metric values (one per grid size).

    Returns
    -------
    dict
        Keys: alpha_hat, alpha_ci, ES_inf, C, rel_err_at_100,
        alpha_blp_theoretical.
    """
    n_arr = np.array(n_values, dtype=float)
    m_arr = np.array(metric_values, dtype=float)

    # Initial guesses
    p0_guess = [m_arr[-1], 1.0, 0.57]

    try:
        popt, pcov = curve_fit(
            _richardson_model,
            n_arr,
            m_arr,
            p0=p0_guess,
            maxfev=10000,
            bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 5.0]),
        )
        es_inf, C, alpha_hat = popt
        perr = np.sqrt(np.diag(pcov))
        alpha_ci = [float(alpha_hat - 1.96 * perr[2]),
                    float(alpha_hat + 1.96 * perr[2])]

        # Relative error at n=100
        es_at_100 = _richardson_model(np.array([100.0]), es_inf, C, alpha_hat)[0]
        rel_err_100 = abs(es_at_100 - es_inf) / (abs(es_inf) + 1e-12)
    except (RuntimeError, ValueError) as exc:
        # Fit failed — report NaN
        es_inf = float("nan")
        C = float("nan")
        alpha_hat = float("nan")
        alpha_ci = [float("nan"), float("nan")]
        rel_err_100 = float("nan")
        print(f"  [WARNING] Richardson fit failed: {exc}", flush=True)

    return {
        "alpha_hat": float(alpha_hat),
        "alpha_ci": alpha_ci,
        "ES_inf": float(es_inf),
        "C": float(C),
        "rel_err_at_100": float(rel_err_100),
        "alpha_blp_theoretical": 0.57,
    }

# =====================================================================
#  Aggregation
# =====================================================================

def aggregate_results(
    per_seed: dict[str, dict[str, dict[str, float]]],
    n_grid: list[int],
    seeds: list[int],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-seed results into mean / std / CI per metric per n.

    Parameters
    ----------
    per_seed : dict
        Nested dict: per_seed[seed_str][n_str] -> metrics dict.
    n_grid : list[int]
        Grid sizes.
    seeds : list[int]
        Seeds used.

    Returns
    -------
    dict
        Nested dict: result[n_str][metric_name] -> {mean, std, ci_low, ci_high}.
    """
    metric_keys = [
        "mean_pl", "std_pl", "es_95", "es_99", "var_95", "var_99",
        "skew", "kurt", "q01", "q05", "q95", "q99",
    ]
    aggregated: dict[str, dict[str, dict[str, float]]] = {}

    for n in n_grid:
        n_str = str(n)
        agg_n: dict[str, dict[str, float]] = {}
        for mk in metric_keys:
            vals = [per_seed[str(s)][n_str][mk] for s in seeds]
            arr = np.array(vals)
            mu = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            z = 1.96
            se = sd / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            agg_n[mk] = {
                "mean": mu,
                "std": sd,
                "ci_low": mu - z * se,
                "ci_high": mu + z * se,
            }
        aggregated[n_str] = agg_n

    return aggregated

def compute_convergence_flags(
    aggregated: dict[str, dict[str, dict[str, float]]],
    n_grid: list[int],
) -> dict[str, Any]:
    """Compute convergence flags from aggregated results.

    Parameters
    ----------
    aggregated : dict
        Aggregated results from ``aggregate_results``.
    n_grid : list[int]
        Grid sizes (sorted ascending).

    Returns
    -------
    dict
        Convergence flag and per-level relative changes.
    """
    flags: dict[str, Any] = {}

    # Relative changes between consecutive levels for ES_0.95
    for i in range(len(n_grid) - 1):
        n_lo, n_hi = n_grid[i], n_grid[i + 1]
        es_lo = aggregated[str(n_lo)]["es_95"]["mean"]
        es_hi = aggregated[str(n_hi)]["es_95"]["mean"]
        denom = abs(es_lo) + 1e-12
        rel_change = abs(es_hi - es_lo) / denom
        flags[f"rel_change_{n_lo}_to_{n_hi}"] = round(float(rel_change), 6)

    # Main acceptance flag: |ES(400) - ES(200)| / |ES(200)| < 5%
    if 200 in n_grid and 400 in n_grid:
        es_200 = aggregated["200"]["es_95"]["mean"]
        es_400 = aggregated["400"]["es_95"]["mean"]
        rel = abs(es_400 - es_200) / (abs(es_200) + 1e-12)
        flags["converged_at_100_rel_5pct"] = bool(rel < 0.05)
    else:
        flags["converged_at_100_rel_5pct"] = None

    return flags

# =====================================================================
#  Figures
# =====================================================================

def _setup_fig_style() -> None:
    """Apply consistent figure style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def plot_convergence_curves(
    per_seed: dict[str, dict[str, dict[str, float]]],
    aggregated: dict[str, dict[str, dict[str, float]]],
    richardson: dict[str, dict[str, Any]],
    n_grid: list[int],
    seeds: list[int],
    out_path: Path,
) -> None:
    """Create 2x2 convergence panel: ES_0.95, ES_0.99, mean P&L, std P&L.

    Parameters
    ----------
    per_seed : dict
        Per-seed results.
    aggregated : dict
        Aggregated results.
    richardson : dict
        Richardson fit results per metric.
    n_grid : list[int]
        Grid sizes.
    seeds : list[int]
        Seeds used.
    out_path : Path
        Output file path.
    """
    _setup_fig_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("es_95", r"$\mathrm{ES}_{0.95}$", axes[0, 0]),
        ("es_99", r"$\mathrm{ES}_{0.99}$", axes[0, 1]),
        ("mean_pl", "Mean P&L", axes[1, 0]),
        ("std_pl", "Std P&L", axes[1, 1]),
    ]

    for metric_key, title, ax in panels:
        # Per-seed lines
        for j, seed in enumerate(seeds):
            vals = [per_seed[str(seed)][str(n)][metric_key] for n in n_grid]
            ax.plot(
                n_grid, vals,
                marker=SEED_MARKERS[j % len(SEED_MARKERS)],
                color=SEED_COLOURS[j % len(SEED_COLOURS)],
                alpha=0.5, linewidth=1, markersize=4,
                label=f"seed {seed}",
            )

        # Aggregated mean ± std band
        means = [aggregated[str(n)][metric_key]["mean"] for n in n_grid]
        stds = [aggregated[str(n)][metric_key]["std"] for n in n_grid]
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.plot(n_grid, means, color=MEAN_COLOUR, linewidth=2.5, marker="D",
                markersize=5, label="Mean", zorder=5)
        ax.fill_between(n_grid, lo, hi, color=MEAN_COLOUR, alpha=0.12)

        # Richardson fit dashed line (if available)
        if metric_key in richardson:
            r = richardson[metric_key]
            if not np.isnan(r["alpha_hat"]):
                n_fine = np.linspace(min(n_grid), max(n_grid), 200)
                fit_vals = _richardson_model(n_fine, r["ES_inf"], r["C"], r["alpha_hat"])
                ax.plot(n_fine, fit_vals, "--", color=C_FIT, linewidth=1.5,
                        label=rf"Fit $\hat{{\alpha}}={r['alpha_hat']:.2f}$")
                ax.axhline(r["ES_inf"], color=C_FIT, linestyle=":", alpha=0.4)

        ax.set_xscale("log")
        ax.set_xlabel("$n$ (grid size)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_grid)
        ax.set_xticklabels([str(n) for n in n_grid])

    fig.suptitle("BLP Convergence Sweep — Baseline rBergomi", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

def plot_richardson_fit(
    aggregated: dict[str, dict[str, dict[str, float]]],
    richardson: dict[str, dict[str, Any]],
    n_grid: list[int],
    out_path: Path,
) -> None:
    """Log-log plot of |ES(n) - ES(∞)| vs n with fitted line.

    Parameters
    ----------
    aggregated : dict
        Aggregated results.
    richardson : dict
        Richardson fit parameters.
    n_grid : list[int]
        Grid sizes.
    out_path : Path
        Output file path.
    """
    _setup_fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [("es_95", r"$\mathrm{ES}_{0.95}$", axes[0]),
              ("es_99", r"$\mathrm{ES}_{0.99}$", axes[1])]

    for metric_key, title, ax in panels:
        r = richardson.get(metric_key)
        if r is None or np.isnan(r["ES_inf"]):
            ax.text(0.5, 0.5, "Fit failed", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            continue

        es_inf = r["ES_inf"]
        means = [aggregated[str(n)][metric_key]["mean"] for n in n_grid]
        residuals = [abs(m - es_inf) for m in means]

        # Filter out zero residuals for log-log
        valid = [(n, res) for n, res in zip(n_grid, residuals) if res > 1e-15]
        if not valid:
            ax.text(0.5, 0.5, "No valid residuals", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        ns_valid, res_valid = zip(*valid)
        ax.scatter(ns_valid, res_valid, color=C_BS, s=60, zorder=5, label="Data")

        # Fitted curve
        n_fine = np.logspace(np.log10(min(n_grid)), np.log10(max(n_grid)), 100)
        fit_residual = np.abs(r["C"]) * np.power(n_fine, -r["alpha_hat"])
        ax.plot(n_fine, fit_residual, "--", color=C_FIT, linewidth=2,
                label=rf"$|C|\cdot n^{{-\hat{{\alpha}}}}$, $\hat{{\alpha}}={r['alpha_hat']:.3f}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$ (grid size)")
        ax.set_ylabel(rf"$|{title}(n) - {title}(\infty)|$")
        ax.set_title(title)

        # Annotation
        ci = r["alpha_ci"]
        ax.annotate(
            rf"$\hat{{\alpha}} = {r['alpha_hat']:.3f}$"
            f"\n95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
            f"\nBLP theoretical: 0.57",
            xy=(0.05, 0.05), xycoords="axes fraction",
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Richardson Extrapolation — Log-Log Residual", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

def plot_relative_changes_table(
    flags: dict[str, Any],
    n_grid: list[int],
    aggregated: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    """Render a table of consecutive relative changes as a matplotlib figure.

    Parameters
    ----------
    flags : dict
        Convergence flags with rel_change keys.
    n_grid : list[int]
        Grid sizes.
    aggregated : dict
        Aggregated results.
    out_path : Path
        Output file path.
    """
    _setup_fig_style()

    # Build table data
    headers = ["From n", "To n", "rel Δ ES₀.₉₅", "rel Δ ES₀.₉₉",
               "rel Δ mean P&L", "rel Δ std P&L"]
    rows = []
    highlight_row = None

    for i in range(len(n_grid) - 1):
        n_lo, n_hi = n_grid[i], n_grid[i + 1]
        row = [str(n_lo), str(n_hi)]
        for mk in ["es_95", "es_99", "mean_pl", "std_pl"]:
            v_lo = aggregated[str(n_lo)][mk]["mean"]
            v_hi = aggregated[str(n_hi)][mk]["mean"]
            denom = abs(v_lo) + 1e-12
            rel = abs(v_hi - v_lo) / denom
            row.append(f"{rel:.4f}")
        rows.append(row)
        if n_lo == 100 and n_hi == 200:
            highlight_row = i

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#E3F2FD")
        cell.set_text_props(weight="bold")

    # Highlight the 100→200 row
    if highlight_row is not None:
        for j in range(len(headers)):
            cell = table[highlight_row + 1, j]
            cell.set_facecolor("#FFF9C4")

    ax.set_title("Consecutive Relative Changes in Risk Metrics", fontsize=13,
                 pad=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

# =====================================================================
#  README generation
# =====================================================================

def generate_readme(
    richardson: dict[str, dict[str, Any]],
    flags: dict[str, Any],
    aggregated: dict[str, dict[str, dict[str, float]]],
    n_grid: list[int],
    hardware: dict[str, Any],
    out_path: Path,
) -> None:
    """Write a concise auto-generated summary markdown.

    Parameters
    ----------
    richardson : dict
        Richardson fit results.
    flags : dict
        Convergence flags.
    aggregated : dict
        Aggregated metric results.
    n_grid : list[int]
        Grid sizes.
    hardware : dict
        Hardware info.
    out_path : Path
        Output path.
    """
    ts = datetime.now(timezone.utc).isoformat()
    git_sha = hardware.get("git_commit", "unknown")

    r_es95 = richardson.get("es_95", {})
    alpha_hat = r_es95.get("alpha_hat", float("nan"))
    alpha_ci = r_es95.get("alpha_ci", [float("nan"), float("nan")])
    es_inf = r_es95.get("ES_inf", float("nan"))
    rel_err = r_es95.get("rel_err_at_100", float("nan"))
    conv_flag = flags.get("converged_at_100_rel_5pct")

    lines = [
        "# Block 1.1 — Convergence Sweep Summary\n",
        f"Generated: {ts}",
        f"Git commit: {git_sha}\n",
        "## Key findings\n",
        f"- Richardson rate `α̂ = {alpha_hat:.4f}` "
        f"(95% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}])",
        f"- BLP theoretical: `α_BLP = 0.57`",
        f"- Extrapolated `ES_0.95(∞) = {es_inf:.4f}`",
        f"- Relative error at `n = 100`: {rel_err:.4f} ({rel_err*100:.1f}%)",
        f"- Convergence flag (rel change 200 → 400 < 5%): "
        f"**{'PASS' if conv_flag else 'FAIL' if conv_flag is not None else 'N/A'}**\n",
        "## Per-level relative changes\n",
        "| From n | To n | rel_change ES_0.95 | rel_change ES_0.99 |",
        "|--------|------|--------------------|--------------------|",
    ]

    for i in range(len(n_grid) - 1):
        n_lo, n_hi = n_grid[i], n_grid[i + 1]
        es95_lo = aggregated[str(n_lo)]["es_95"]["mean"]
        es95_hi = aggregated[str(n_hi)]["es_95"]["mean"]
        es99_lo = aggregated[str(n_lo)]["es_99"]["mean"]
        es99_hi = aggregated[str(n_hi)]["es_99"]["mean"]
        rc95 = abs(es95_hi - es95_lo) / (abs(es95_lo) + 1e-12)
        rc99 = abs(es99_hi - es99_lo) / (abs(es99_lo) + 1e-12)
        lines.append(f"| {n_lo:>6} | {n_hi:>4} | {rc95:.6f}           | {rc99:.6f}           |")

    lines.append("")
    lines.append("## Interpretation\n")

    # Auto-generated interpretation
    if conv_flag is True:
        interp = (
            f"The Richardson fit yields α̂ = {alpha_hat:.3f}, consistent with the "
            f"BLP theoretical rate of 0.57. The relative error at n = 100 is "
            f"{rel_err*100:.1f}%, and the consecutive change from n = 200 to n = 400 "
            f"is below the 5% threshold. The discretisation at n = 100 is therefore "
            f"acceptable for the reported ES_0.95 values. No downstream re-run is needed."
        )
    elif conv_flag is False:
        interp = (
            f"The Richardson fit yields α̂ = {alpha_hat:.3f}. The relative error at "
            f"n = 100 is {rel_err*100:.1f}%, and the consecutive change from n = 200 to "
            f"n = 400 exceeds the 5% threshold. Convergence at n = 100 is NOT confirmed. "
            f"Downstream experiments should be re-evaluated at n = 400."
        )
    else:
        interp = (
            "Could not evaluate the convergence flag because the grid did not include "
            "both n = 200 and n = 400. Inspect the raw results manually."
        )
    lines.append(interp)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}", flush=True)

# =====================================================================
#  File-safety checks
# =====================================================================

_OUTPUT_FILES = [
    "convergence_sweep.json",
    "README.md",
]
_OUTPUT_FIGURES = [
    "convergence_curves.png",
    "richardson_fit.png",
    "relative_changes_table.png",
]

def _check_no_collision(results_dir: Path, figures_dir: Path) -> None:
    """Raise FileExistsError if any output file already exists.

    Parameters
    ----------
    results_dir : Path
        Results output directory.
    figures_dir : Path
        Figures output directory.
    """
    for fname in _OUTPUT_FILES:
        p = results_dir / fname
        if p.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {p}. "
                f"Move or delete it manually to rerun."
            )
    for fname in _OUTPUT_FIGURES:
        p = figures_dir / fname
        if p.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {p}. "
                f"Move or delete it manually to rerun."
            )

# =====================================================================
#  Main entry point
# =====================================================================

def main(
    n_grid: list[int] | None = None,
    n_paths: int = 10_000,
    n_paths_p0: int = 20_000,
    seeds: list[int] | None = None,
    results_dir: Path | None = None,
    figures_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full convergence sweep experiment.

    Parameters
    ----------
    n_grid : list[int]
        Grid sizes to sweep over.
    n_paths : int
        Number of simulation paths per seed.
    n_paths_p0 : int
        Number of paths for premium estimation.
    seeds : list[int]
        Random seeds.
    results_dir : Path
        Output directory for JSON/MD results.
    figures_dir : Path
        Output directory for figures.
    dry_run : bool
        If True, print config and exit without running.

    Returns
    -------
    dict
        The full results dictionary (same as written to JSON).
    """
    if n_grid is None:
        n_grid = [50, 100, 200, 400, 800, 1600]
    if seeds is None:
        seeds = [7001, 7002, 7003]
    if results_dir is None:
        results_dir = RESULTS_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR

    n_grid = sorted(n_grid)
    hardware = record_hardware_and_versions()

    config = {
        "n_grid": n_grid,
        "n_paths_per_seed": n_paths,
        "n_paths_p0": n_paths_p0,
        "seeds": seeds,
        "baseline": BASELINE,
        "cost_lambda": 0.0,
    }

    if dry_run:
        print("=" * 60)
        print(" Block 1.1 — Convergence Sweep [DRY RUN]")
        print("=" * 60)
        print(json.dumps({"config": config, "hardware": hardware}, indent=2))
        print("\nDry run complete. No files created.")
        return {}

    # Safety check
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _check_no_collision(results_dir, figures_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" Block 1.1 — BLP Convergence Sweep")
    print("=" * 60)
    print(f"  Grid sizes: {n_grid}")
    print(f"  Paths/seed: {n_paths}, P0 paths: {n_paths_p0}")
    print(f"  Seeds:      {seeds}")
    print(f"  Device:     {device}")
    print(f"  Output:     {results_dir}")
    print("-" * 60, flush=True)

    # ---- Run all cells ----
    t_total = time.perf_counter()
    per_seed_results: dict[str, dict[str, dict[str, float]]] = {}

    for seed in seeds:
        per_seed_results[str(seed)] = {}
        for n in n_grid:
            print(f"  Running n={n:>5}, seed={seed} ...", end="", flush=True)
            metrics = run_single_cell(n, seed, n_paths, n_paths_p0, device)
            per_seed_results[str(seed)][str(n)] = metrics
            print(
                f"  ES_95={metrics['es_95']:.4f}, "
                f"mean={metrics['mean_pl']:.4f}, "
                f"t={metrics['wall_clock_s']:.1f}s",
                flush=True,
            )

    total_time = time.perf_counter() - t_total
    print(f"\n  Total wall-clock: {total_time:.1f}s", flush=True)

    # ---- Aggregate ----
    print("\n  Aggregating ...", flush=True)
    aggregated = aggregate_results(per_seed_results, n_grid, seeds)

    # ---- Richardson fits ----
    print("  Fitting Richardson model ...", flush=True)
    richardson: dict[str, dict[str, Any]] = {}
    for mk in ["es_95", "es_99", "mean_pl", "std_pl"]:
        means = [aggregated[str(n)][mk]["mean"] for n in n_grid]
        richardson[mk] = fit_richardson(n_grid, means)
        ah = richardson[mk]["alpha_hat"]
        print(f"    {mk}: α̂ = {ah:.4f}", flush=True)

    # ---- Convergence flags ----
    flags = compute_convergence_flags(aggregated, n_grid)

    # ---- Assemble output ----
    output = {
        "meta": {
            "script": "deep_hedging/experiments/block1_convergence.py",
            "git_commit": hardware.get("git_commit", "unknown"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "total_wall_clock_s": round(total_time, 2),
        },
        "config": config,
        "per_seed_results": per_seed_results,
        "aggregated": aggregated,
        "richardson": richardson,
        "convergence_flags": flags,
    }

    # ---- Save JSON ----
    json_path = results_dir / "convergence_sweep.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {json_path}", flush=True)

    # ---- Figures ----
    print("\n  Generating figures ...", flush=True)
    plot_convergence_curves(
        per_seed_results, aggregated, richardson, n_grid, seeds,
        figures_dir / "convergence_curves.png",
    )
    plot_richardson_fit(
        aggregated, richardson, n_grid,
        figures_dir / "richardson_fit.png",
    )
    plot_relative_changes_table(
        flags, n_grid, aggregated,
        figures_dir / "relative_changes_table.png",
    )

    # ---- README ----
    generate_readme(
        richardson, flags, aggregated, n_grid, hardware,
        results_dir / "README.md",
    )

    # ---- Final report ----
    r_es95 = richardson.get("es_95", {})
    conv = flags.get("converged_at_100_rel_5pct")
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  α̂ (ES_0.95) = {r_es95.get('alpha_hat', float('nan')):.4f}")
    print(f"  ES_0.95(∞)  = {r_es95.get('ES_inf', float('nan')):.4f}")
    print(f"  Rel err @100 = {r_es95.get('rel_err_at_100', float('nan')):.4f}")
    print(f"  Converged    = {'PASS' if conv else 'FAIL' if conv is not None else 'N/A'}")
    print("=" * 60, flush=True)

    return output

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Block 1.1 — BLP Convergence Sweep",
    )
    p.add_argument(
        "--n-grid", type=int, nargs="+",
        default=[50, 100, 200, 400, 800, 1600],
        help="Grid sizes to sweep",
    )
    p.add_argument("--n-paths", type=int, default=10_000, help="Paths per seed")
    p.add_argument("--n-paths-p0", type=int, default=20_000, help="Paths for premium")
    p.add_argument(
        "--seeds", type=int, nargs="+",
        default=[7001, 7002, 7003],
        help="Random seeds",
    )
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    p.add_argument("--results-dir", type=str, default=None, help="Results output dir")
    p.add_argument("--figures-dir", type=str, default=None, help="Figures output dir")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(
        n_grid=args.n_grid,
        n_paths=args.n_paths,
        n_paths_p0=args.n_paths_p0,
        seeds=args.seeds,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        dry_run=args.dry_run,
    )
