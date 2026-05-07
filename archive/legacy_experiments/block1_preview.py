#!/usr/bin/env python
"""
Block 1 Preview — Quick Convergence Pre-Flight Check (P01.5).

Runs a reduced-budget convergence sweep (3 n-levels × 1 seed × 2k paths)
to get a first-order estimate before committing to the full P01 experiment.
Produces a verdict (GREEN/YELLOW/ORANGE/RED) and recommended next action.

All outputs go into results/block1_preview/ and figures/block1_preview/.
No existing files are modified.

Run:
    python -u -m deep_hedging.experiments.block1_preview
"""
from __future__ import annotations

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

from deep_hedging.experiments.block1_convergence import (
    BASELINE,
    C_BS,
    C_FIT,
    MEAN_COLOUR,
    _richardson_model,
    aggregate_results,
    fit_richardson,
    run_single_cell,
)
from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

# ─── Output directories ────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "block1_preview"
FIGURES_DIR = REPO_ROOT / "figures" / "block1_preview"

# ─── Preview configuration ─────────────────────────────────────
PREVIEW_N_GRID = [50, 100, 400]
PREVIEW_SEEDS = [7101]
PREVIEW_N_PATHS = 2000
PREVIEW_N_PATHS_P0 = 4000


# =====================================================================
#  Verdict logic
# =====================================================================

def _compute_verdict(
    metrics_by_n: dict[str, dict[str, float]],
) -> tuple[str, str, str]:
    """Determine GREEN/YELLOW/ORANGE/RED verdict.

    Parameters
    ----------
    metrics_by_n : dict
        Mapping n_str -> metrics dict (with "es_95" key).

    Returns
    -------
    tuple of (verdict, explanation, next_action)
    """
    es_100 = metrics_by_n["100"]["es_95"]
    es_400 = metrics_by_n["400"]["es_95"]
    es_50 = metrics_by_n["50"]["es_95"]

    rel_100_400 = abs(es_400 - es_100) / (abs(es_100) + 1e-12)
    rel_50_100 = abs(es_100 - es_50) / (abs(es_50) + 1e-12)

    # Check for pathological values
    all_vals = [es_50, es_100, es_400]
    has_nan = any(np.isnan(v) or np.isinf(v) for v in all_vals)
    has_negative_es = any(v < 0 for v in all_vals)

    if has_nan or has_negative_es:
        return (
            "RED",
            f"Pathological values detected: ES_0.95 = {all_vals}. "
            f"NaN/Inf or negative ES indicates a simulator or numerical issue.",
            "Halt and debug. Run validate_simulator.py and inspect volterra.py "
            "for regression before any further experiments.",
        )

    if rel_100_400 > 0.20:
        return (
            "RED",
            f"Relative change n=100→400 is {rel_100_400:.1%}, exceeding 20%. "
            f"This suggests severe discretisation bias at n=100 or a potential "
            f"bug in the hybrid scheme. ES_0.95: n=50→{es_50:.2f}, "
            f"n=100→{es_100:.2f}, n=400→{es_400:.2f}.",
            "Halt and debug. Run validate_simulator.py immediately. "
            "If it passes, run exact Cholesky smoke-test (P02 scaffolding) "
            "on 500 paths at n=100 to isolate hybrid-scheme bias. "
            "Do not run full P01.",
        )

    if rel_100_400 > 0.10:
        return (
            "ORANGE",
            f"Relative change n=100→400 is {rel_100_400:.1%} (10-20% range). "
            f"Non-trivial discretisation bias present. ES_0.95: n=50→{es_50:.2f}, "
            f"n=100→{es_100:.2f}, n=400→{es_400:.2f}. The convergence curve is "
            f"monotone but slow.",
            "Investigate before full sweep. Run exact Cholesky smoke-test "
            "(P02 scaffolding) on 500 paths at n=100 to isolate hybrid-scheme "
            "bias from inherent roughness convergence. Then decide on full P01.",
        )

    if rel_100_400 > 0.03:
        return (
            "YELLOW",
            f"Relative change n=100→400 is {rel_100_400:.1%} (3-10% range). "
            f"Convergence is borderline. ES_0.95: n=50→{es_50:.2f}, "
            f"n=100→{es_100:.2f}, n=400→{es_400:.2f}. Full sweep will clarify "
            f"the picture.",
            "Proceed with full P01 sweep. Also prepare to extend grid to "
            "n=800,1600 if the full sweep confirms borderline convergence.",
        )

    return (
        "GREEN",
        f"Relative change n=100→400 is {rel_100_400:.1%} (<3%). "
        f"Sampling variance likely masks any signal. ES_0.95: n=50→{es_50:.2f}, "
        f"n=100→{es_100:.2f}, n=400→{es_400:.2f}. Convergence looks plausible.",
        "Proceed to full P01 sweep. The investment is safe.",
    )


# =====================================================================
#  Figures
# =====================================================================

def plot_preview_convergence(
    metrics_by_n: dict[str, dict[str, float]],
    n_grid: list[int],
    out_path: Path,
) -> None:
    """Simple convergence plot for preview.

    Parameters
    ----------
    metrics_by_n : dict
        Metrics per n level.
    n_grid : list[int]
        Grid sizes.
    out_path : Path
        Output PNG path.
    """
    plt.rcParams.update({
        "font.size": 11, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        ("es_95", r"$\mathrm{ES}_{0.95}$"),
        ("es_99", r"$\mathrm{ES}_{0.99}$"),
        ("mean_pl", "Mean P&L"),
    ]

    for (mk, title), ax in zip(panels, axes):
        vals = [metrics_by_n[str(n)][mk] for n in n_grid]
        ax.plot(n_grid, vals, "o-", color=C_BS, linewidth=2, markersize=8)
        for n_val, v in zip(n_grid, vals):
            ax.annotate(f"{v:.2f}", (n_val, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("$n$ (grid size)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_grid)
        ax.set_xticklabels([str(n) for n in n_grid])

    fig.suptitle("Convergence Preview — 1 seed × 2k paths", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# =====================================================================
#  Decision markdown
# =====================================================================

def generate_decision_md(
    metrics_by_n: dict[str, dict[str, float]],
    n_grid: list[int],
    verdict: str,
    explanation: str,
    next_action: str,
    hardware: dict[str, Any],
    wall_clock: float,
    out_path: Path,
) -> None:
    """Write auto-generated decision.md.

    Parameters
    ----------
    metrics_by_n : dict
        Metrics per n level.
    n_grid : list[int]
        Grid sizes.
    verdict : str
        GREEN/YELLOW/ORANGE/RED.
    explanation : str
        One paragraph explanation.
    next_action : str
        Recommended next step.
    hardware : dict
        Hardware metadata.
    wall_clock : float
        Total seconds.
    out_path : Path
        Output path.
    """
    ts = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Convergence Preview — Decision\n",
        f"Run completed: {ts}",
        f"Budget: {len(n_grid)} n-levels × 1 seed × {PREVIEW_N_PATHS} paths",
        f"Wall-clock: {wall_clock:.1f}s\n",
        "## Numbers\n",
        "| n | ES_0.95 | ES_0.99 | Mean P&L | Std P&L | Wall-clock |",
        "|---|---------|---------|----------|---------|------------|",
    ]
    for n in n_grid:
        m = metrics_by_n[str(n)]
        lines.append(
            f"| {n} | {m['es_95']:.4f} | {m['es_99']:.4f} | "
            f"{m['mean_pl']:.4f} | {m['std_pl']:.4f} | {m['wall_clock_s']:.1f}s |"
        )

    lines.append("")
    lines.append("## Relative changes\n")

    for i in range(len(n_grid) - 1):
        n_lo, n_hi = n_grid[i], n_grid[i + 1]
        es_lo = metrics_by_n[str(n_lo)]["es_95"]
        es_hi = metrics_by_n[str(n_hi)]["es_95"]
        rel = abs(es_hi - es_lo) / (abs(es_lo) + 1e-12)
        lines.append(f"- `n={n_lo} → n={n_hi}`: ES_0.95 changes by {rel:.1%}.")

    lines.append("")
    lines.append(f"## Verdict: **{verdict}**\n")
    lines.append(explanation)
    lines.append("")
    lines.append("## Recommended next action\n")
    lines.append(next_action)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}", flush=True)


# =====================================================================
#  File-safety
# =====================================================================

_OUTPUT_FILES_R = ["preview_results.json", "decision.md"]
_OUTPUT_FILES_F = ["preview_convergence.png"]


def _check_no_collision(results_dir: Path, figures_dir: Path) -> None:
    """Raise FileExistsError if any output file already exists."""
    for fname in _OUTPUT_FILES_R:
        p = results_dir / fname
        if p.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {p}. "
                f"Move or delete it manually to rerun."
            )
    for fname in _OUTPUT_FILES_F:
        p = figures_dir / fname
        if p.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {p}. "
                f"Move or delete it manually to rerun."
            )


# =====================================================================
#  Main
# =====================================================================

def main() -> dict[str, Any]:
    """Run the preview convergence check.

    Returns
    -------
    dict
        Full results dictionary.
    """
    results_dir = RESULTS_DIR
    figures_dir = FIGURES_DIR

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _check_no_collision(results_dir, figures_dir)

    hardware = record_hardware_and_versions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" Block 1 Preview — Quick Convergence Check")
    print("=" * 60)
    print(f"  Grid:   {PREVIEW_N_GRID}")
    print(f"  Paths:  {PREVIEW_N_PATHS} (P0: {PREVIEW_N_PATHS_P0})")
    print(f"  Seeds:  {PREVIEW_SEEDS}")
    print(f"  Device: {device}")
    print("-" * 60, flush=True)

    t_total = time.perf_counter()

    # ---- Run cells ----
    metrics_by_n: dict[str, dict[str, float]] = {}
    for n in PREVIEW_N_GRID:
        print(f"  Running n={n:>5} ...", end="", flush=True)
        m = run_single_cell(n, PREVIEW_SEEDS[0], PREVIEW_N_PATHS,
                            PREVIEW_N_PATHS_P0, device)
        metrics_by_n[str(n)] = m
        print(f"  ES_95={m['es_95']:.4f}, mean={m['mean_pl']:.4f}, "
              f"t={m['wall_clock_s']:.1f}s", flush=True)

    wall_clock = time.perf_counter() - t_total

    # ---- Verdict ----
    verdict, explanation, next_action = _compute_verdict(metrics_by_n)

    # ---- Save JSON ----
    output = {
        "meta": {
            "script": "deep_hedging/experiments/block1_preview.py",
            "git_commit": hardware.get("git_commit", "unknown"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "total_wall_clock_s": round(wall_clock, 2),
        },
        "config": {
            "n_grid": PREVIEW_N_GRID,
            "n_paths": PREVIEW_N_PATHS,
            "n_paths_p0": PREVIEW_N_PATHS_P0,
            "seeds": PREVIEW_SEEDS,
            "baseline": BASELINE,
        },
        "metrics_by_n": metrics_by_n,
        "verdict": verdict,
        "explanation": explanation,
        "next_action": next_action,
    }

    json_path = results_dir / "preview_results.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\n  Saved: {json_path}", flush=True)

    # ---- Figure ----
    plot_preview_convergence(metrics_by_n, PREVIEW_N_GRID,
                             figures_dir / "preview_convergence.png")

    # ---- Decision markdown ----
    generate_decision_md(
        metrics_by_n, PREVIEW_N_GRID,
        verdict, explanation, next_action,
        hardware, wall_clock,
        results_dir / "decision.md",
    )

    # ---- Print verdict ----
    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 60}")
    print(f"  {explanation}")
    print(f"\n  Next: {next_action}")
    print(f"{'=' * 60}", flush=True)

    return output


if __name__ == "__main__":
    main()
