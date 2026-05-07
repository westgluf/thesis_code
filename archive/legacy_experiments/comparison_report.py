#!/usr/bin/env python
"""
Generate the comparison report for the canonical re-run results.

Reads:
  - results/canonical_v2/baseline_5seeds.json
  - results/canonical_v2/decomposition_5seeds.json
  - figures/section_63_metrics.json   (old single-run canonical)
  - figures/decomposition_closed.json (old decomposition)

Produces:
  - results/canonical_v2/comparison_report.md
  - figures/canonical_v2/gamma_5seeds.png
  - figures/canonical_v2/decomposition_5seeds.png
  - figures/canonical_v2/es_distribution_comparison.png

Run:
    python -u -m deep_hedging.experiments.comparison_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "canonical_v2"
FIGURES_DIR = REPO_ROOT / "figures" / "canonical_v2"
BASELINE_JSON = RESULTS_DIR / "baseline_5seeds.json"
DECOMP_JSON = RESULTS_DIR / "decomposition_5seeds.json"
REPORT_PATH = RESULTS_DIR / "comparison_report.md"

OLD_BASELINE_JSON = REPO_ROOT / "figures" / "section_63_metrics.json"
OLD_DECOMP_JSON = REPO_ROOT / "figures" / "decomposition_closed.json"

BASELINE_SEEDS = [2024, 2025, 2026, 2027, 2028]
DECOMP_SEEDS = [3024, 3025, 3026, 3027, 3028]


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


def _delta_str(new: float, old: float) -> str:
    """Format '(abs=..., rel=...%)'."""
    da = new - old
    dr = (new - old) / old * 100 if abs(old) > 1e-12 else float("nan")
    return f"{da:+.4f} ({dr:+.1f}%)"


def _fmt_mean_std(agg: dict[str, float], fmt: str = ".4f") -> str:
    return f"{agg['mean']:{fmt}} ± {agg['std']:{fmt}}"


def build_report(
    baseline_json: dict[str, Any],
    decomp_json: dict[str, Any],
    old_baseline_data: dict[str, Any],
    old_decomp_data: dict[str, Any],
) -> str:
    """Build the markdown report content."""
    ts = dt.datetime.now().isoformat(timespec="seconds")

    # ---------- reproducibility ----------
    lines: list[str] = []
    lines.append("# Canonical Re-run Results — Post Seeding Fix")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append(f"Git commit (pre-fix): cdbd9f1 (Pre-fix snapshot)")
    lines.append(f"Git commit (post-fix): {baseline_json['meta']['git_commit']}")
    lines.append("")

    # Reproducibility check section
    lines.append("## Reproducibility verification")
    lines.append("")

    lines.append("### Baseline re-run (seed 2024)")
    lines.append("")
    repro_b = baseline_json.get("reproducibility_check")
    if repro_b is None or "error" in (repro_b or {}):
        lines.append("_Not available — reproducibility subprocess check was skipped "
                     "or failed._")
        lines.append("")
    else:
        lines.append("| Metric | Run 1 | Run 2 | Match? |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Γ (λ=0.0) | {repro_b['run1']['gamma']:.6f} | "
                     f"{repro_b['run2']['gamma']:.6f} | "
                     f"{'✓' if repro_b['gamma_match'] else '✗'} |")
        lines.append(f"| ES_0.95_DH | {repro_b['run1']['es95_dh']:.6f} | "
                     f"{repro_b['run2']['es95_dh']:.6f} | "
                     f"{'✓' if repro_b['es95_dh_match'] else '✗'} |")
        lines.append(f"| first_weight_sum | {repro_b['run1']['first_weight_sum']:.6f} | "
                     f"{repro_b['run2']['first_weight_sum']:.6f} | "
                     f"{'✓' if repro_b['first_weight_sum_match'] else '✗'} |")
        lines.append("")
        lines.append(f"Verdict: **{repro_b['verdict']}**")
        lines.append("")

    lines.append("### Decomposition re-run (seed 3024)")
    lines.append("")
    repro_d = decomp_json.get("reproducibility_check")
    if repro_d is None or "error" in (repro_d or {}):
        lines.append("_Not available — reproducibility subprocess check was skipped "
                     "or failed._")
        lines.append("")
    else:
        lines.append("| Metric | Run 1 | Run 2 | Match? |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Γ_total | {repro_d['run1']['gamma_total']:.6f} | "
                     f"{repro_d['run2']['gamma_total']:.6f} | "
                     f"{'✓' if repro_d['gamma_match'] else '✗'} |")
        lines.append(f"| Objective % | {repro_d['run1']['objective_pct']:.4f} | "
                     f"{repro_d['run2']['objective_pct']:.4f} | "
                     f"{'✓' if repro_d['objective_pct_match'] else '✗'} |")
        lines.append(f"| ES_A_dh | {repro_d['run1']['esA_dh']:.6f} | "
                     f"{repro_d['run2']['esA_dh']:.6f} | "
                     f"{'✓' if repro_d['experiment_A_match'] else '✗'} |")
        lines.append("")
        lines.append(f"Verdict: **{repro_d['verdict']}**")
        lines.append("")

    # ---------- baseline comparison ----------
    lines.append("## Baseline comparison (λ=0)")
    lines.append("")
    agg = baseline_json["aggregated"]["0.0"]
    old = old_baseline_data.get("lambda_0.0", {})

    lines.append("| Metric | Old (single run) | New (mean ± std, 5 seeds) | Δ abs | Δ rel |")
    lines.append("|---|---|---|---|---|")
    for key, label, fmt in [
        ("es95_bs", "ES_0.95 BS", ".4f"),
        ("es95_dh", "ES_0.95 DH", ".4f"),
        ("gamma", "Γ", ".4f"),
        ("mean_pl_dh", "Mean PL (DH)", ".4f"),
        ("std_pl_dh", "Std PL (DH)", ".4f"),
    ]:
        old_val = old.get(key, None)
        new_mean = agg[key]["mean"]
        new_std = agg[key]["std"]
        if old_val is not None:
            da = new_mean - old_val
            dr = (da / old_val * 100) if abs(old_val) > 1e-12 else float("nan")
            old_str = f"{old_val:{fmt}}"
            da_str = f"{da:+{fmt}}"
            dr_str = f"{dr:+.1f}%"
        else:
            old_str = "N/A"
            da_str = "N/A"
            dr_str = "N/A"
        lines.append(f"| {label} | {old_str} | "
                     f"{new_mean:{fmt}} ± {new_std:{fmt}} | {da_str} | {dr_str} |")
    lines.append("")

    # With-costs comparison
    if "0.001" in baseline_json["aggregated"]:
        lines.append("## Baseline comparison (λ=0.001)")
        lines.append("")
        agg_c = baseline_json["aggregated"]["0.001"]
        old_c = old_baseline_data.get("lambda_0.001", {})
        lines.append("| Metric | Old (single run) | New (mean ± std, 5 seeds) | Δ abs | Δ rel |")
        lines.append("|---|---|---|---|---|")
        for key, label, fmt in [
            ("es95_bs", "ES_0.95 BS", ".4f"),
            ("es95_dh", "ES_0.95 DH", ".4f"),
            ("gamma", "Γ", ".4f"),
        ]:
            old_val = old_c.get(key, None)
            new_mean = agg_c[key]["mean"]
            new_std = agg_c[key]["std"]
            if old_val is not None:
                da = new_mean - old_val
                dr = (da / old_val * 100) if abs(old_val) > 1e-12 else float("nan")
                old_str = f"{old_val:{fmt}}"
                da_str = f"{da:+{fmt}}"
                dr_str = f"{dr:+.1f}%"
            else:
                old_str = "N/A"
                da_str = "N/A"
                dr_str = "N/A"
            lines.append(f"| {label} | {old_str} | "
                         f"{new_mean:{fmt}} ± {new_std:{fmt}} | {da_str} | {dr_str} |")
        lines.append("")

    # ---------- decomposition ----------
    lines.append("## Decomposition comparison")
    lines.append("")
    pct_agg = decomp_json["aggregated"]["percentages"]
    old_pct = old_decomp_data.get("percentages", {})

    lines.append("| Bucket | Old % | New mean % ± std | Δ (pp) |")
    lines.append("|---|---|---|---|")
    for key, label in [
        ("objective", "Objective"),
        ("interaction", "Interaction"),
        ("stoch_vol", "Stoch vol"),
        ("roughness", "Roughness"),
        ("architecture", "Architecture"),
    ]:
        old_val = old_pct.get(key, None)
        new_mean = pct_agg[key]["mean"]
        new_std = pct_agg[key]["std"]
        if old_val is not None:
            da = new_mean - old_val
            old_str = f"{old_val:+.2f}"
            da_str = f"{da:+.2f} pp"
        else:
            old_str = "N/A"
            da_str = "N/A"
        lines.append(f"| {label} | {old_str} | {new_mean:+.2f} ± {new_std:.2f} | {da_str} |")
    lines.append("")

    # Old and new Gamma_total
    old_gt = old_decomp_data.get("Gamma_total", None)
    new_gt = decomp_json["aggregated"]["absolute"]["Gamma_total"]
    lines.append(f"Γ_total (decomposition baseline): "
                 f"old = {old_gt:.4f}; new mean = {new_gt['mean']:.4f} ± {new_gt['std']:.4f}"
                 if old_gt is not None
                 else f"Γ_total (decomposition baseline): new = {new_gt['mean']:.4f} ± {new_gt['std']:.4f}")
    lines.append("")

    # ---------- qualitative ----------
    lines.append("## Qualitative assessment")
    lines.append("")

    # Does Γ > 0 across all 5 seeds?
    gammas_per_seed = [
        baseline_json["per_seed"][str(s)]["0.0"]["gamma"] for s in BASELINE_SEEDS
    ]
    all_pos = all(g > 0 for g in gammas_per_seed)
    lines.append(f"- Does Γ > 0 across all 5 seeds (baseline, λ=0)? "
                 f"**{'YES' if all_pos else 'NO'}** (values: "
                 + ", ".join(f"{g:+.4f}" for g in gammas_per_seed) + ")")

    # Is ranking preserved?
    ranking_per_seed_new = []
    for s in DECOMP_SEEDS:
        pct = decomp_json["per_seed"][str(s)]["decomposition"]["percentages_of_total"]
        ranking = sorted(
            ["objective", "interaction", "stoch_vol", "roughness", "architecture"],
            key=lambda k: -pct[k],
        )
        ranking_per_seed_new.append(ranking)

    # Expected old ranking: objective > interaction > stoch_vol > roughness > architecture
    expected = ["objective", "interaction", "stoch_vol", "roughness", "architecture"]
    ranking_preserved = all(r == expected for r in ranking_per_seed_new)
    lines.append(f"- Is decomposition ranking preserved (objective > interaction > "
                 f"stoch vol > roughness > architecture)? "
                 f"**{'YES' if ranking_preserved else 'NO'}**")
    if not ranking_preserved:
        for i, r in enumerate(ranking_per_seed_new):
            lines.append(f"  - seed {DECOMP_SEEDS[i]}: {' > '.join(r)}")

    # Does old Γ = 1.172 fall within new mean ± 2σ?
    new_gamma_mean = agg["gamma"]["mean"]
    new_gamma_std = agg["gamma"]["std"]
    old_gamma = old_baseline_data.get("lambda_0.0", {}).get("gamma", None)
    in_2sig = None
    if old_gamma is not None:
        lo2 = new_gamma_mean - 2 * new_gamma_std
        hi2 = new_gamma_mean + 2 * new_gamma_std
        in_2sig = lo2 <= old_gamma <= hi2
        lines.append(f"- Does old Γ ({old_gamma:+.4f}) fall within new mean ± 2σ "
                     f"([{lo2:+.4f}, {hi2:+.4f}])? **{'YES' if in_2sig else 'NO'}**")

    lines.append("")

    # ---------- verdict ----------
    lines.append("## Verdict")
    lines.append("")
    if not all_pos or not ranking_preserved:
        verdict = "QUALITATIVE_CHANGE"
    elif in_2sig is False:
        verdict = "NUMBERS_SHIFTED_QUALITATIVELY_PRESERVED"
    else:
        verdict = "NUMBERS_CONSISTENT"

    verdict_explanations = {
        "NUMBERS_CONSISTENT": (
            "Old values within 2σ of new mean; qualitative findings preserved; "
            "proceed with text update only."
        ),
        "NUMBERS_SHIFTED_QUALITATIVELY_PRESERVED": (
            "Old values outside 2σ but sign and ranking preserved; update all tables "
            "in dissertation."
        ),
        "QUALITATIVE_CHANGE": (
            "Sign flip or ranking change; major revision required."
        ),
    }
    lines.append(f"**{verdict}** — {verdict_explanations[verdict]}")
    lines.append("")

    # ---------- per-seed detail ----------
    lines.append("## Per-seed detail (baseline, λ=0)")
    lines.append("")
    lines.append("| Seed | ES_BS | ES_DH | Γ | Mean PL DH | Std PL DH |")
    lines.append("|---|---|---|---|---|---|")
    for s in BASELINE_SEEDS:
        ps = baseline_json["per_seed"][str(s)]["0.0"]
        lines.append(f"| {s} | {ps['es95_bs']:.4f} | {ps['es95_dh']:.4f} | "
                     f"{ps['gamma']:+.4f} | {ps['mean_pl_dh']:+.4f} | {ps['std_pl_dh']:.4f} |")
    lines.append("")

    lines.append("## Per-seed detail (decomposition)")
    lines.append("")
    lines.append("| Seed | Γ_total | Obj% | Int% | SV% | R% | Arch% |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in DECOMP_SEEDS:
        ps = decomp_json["per_seed"][str(s)]["decomposition"]
        pct = ps["percentages_of_total"]
        lines.append(f"| {s} | {ps['Gamma_total']:+.4f} | "
                     f"{pct['objective']:+.2f} | {pct['interaction']:+.2f} | "
                     f"{pct['stoch_vol']:+.2f} | {pct['roughness']:+.2f} | "
                     f"{pct['architecture']:+.2f} |")
    lines.append("")

    # ---------- methodology ----------
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- Baseline: {baseline_json['meta']['script']} with seeds "
                 f"{baseline_json['meta']['seeds']}")
    lines.append(f"- Data: n_train={baseline_json['meta']['n_train']}, "
                 f"n_val={baseline_json['meta']['n_val']}, "
                 f"n_test={baseline_json['meta']['n_test']}")
    lines.append(f"- Training: epochs={baseline_json['meta']['epochs']}, "
                 f"patience={baseline_json['meta']['patience']}, "
                 f"batch_size={baseline_json['meta']['batch_size']}, "
                 f"lr={baseline_json['meta']['lr']}")
    lines.append(f"- Decomposition: {decomp_json['meta']['script']} with seeds "
                 f"{decomp_json['meta']['seeds']}")
    lines.append("")

    return "\n".join(lines)


def plot_gamma_5seeds(
    baseline_json: dict[str, Any],
    old_baseline_data: dict[str, Any],
    path: Path,
) -> None:
    """Bar chart of Γ per seed with mean±std band and old canonical value."""
    fig, ax = plt.subplots(figsize=(9, 5))

    seeds = BASELINE_SEEDS
    gammas_new = [baseline_json["per_seed"][str(s)]["0.0"]["gamma"] for s in seeds]
    new_mean = baseline_json["aggregated"]["0.0"]["gamma"]["mean"]
    new_std = baseline_json["aggregated"]["0.0"]["gamma"]["std"]

    x = np.arange(len(seeds))
    bars = ax.bar(x, gammas_new, color="#4CAF50", alpha=0.8, edgecolor="k", lw=0.8,
                  label=f"Per-seed Γ (new)")
    for xi, gi in zip(x, gammas_new):
        ax.text(xi, gi + 0.015, f"{gi:.3f}", ha="center", va="bottom", fontsize=9)

    # Mean ± std band
    ax.axhline(new_mean, color="#2E7D32", lw=1.5, ls="-",
               label=f"New mean = {new_mean:.4f}")
    ax.axhspan(new_mean - new_std, new_mean + new_std,
               color="#81C784", alpha=0.25, label=f"±1σ = ±{new_std:.4f}")

    # Old canonical value
    old_gamma = old_baseline_data.get("lambda_0.0", {}).get("gamma", None)
    if old_gamma is not None:
        ax.axhline(old_gamma, color="#D32F2F", lw=1.5, ls="--",
                   label=f"Old canonical Γ = {old_gamma:.4f}")

    # Zero line
    ax.axhline(0, color="grey", lw=0.7, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel(r"$\Gamma = \mathrm{ES}_{0.95}^{\mathrm{BS}} - \mathrm{ES}_{0.95}^{\mathrm{DH}}$",
                  fontsize=11)
    ax.set_title("Advantage gap Γ across 5 seeds (canonical v2, λ=0)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def plot_decomposition_5seeds(
    decomp_json: dict[str, Any],
    old_decomp_data: dict[str, Any],
    path: Path,
) -> None:
    """Grouped bar chart of 5-bucket decomposition per seed with old reference lines."""
    seeds = DECOMP_SEEDS
    buckets = ["objective", "interaction", "stoch_vol", "roughness", "architecture"]
    bucket_labels = ["Objective", "Interaction", "Stoch vol", "Roughness", "Architecture"]
    bucket_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    # Matrix: (n_seeds, n_buckets)
    matrix = np.array([
        [decomp_json["per_seed"][str(s)]["decomposition"]["percentages_of_total"][b]
         for b in buckets]
        for s in seeds
    ])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.16
    x = np.arange(len(seeds))

    old_pct = old_decomp_data.get("percentages", {})

    for i, (b, label, color) in enumerate(zip(buckets, bucket_labels, bucket_colors)):
        offset = (i - 2) * width
        ax.bar(x + offset, matrix[:, i], width, color=color, alpha=0.85,
               edgecolor="k", lw=0.4, label=label)

    # Old reference lines (one per bucket)
    for b, color, label in zip(buckets, bucket_colors, bucket_labels):
        if b in old_pct:
            ax.axhline(old_pct[b], color=color, ls="--", lw=1.0, alpha=0.6)
            ax.text(len(seeds) - 0.3, old_pct[b] + 0.5,
                    f"old {label}={old_pct[b]:+.1f}%",
                    color=color, fontsize=8, alpha=0.9)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel("% of Γ_total", fontsize=11)
    ax.set_title("5-bucket decomposition per seed vs old canonical (dashed)", fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def plot_es_distribution_comparison(
    baseline_json: dict[str, Any],
    path: Path,
    old_baseline_data: dict[str, Any] | None = None,
) -> None:
    """Per-seed ES comparison for DH vs BS vs Plug-in, plus ES99 and old canonical overlay."""
    seeds = BASELINE_SEEDS
    es_bs = [baseline_json["per_seed"][str(s)]["0.0"]["es95_bs"] for s in seeds]
    es_dh = [baseline_json["per_seed"][str(s)]["0.0"]["es95_dh"] for s in seeds]
    es_plugin = [baseline_json["per_seed"][str(s)]["0.0"]["es95_plugin"] for s in seeds]
    es99_bs = [baseline_json["per_seed"][str(s)]["0.0"]["es99_bs"] for s in seeds]
    es99_dh = [baseline_json["per_seed"][str(s)]["0.0"]["es99_dh"] for s in seeds]
    mean_pl_bs = [baseline_json["per_seed"][str(s)]["0.0"]["mean_pl_bs"] for s in seeds]
    mean_pl_dh = [baseline_json["per_seed"][str(s)]["0.0"]["mean_pl_dh"] for s in seeds]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: ES_0.95 comparison (BS vs Plug-in vs DH)
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    width = 0.26
    ax.bar(x - width, es_bs, width, color="#2196F3", label="BS Delta", alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x, es_plugin, width, color="#FF9800", label="Plug-in Delta", alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x + width, es_dh, width, color="#4CAF50", label="Deep Hedger", alpha=0.85, edgecolor="k", lw=0.4)
    if old_baseline_data is not None:
        ob = old_baseline_data.get("lambda_0.0", {})
        if "es95_bs" in ob:
            ax.axhline(ob["es95_bs"], color="#1565C0", ls="--", lw=1.0, alpha=0.8,
                       label=f"old BS = {ob['es95_bs']:.3f}")
        if "es95_dh" in ob:
            ax.axhline(ob["es95_dh"], color="#2E7D32", ls="--", lw=1.0, alpha=0.8,
                       label=f"old DH = {ob['es95_dh']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(r"$\mathrm{ES}_{0.95}$")
    ax.set_title(r"$\mathrm{ES}_{0.95}$ per seed (λ=0, canonical v2)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: ES_0.99 comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, es99_bs, width, color="#2196F3", label="BS Delta", alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x + width/2, es99_dh, width, color="#4CAF50", label="Deep Hedger", alpha=0.85, edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(r"$\mathrm{ES}_{0.99}$")
    ax.set_title(r"$\mathrm{ES}_{0.99}$ per seed (λ=0, canonical v2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Gamma per seed with mean±std band, old canonical line
    ax = axes[1, 0]
    gammas = [baseline_json["per_seed"][str(s)]["0.0"]["gamma"] for s in seeds]
    new_mean = baseline_json["aggregated"]["0.0"]["gamma"]["mean"]
    new_std = baseline_json["aggregated"]["0.0"]["gamma"]["std"]
    bars = ax.bar(x, gammas, color="#4CAF50", alpha=0.8, edgecolor="k", lw=0.4,
                  label=f"Per-seed Γ")
    for xi, gi in zip(x, gammas):
        ax.text(xi, gi + 0.01, f"{gi:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(new_mean, color="#2E7D32", lw=1.5, ls="-",
               label=f"mean = {new_mean:.4f}")
    ax.axhspan(new_mean - new_std, new_mean + new_std,
               color="#81C784", alpha=0.25, label=f"±1σ = {new_std:.4f}")
    ax.axhspan(new_mean - 2 * new_std, new_mean + 2 * new_std,
               color="#C8E6C9", alpha=0.18, label=f"±2σ")
    if old_baseline_data is not None:
        og = old_baseline_data.get("lambda_0.0", {}).get("gamma", None)
        if og is not None:
            ax.axhline(og, color="#D32F2F", lw=1.5, ls="--",
                       label=f"old Γ = {og:.4f}")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title(r"$\Gamma = \mathrm{ES}_{0.95}^{\mathrm{BS}} - \mathrm{ES}_{0.95}^{\mathrm{DH}}$ (λ=0)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Mean PL per seed
    ax = axes[1, 1]
    ax.bar(x - width/2, mean_pl_bs, width, color="#2196F3", label="BS Delta", alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x + width/2, mean_pl_dh, width, color="#4CAF50", label="Deep Hedger", alpha=0.85, edgecolor="k", lw=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean P&L")
    ax.set_title("Mean terminal P&L per seed (λ=0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Canonical v2 — Section 6.3 baseline: per-seed breakdown",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=str, default=str(BASELINE_JSON))
    parser.add_argument("--decomp-json", type=str, default=str(DECOMP_JSON))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {args.baseline_json}", flush=True)
    with open(args.baseline_json) as f:
        baseline_json = json.load(f)

    print(f"  Loading {args.decomp_json}", flush=True)
    with open(args.decomp_json) as f:
        decomp_json = json.load(f)

    # Old canonical
    old_baseline_data: dict[str, Any] = baseline_json.get("old_canonical", {})
    old_decomp_data: dict[str, Any] = decomp_json.get("old_canonical", {})

    # If old files available directly, also reload for richer data
    if not old_baseline_data and OLD_BASELINE_JSON.exists():
        with open(OLD_BASELINE_JSON) as f:
            old_raw = json.load(f)
        old_baseline_data = {
            "lambda_0.0": {
                "es95_bs": old_raw["0.0"]["BS Delta"]["es_95"],
                "es95_dh": old_raw["0.0"]["Deep Hedger"]["es_95"],
                "gamma": old_raw["0.0"]["BS Delta"]["es_95"] - old_raw["0.0"]["Deep Hedger"]["es_95"],
                "mean_pl_dh": old_raw["0.0"]["Deep Hedger"]["mean_pnl"],
                "std_pl_dh": old_raw["0.0"]["Deep Hedger"]["std_pnl"],
            },
            "lambda_0.001": {
                "es95_bs": old_raw["0.001"]["BS Delta"]["es_95"],
                "es95_dh": old_raw["0.001"]["Deep Hedger"]["es_95"],
                "gamma": old_raw["0.001"]["BS Delta"]["es_95"] - old_raw["0.001"]["Deep Hedger"]["es_95"],
            },
        }
    if not old_decomp_data and OLD_DECOMP_JSON.exists():
        with open(OLD_DECOMP_JSON) as f:
            old_raw = json.load(f)
        old_decomp_data = {
            "Gamma_total": old_raw["decomposition"]["Gamma_total"],
            "percentages": old_raw["decomposition"]["percentages_of_total"],
        }

    # Build report
    report = build_report(baseline_json, decomp_json, old_baseline_data, old_decomp_data)
    REPORT_PATH.write_text(report)
    print(f"  Wrote {REPORT_PATH}", flush=True)

    # Build figures
    plot_gamma_5seeds(baseline_json, old_baseline_data,
                      FIGURES_DIR / "gamma_5seeds.png")
    plot_decomposition_5seeds(decomp_json, old_decomp_data,
                              FIGURES_DIR / "decomposition_5seeds.png")
    plot_es_distribution_comparison(baseline_json,
                                    FIGURES_DIR / "es_distribution_comparison.png",
                                    old_baseline_data)

    print("\n  ========== Report preview ==========\n", flush=True)
    print(report[:3000], flush=True)


if __name__ == "__main__":
    main()
