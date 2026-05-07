#!/usr/bin/env python
"""6-panel synthesis figure + comprehensive report for Prompt M.

Reads:
  results/perturbation_v2/M{1..6}_*.json

Writes:
  figures/perturbation_v2/perturbation_comprehensive_summary.png
  results/perturbation_v2/perturbation_comprehensive_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "perturbation_v2"
FIGURES = REPO_ROOT / "figures" / "perturbation_v2"
FIGURES.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.load(open(p))


M1 = _load("M1_extended_radius.json")
M2 = _load("M2_axis_sweep.json")
M3 = _load("M3_joint_attacks.json")
M4 = _load("M4_targeted_attacks.json")
M5 = _load("M5_objective_robustness.json")
M6 = _load("M6_hessian.json")

BS_5SEED = 11.5921
DH_5SEED = 10.4442

# --------------------------------------------------------------------------
# Build figure
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08,
                     wspace=0.25, hspace=0.30)

# -- Panel A (M.1): Extended radius — worst-case DH/BS per radius --
ax = axes[0, 0]
if M1 and "results" in M1:
    radii_set = sorted({float(r) for ax_d in M1["results"].values()
                         for dr_d in ax_d.values() for r in dr_d.keys()})
    worst_dh, worst_bs, worst_dh_se, worst_bs_se = [], [], [], []
    for r in radii_set:
        rkey = f"{r:g}"
        dhs, bss, dh_ses, bs_ses = [], [], [], []
        for axis in ("H", "eta", "rho"):
            for direction in ("+", "-"):
                cell = M1["results"].get(axis, {}).get(direction, {}).get(rkey, {})
                ag = cell.get("aggregate", {})
                if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
                    dhs.append(ag["dh_es95"]["mean"])
                    bss.append(ag["bs_es95"]["mean"])
                    dh_ses.append(ag["dh_es95"]["se"])
                    bs_ses.append(ag["bs_es95"]["se"])
        if dhs:
            i = int(np.argmax(np.array(dhs)))  # worst = highest ES
            worst_dh.append(dhs[i]); worst_bs.append(bss[i])
            worst_dh_se.append(dh_ses[i]); worst_bs_se.append(bs_ses[i])
    radii_arr = np.array(radii_set)
    ax.errorbar(radii_arr, worst_dh, yerr=worst_dh_se,
                marker="o", color="#5B8FF9", lw=1.8, capsize=4, label="DH worst-case")
    ax.errorbar(radii_arr, worst_bs, yerr=worst_bs_se,
                marker="s", color="red", lw=1.6, ls="--", capsize=4, label="BS worst-case")
    ax.axvline(2.0, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax.text(2.0, ax.get_ylim()[1]*0.95, " r=2.0\n (current limit)",
            ha="left", va="top", fontsize=8, alpha=0.7)
    cross = M1.get("crossover_analysis", {})
    if cross.get("r_star") is not None:
        info = cross["axis_direction_radius_dh_bs"]
        ax.text(0.05, 0.05, f"Crossover at r*={cross['r_star']}\n"
                            f"axis={info[0]}, dir={info[1]}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax.set_xlabel("perturbation radius $r$")
ax.set_ylabel("worst-case ES$_{0.95}$")
ax.set_title("(A) M.1: Extended radius range\n(7 radii × 6 axis-dir × 5 seeds)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# -- Panel B (M.2): 15-point axis sweeps --
ax = axes[0, 1]
if M2 and "results" in M2:
    palettes = {"H": "#5B8FF9", "eta": "#2EBC8C", "rho": "#9C5CFF"}
    for axis, color in palettes.items():
        cells = M2["results"].get(axis, {})
        if not cells:
            continue
        # x-values normalised: distance from baseline / sigma
        baselines = {"H": 0.07, "eta": 1.9, "rho": -0.7}
        sigmas = {"H": 0.05, "eta": 0.5, "rho": 0.2}
        xs, dh_means, bs_means = [], [], []
        for vk in sorted(cells.keys(), key=float):
            cell = cells[vk]
            v = float(vk)
            ag = cell.get("aggregate", {})
            if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
                xs.append((v - baselines[axis]) / sigmas[axis])
                dh_means.append(ag["dh_es95"]["mean"])
                bs_means.append(ag["bs_es95"]["mean"])
        xs = np.array(xs); dh_means = np.array(dh_means); bs_means = np.array(bs_means)
        ax.plot(xs, dh_means, "-o", color=color, lw=1.4, markersize=3,
                label=f"{axis} DH")
        ax.plot(xs, bs_means, "--", color=color, lw=1.0, alpha=0.6,
                label=f"{axis} BS")
ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)
ax.set_xlabel("normalised distance from baseline ($\\Delta / \\sigma$)")
ax.set_ylabel("ES$_{0.95}$")
ax.set_title("(B) M.2: Axis sweeps\n(3 axes × 15 grid × 5 seeds)")
ax.legend(loc="upper left", fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# -- Panel C (M.3): Joint vs marginal --
ax = axes[0, 2]
if M3 and "results" in M3 and M1 and "results" in M1:
    # Joint
    j_radii, j_dh, j_dh_se = [], [], []
    for rkey in sorted(M3["results"].keys(), key=float):
        cell = M3["results"][rkey]
        ag = cell.get("aggregate", {})
        if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
            j_radii.append(float(rkey))
            j_dh.append(ag["dh_es95"]["mean"])
            j_dh_se.append(ag["dh_es95"]["se"])
    # Marginal worst (from M.1)
    m_radii, m_dh = [], []
    for r in j_radii:
        rkey = f"{r:g}"
        worst = 0.0
        for axis in ("H", "eta", "rho"):
            for direction in ("+", "-"):
                cell = M1["results"].get(axis, {}).get(direction, {}).get(rkey, {})
                ag = cell.get("aggregate", {})
                if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
                    worst = max(worst, ag["dh_es95"]["mean"])
        m_radii.append(r); m_dh.append(worst)
    if j_radii:
        ax.errorbar(j_radii, j_dh, yerr=j_dh_se,
                    marker="o", color="#FF7A5C", lw=1.8, capsize=4,
                    label="joint 3D PGD")
        ax.plot(m_radii, m_dh, "--s", color="#5B8FF9", lw=1.4,
                markersize=4, label="marginal (M.1)")
ax.set_xlabel("perturbation radius $r$")
ax.set_ylabel("worst-case DH ES$_{0.95}$")
ax.set_title("(C) M.3: Joint 3D vs marginal axis-aligned\n(5 radii × 5 seeds)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# -- Panel D (M.4): Targeted attacks --
ax = axes[1, 0]
if M4 and "results" in M4:
    radii_set = sorted({float(r) for m in M4["results"].values() for r in m.keys()})
    targeted_gap, favorable_gap = [], []
    for r in radii_set:
        rkey = f"{r:g}"
        for mode_key, lst in (("dh_targeted", targeted_gap),
                                ("dh_favorable", favorable_gap)):
            cell = M4["results"].get(mode_key, {}).get(rkey, {})
            ag = cell.get("aggregate", {})
            lst.append(ag.get("gap", {}).get("mean", np.nan)
                       if "gap" in ag else np.nan)
    xs = np.arange(len(radii_set))
    w = 0.36
    ax.bar(xs - w/2, targeted_gap, w, color="#FF7A5C",
           label="DH-targeted (max gap)", edgecolor="black", linewidth=0.8)
    ax.bar(xs + w/2, favorable_gap, w, color="#5B8FF9",
           label="DH-favorable (min gap)", edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([f"r={r:g}" for r in radii_set])
ax.set_ylabel("ES$_{DH}$ − ES$_{BS}$ at attack point")
ax.set_title("(D) M.4: Targeted attacks\n(3 radii × 3 seeds × 2 modes)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# -- Panel E (M.5): Objective robustness --
ax = axes[1, 1]
if M5 and "results" in M5:
    objectives = list(M5["results"].keys())
    radii_set = []
    for obj in objectives:
        agg = M5["results"][obj].get("aggregate_per_radius", {})
        radii_set = sorted(set(radii_set) | set(float(r) for r in agg.keys()))
    radii_set = sorted(radii_set) or [1.0, 2.0, 3.0]
    xs = np.arange(len(objectives))
    w = 0.27
    colors = ["#5B8FF9", "#2EBC8C", "#FF7A5C"]
    for i, r in enumerate(radii_set):
        rkey = f"{r:g}"
        means = []; ses = []
        for obj in objectives:
            ag = M5["results"][obj].get("aggregate_per_radius", {}).get(rkey, {})
            means.append(ag.get("mean", np.nan))
            ses.append(ag.get("se", 0.0))
        offset = (i - (len(radii_set)-1)/2) * w
        ax.bar(xs + offset, means, w, yerr=ses, capsize=3,
               color=colors[i % len(colors)], edgecolor="black",
               linewidth=0.7, label=f"r={r:g}")
    ax.set_xticks(xs); ax.set_xticklabels(objectives, rotation=20, fontsize=8)
ax.set_ylabel("worst-case DH ES$_{0.95}$")
ax.set_title("(E) M.5: Objective dependence\n(5 obj × 5 seeds × 3 radii × 6 axis-dir)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# -- Panel F (M.6): Hessian top-1 eigenvectors --
ax = axes[1, 2]
if M6 and "comparison" in M6:
    cmp = M6["comparison"]
    v_dh = np.array(cmp["top1_eigenvector_DH"])
    v_bs = np.array(cmp["top1_eigenvector_BS"])
    cos = cmp["top1_eigenvector_cosine_DH_BS"]
    ratio = cmp["top1_eigenvalue_ratio_DH_over_BS"]
    axis_labels = ["H", "η", "ρ"]
    xs = np.arange(3)
    w = 0.36
    ax.bar(xs - w/2, v_dh, w, color="#5B8FF9", label="DH top-1 eigvec",
           edgecolor="black", linewidth=0.8)
    ax.bar(xs + w/2, v_bs, w, color="#FF7A5C", label="BS top-1 eigvec",
           edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(axis_labels)
    ax.set_ylabel("eigenvector component")
    ax.set_title(f"(F) M.6: Hessian top-1 eigenvector\ncos(DH, BS) = {cos:.3f}, "
                 f"λ_DH/λ_BS = {ratio:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
else:
    ax.set_title("(F) M.6: Hessian eigenstructure (pending)")

fig.suptitle("Prompt M — Perturbation Robustness Comprehensive Extension (6-panel synthesis)",
             fontsize=13, y=0.995)

out_png = FIGURES / "perturbation_comprehensive_summary.png"
fig.savefig(out_png, dpi=170, bbox_inches="tight")
print(f"  wrote {out_png}")

# --------------------------------------------------------------------------
# Build report
# --------------------------------------------------------------------------

report_path = RESULTS / "perturbation_comprehensive_report.md"

def _m1_table() -> str:
    if not M1 or "results" not in M1:
        return "M.1 not yet run."
    lines = ["| radius | worst dir | DH ES (mean ± SE) | BS ES (mean ± SE) | gap |",
             "|---|---|---|---|---|"]
    radii_set = sorted({float(r) for ax_d in M1["results"].values()
                         for dr_d in ax_d.values() for r in dr_d.keys()})
    for r in radii_set:
        rkey = f"{r:g}"
        worst_dh, worst_axis, worst_dr = -1, "", ""
        worst_bs, worst_dh_se, worst_bs_se = 0, 0, 0
        for axis in ("H", "eta", "rho"):
            for direction in ("+", "-"):
                cell = M1["results"].get(axis, {}).get(direction, {}).get(rkey, {})
                ag = cell.get("aggregate", {})
                if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
                    if ag["dh_es95"]["mean"] > worst_dh:
                        worst_dh = ag["dh_es95"]["mean"]
                        worst_axis, worst_dr = axis, direction
                        worst_bs = ag["bs_es95"]["mean"]
                        worst_dh_se = ag["dh_es95"]["se"]
                        worst_bs_se = ag["bs_es95"]["se"]
        if worst_dh > 0:
            lines.append(f"| {r:g} | {worst_axis}{worst_dr} | "
                         f"{worst_dh:.4f} ± {worst_dh_se:.4f} | "
                         f"{worst_bs:.4f} ± {worst_bs_se:.4f} | "
                         f"{worst_dh - worst_bs:+.4f} |")
    cross = M1.get("crossover_analysis", {})
    if cross.get("r_star") is not None:
        info = cross["axis_direction_radius_dh_bs"]
        lines.append(f"\n**CROSSOVER at r* = {cross['r_star']}** "
                     f"in ({info[0]}, {info[1]}) direction "
                     f"(DH ES = {info[3]:.4f} > BS ES = {info[4]:.4f}).")
    else:
        lines.append("\n**No crossover up to r=5.0** in any axis-direction "
                     "of *worst absolute DH ES*.")
    return "\n".join(lines)


def _m2_summary() -> str:
    if not M2 or "results" not in M2:
        return "M.2 not yet run."
    lines = []
    for axis in ("H", "eta", "rho"):
        cells = M2["results"].get(axis, {})
        if not cells:
            continue
        lines.append(f"\n### {axis} sweep\n")
        lines.append("| value | DH ES (mean ± SE) | BS ES (mean ± SE) | gap |")
        lines.append("|---|---|---|---|")
        for vk in sorted(cells.keys(), key=float):
            cell = cells[vk]
            ag = cell.get("aggregate", {})
            if "dh_es95" in ag and ag["dh_es95"].get("n", 0) > 0:
                dh = ag["dh_es95"]; bs = ag["bs_es95"]; gp = ag["gap"]
                lines.append(f"| {float(vk):.4f} | {dh['mean']:.4f} ± {dh['se']:.4f} "
                             f"| {bs['mean']:.4f} ± {bs['se']:.4f} | "
                             f"{gp['mean']:+.4f} |")
    return "\n".join(lines)


def _m3_summary() -> str:
    if not M3 or "results" not in M3:
        return "M.3 not yet run."
    lines = ["| radius | joint DH ES (mean ± SE) | joint BS ES | joint gap | marginal DH ES (M.1) |",
             "|---|---|---|---|---|"]
    radii = sorted(M3["results"].keys(), key=float)
    for rkey in radii:
        ag = M3["results"][rkey].get("aggregate", {})
        if "dh_es95" not in ag or ag["dh_es95"].get("n", 0) == 0:
            continue
        dh, bs, gap = ag["dh_es95"], ag["bs_es95"], ag["gap"]
        # marginal worst from M.1
        marginal = 0
        if M1 and "results" in M1:
            for axis in ("H", "eta", "rho"):
                for direction in ("+", "-"):
                    cell = M1["results"].get(axis, {}).get(direction, {}).get(rkey, {})
                    a = cell.get("aggregate", {})
                    if "dh_es95" in a and a["dh_es95"].get("n", 0) > 0:
                        marginal = max(marginal, a["dh_es95"]["mean"])
        lines.append(f"| {rkey} | {dh['mean']:.4f} ± {dh['se']:.4f} | "
                     f"{bs['mean']:.4f} ± {bs['se']:.4f} | "
                     f"{gap['mean']:+.4f} | {marginal:.4f} |")
    return "\n".join(lines)


def _m4_summary() -> str:
    if not M4 or "results" not in M4:
        return "M.4 not yet run."
    lines = []
    for mode in ("dh_targeted", "dh_favorable"):
        cells = M4["results"].get(mode, {})
        if not cells:
            continue
        lines.append(f"\n### Mode = {mode}\n")
        lines.append("| radius | DH ES | BS ES | gap | direction (H, η, ρ) |")
        lines.append("|---|---|---|---|---|")
        for rkey in sorted(cells.keys(), key=float):
            cell = cells[rkey]
            ag = cell.get("aggregate", {})
            if "dh_es95" not in ag or ag["dh_es95"].get("n", 0) == 0:
                continue
            dh, bs, gap = ag["dh_es95"], ag["bs_es95"], ag["gap"]
            # representative direction from first seed
            seeds = list(cell["per_seed"].keys())
            if seeds:
                fin = cell["per_seed"][seeds[0]].get("final", {})
                direction = (f"({fin.get('H',0):.3f}, "
                             f"{fin.get('eta',0):.3f}, "
                             f"{fin.get('rho',0):.3f})")
            else:
                direction = "?"
            lines.append(f"| {rkey} | {dh['mean']:.4f} ± {dh['se']:.4f} | "
                         f"{bs['mean']:.4f} ± {bs['se']:.4f} | "
                         f"{gap['mean']:+.4f} | {direction} |")
    return "\n".join(lines)


def _m5_summary() -> str:
    if not M5 or "results" not in M5:
        return "M.5 not yet run."
    lines = ["| objective | r=1.0 | r=2.0 | r=3.0 |", "|---|---|---|---|"]
    for obj in M5["results"]:
        agg = M5["results"][obj].get("aggregate_per_radius", {})
        cells = []
        for r in ("1", "2", "3"):
            ag = agg.get(r, {})
            if ag.get("n", 0) > 0:
                cells.append(f"{ag['mean']:.4f} ± {ag['se']:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {obj} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def _m6_summary() -> str:
    if not M6 or "comparison" not in M6:
        return "M.6 not yet run."
    cmp = M6["comparison"]
    cos = cmp["top1_eigenvector_cosine_DH_BS"]
    ratio = cmp["top1_eigenvalue_ratio_DH_over_BS"]
    v_dh = cmp["top1_eigenvector_DH"]
    v_bs = cmp["top1_eigenvector_BS"]
    h_factors = sorted(M6["results"]["dh"].keys(), key=float)
    lines = [f"Top-1 eigenvalue ratio (DH/BS) at h=0.01: **{ratio:.4f}**",
             f"Top-1 eigenvector cosine similarity (DH, BS): **{cos:.4f}**",
             "",
             f"DH top-1 eigenvector: ({v_dh[0]:.3f}, {v_dh[1]:.3f}, {v_dh[2]:.3f}) on (H, η, ρ)",
             f"BS top-1 eigenvector: ({v_bs[0]:.3f}, {v_bs[1]:.3f}, {v_bs[2]:.3f}) on (H, η, ρ)",
             "",
             "Step-size stability (eigenvalue magnitudes):"]
    lines.append("| strategy | h=0.005 | h=0.01 | h=0.02 |")
    lines.append("|---|---|---|---|")
    for strat in ("bs", "dh"):
        cells = []
        for h in h_factors:
            evals = M6["results"][strat][h]["eigenvalues"]
            cells.append(f"({evals[0]:.2f}, {evals[1]:.2f}, {evals[2]:.2f})")
        lines.append(f"| {strat} | {cells[0]} | {cells[1] if len(cells)>1 else '—'} "
                     f"| {cells[2] if len(cells)>2 else '—'} |")
    return "\n".join(lines)


report = f"""# Perturbation Robustness Comprehensive Extension — Final Report

Generated by `deep_hedging.experiments.perturbation_synthesis`.

## Headline

PLACEHOLDER (to be filled in once all sub-experiments complete).

## Reference benchmarks (canonical rough Bergomi)

* BS delta: **{BS_5SEED:.4f}** (5-seed canonical, Prompt B)
* Canonical DH: **{DH_5SEED:.4f}** (5-seed canonical, Prompt B)

---

## M.1 — Extended radius range (axis-aligned)

{_m1_table()}

---

## M.2 — Higher-resolution axis sweeps (15-point × 5 seeds)

{_m2_summary()}

---

## M.3 — Joint 3D PGD vs marginal axis-aligned

{_m3_summary()}

---

## M.4 — Targeted attacks on DH (PGD on DH−BS gap)

{_m4_summary()}

---

## M.5 — Objective-dependent robustness

{_m5_summary()}

---

## M.6 — Hessian eigenstructure

{_m6_summary()}

---

## Synthesis

PLACEHOLDER — characterisation of the basin (size, shape, asymmetry,
training-objective dependence). Reference Panel-A through Panel-F of the
6-panel synthesis figure
(`figures/perturbation_v2/perturbation_comprehensive_summary.png`).

## Connection to Prompt L L.4 finding

PLACEHOLDER — does M.4's DH-targeted direction approach the (η→0)
edge that explains the rough-Bergomi-DH → GBM asymmetry?
"""

report_path.write_text(report)
print(f"  wrote {report_path}")

print("\nDONE.")
