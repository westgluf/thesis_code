#!/usr/bin/env python
"""
Build the closed 5-component decomposition of the deep hedging advantage.

Reads figures/diagnostic_controls_results.json (which must contain keys
A, A_prime, C, D) and emits figures/decomposition_closed.json.

No training — this script is pure arithmetic.

Run:
    python -u -m deep_hedging.experiments.build_decomposition
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
INPUT_JSON  = FIGURE_DIR / "diagnostic_controls_results.json"
OUTPUT_JSON = FIGURE_DIR / "decomposition_closed.json"


def _git_commit_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 2x2 Factorial
# ---------------------------------------------------------------------------

def compute_2x2_factorial(data: dict) -> dict:
    """Extract the four corners of the (eta, objective) 2x2 design.

    Returns dict with gamma values and the ES95 inputs.
    """
    A      = data["A"]          # eta=0, ES objective
    A_prime = data["A_prime"]   # eta=0, MSE objective
    C      = data["C"]          # eta=1.9, all three objectives

    # ES95 values for each cell
    bs_eta0    = A["es95_bs"]
    bs_eta1p9  = C["bs"]["es95"]
    dh_es_eta0    = A["es95_dh"]         # DH-ES at eta=0
    dh_mse_eta0   = A_prime["es95_dh"]   # DH-MSE at eta=0
    dh_es_eta1p9  = C["dh_es"]["es95"]   # DH-ES at eta=1.9
    dh_mse_eta1p9 = C["dh_mse"]["es95"]  # DH-MSE at eta=1.9

    # Gamma = ES95_BS - ES95_DH (positive means DH is better)
    gamma_00 = bs_eta0   - dh_mse_eta0     # eta=0, MSE
    gamma_01 = bs_eta0   - dh_es_eta0      # eta=0, ES  (== A["gamma"])
    gamma_10 = bs_eta1p9 - dh_mse_eta1p9   # eta=1.9, MSE
    gamma_11 = bs_eta1p9 - dh_es_eta1p9    # eta=1.9, ES  (== C total)

    return {
        "gamma_00_eta0_mse":   gamma_00,
        "gamma_01_eta0_es":    gamma_01,
        "gamma_10_eta1p9_mse": gamma_10,
        "gamma_11_eta1p9_es":  gamma_11,
        "es95_values": {
            "bs_eta0":        bs_eta0,
            "bs_eta1p9":      bs_eta1p9,
            "dh_mse_eta0":    dh_mse_eta0,
            "dh_mse_eta1p9":  dh_mse_eta1p9,
            "dh_es_eta0":     dh_es_eta0,
            "dh_es_eta1p9":   dh_es_eta1p9,
        },
    }


def decompose_2x2(fact: dict) -> dict:
    """2x2 factorial ANOVA decomposition.

    Returns architecture, objective, dynamics, interaction components.
    """
    g00 = fact["gamma_00_eta0_mse"]
    g01 = fact["gamma_01_eta0_es"]
    g10 = fact["gamma_10_eta1p9_mse"]
    g11 = fact["gamma_11_eta1p9_es"]

    architecture     = g00
    objective        = g10 - g00
    dynamics         = g01 - g00
    interaction_2x2  = g11 - g01 - g10 + g00

    # Verify closure
    total = g11
    recon = architecture + objective + dynamics + interaction_2x2
    assert abs(recon - total) < 1e-9, (
        f"2x2 closure failed: {recon} != {total}, diff={recon - total}"
    )

    return {
        "Gamma_total":           total,
        "Gamma_architecture":    architecture,
        "Gamma_objective":       objective,
        "Gamma_dynamics":        dynamics,
        "Gamma_interaction_2x2": interaction_2x2,
    }


# ---------------------------------------------------------------------------
# 3x3 ANOVA on Experiment D grid
# ---------------------------------------------------------------------------

def compute_3x3_anova(D_list: list[dict]) -> dict:
    """Two-way ANOVA on the 9-cell (H, eta) grid from Experiment D.

    Returns SS decomposition and variance fractions.
    """
    H_values   = sorted(set(r["H"]   for r in D_list))
    eta_values = sorted(set(r["eta"] for r in D_list))
    n_rows = len(H_values)
    n_cols = len(eta_values)

    # Build gamma matrix
    lookup = {(r["H"], r["eta"]): r["gamma"] for r in D_list}
    matrix = np.array([
        [lookup[(h, e)] for e in eta_values]
        for h in H_values
    ])

    grand_mean = matrix.mean()
    row_means  = matrix.mean(axis=1)   # mean over eta for each H
    col_means  = matrix.mean(axis=0)   # mean over H for each eta

    SS_H   = n_cols * float(np.sum((row_means - grand_mean) ** 2))
    SS_eta = n_rows * float(np.sum((col_means - grand_mean) ** 2))
    SS_tot = float(np.sum((matrix - grand_mean) ** 2))
    SS_int = SS_tot - SS_H - SS_eta

    # Avoid division by zero if SS_tot is tiny
    if SS_tot < 1e-15:
        f_H = f_eta = f_int = 1.0 / 3.0
    else:
        f_H   = SS_H   / SS_tot
        f_eta = SS_eta / SS_tot
        f_int = SS_int / SS_tot

    return {
        "H_values":        H_values,
        "eta_values":      eta_values,
        "gamma_matrix":    matrix.tolist(),
        "grand_mean":      float(grand_mean),
        "row_means_H":     row_means.tolist(),
        "col_means_eta":   col_means.tolist(),
        "SS_H":            SS_H,
        "SS_eta":          SS_eta,
        "SS_total":        SS_tot,
        "SS_interaction":  SS_int,
        "f_H":             f_H,
        "f_eta":           f_eta,
        "f_interaction":   f_int,
    }


# ---------------------------------------------------------------------------
# Full 5-component decomposition
# ---------------------------------------------------------------------------

def build_five_components(decomp_2x2: dict, anova: dict) -> dict:
    """Combine 2x2 factorial with 3x3 ANOVA to produce final 5-component split."""
    total        = decomp_2x2["Gamma_total"]
    architecture = decomp_2x2["Gamma_architecture"]
    objective    = decomp_2x2["Gamma_objective"]
    dynamics     = decomp_2x2["Gamma_dynamics"]
    int_2x2      = decomp_2x2["Gamma_interaction_2x2"]

    # Split dynamics into stoch_vol (eta effect) and roughness (H effect)
    f_eta = anova["f_eta"]
    f_H   = anova["f_H"]
    f_int = anova["f_interaction"]

    stoch_vol     = f_eta * dynamics
    roughness     = f_H   * dynamics
    int_dynamics  = f_int * dynamics

    # Total interaction = 2x2 interaction + within-dynamics interaction
    interaction_total = int_2x2 + int_dynamics

    # Verify closure
    recon = architecture + objective + stoch_vol + roughness + interaction_total
    residual = recon - total

    if abs(residual) > 1e-9:
        raise AssertionError(
            f"5-component closure failed: sum={recon}, total={total}, "
            f"residual={residual}\n"
            f"  arch={architecture}, obj={objective}, sv={stoch_vol}, "
            f"rough={roughness}, int={interaction_total}"
        )

    # Percentages
    pcts = {}
    if abs(total) > 1e-12:
        pcts = {
            "architecture":  100.0 * architecture / total,
            "objective":     100.0 * objective / total,
            "stoch_vol":     100.0 * stoch_vol / total,
            "roughness":     100.0 * roughness / total,
            "interaction":   100.0 * interaction_total / total,
        }
    else:
        pcts = {k: 0.0 for k in
                ["architecture", "objective", "stoch_vol", "roughness", "interaction"]}

    return {
        "Gamma_total":                total,
        "Gamma_architecture":         architecture,
        "Gamma_objective":            objective,
        "Gamma_dynamics":             dynamics,
        "Gamma_stoch_vol":            stoch_vol,
        "Gamma_roughness":            roughness,
        "Gamma_interaction_2x2":      int_2x2,
        "Gamma_interaction_dynamics": int_dynamics,
        "Gamma_interaction_total":    interaction_total,
        "percentages_of_total":       pcts,
        "closure_residual":           residual,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_decomposition(decomp: dict, save_path: Path) -> None:
    """Horizontal bar chart of the five components."""
    labels = ["Objective", "Stoch. vol.", "Architecture", "Roughness", "Interaction"]
    keys   = ["Gamma_objective", "Gamma_stoch_vol", "Gamma_architecture",
              "Gamma_roughness", "Gamma_interaction_total"]
    values = [decomp[k] for k in keys]
    pcts   = decomp["percentages_of_total"]
    pct_keys = ["objective", "stoch_vol", "architecture", "roughness", "interaction"]
    pct_vals = [pcts[k] for k in pct_keys]

    # Sort by absolute value (widest first)
    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    pct_vals = [pct_vals[i] for i in order]

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#795548"]
    colors = [colors[i] for i in order]

    total = decomp["Gamma_total"]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="white")

    # Annotate each bar
    for i, (bar, val, pct) in enumerate(zip(bars, values, pct_vals)):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f" {val:.3f}  ({pct:.1f}%)",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    # Dashed anchor line at Gamma_total
    ax.axvline(total, color="black", ls="--", lw=1.5, zorder=0)
    ax.text(total + 0.01, len(labels) - 0.3,
            f"$\\Gamma_{{total}}$ = {total:.3f}",
            fontsize=10, va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Contribution to $\\Gamma$", fontsize=11)
    ax.set_title(
        "Decomposition of the deep hedging advantage\n"
        r"(baseline $\eta$=1.9, H=0.07; decomposition test set)",
        fontsize=12,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def build_from_paths(input_path: Path, output_path: Path) -> dict:
    """Core logic: read input JSON, compute decomposition, write output JSON.

    Returns the full output dict.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Validate required keys
    if "A_prime" not in data:
        raise KeyError(
            "A_prime not found in input JSON. "
            "Run: python -m deep_hedging.experiments.diagnostic_controls --only A_prime"
        )

    # Step 1: 2x2 factorial
    factorial = compute_2x2_factorial(data)
    decomp_2x2 = decompose_2x2(factorial)

    # Step 2: 3x3 ANOVA
    anova = compute_3x3_anova(data["D"])

    # Step 3: Five-component split
    decomp = build_five_components(decomp_2x2, anova)

    output = {
        "meta": {
            "source_script": "deep_hedging/experiments/build_decomposition.py",
            "inputs": str(input_path),
            "baseline_definition": (
                "es95_BS(eta=1.9, H=0.07) - es95_DH_ES(eta=1.9, H=0.07) "
                "on the diagnostic-controls test set (n_test=30000, seed=2024)"
            ),
            "baseline_disclaimer": (
                "This Gamma_total differs from the unified-baseline Gamma "
                "(unified_baseline_results.json) because the decomposition uses "
                "an independent diagnostic-controls test set. See Section 6.1 "
                "footnote in main.tex."
            ),
            "source_commit": _git_commit_sha(),
        },
        "factorial_2x2": factorial,
        "grid_3x3_anova": anova,
        "decomposition": decomp,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    print("=" * 60)
    print("  Build Closed Decomposition")
    print("=" * 60)

    output = build_from_paths(INPUT_JSON, OUTPUT_JSON)
    decomp = output["decomposition"]

    print(f"\n  Gamma_total            = {decomp['Gamma_total']:.3f}")
    print(f"  Gamma_architecture     = {decomp['Gamma_architecture']:.3f}"
          f"  ({decomp['percentages_of_total']['architecture']:.1f}%)")
    print(f"  Gamma_objective        = {decomp['Gamma_objective']:.3f}"
          f"  ({decomp['percentages_of_total']['objective']:.1f}%)")
    print(f"  Gamma_stoch_vol        = {decomp['Gamma_stoch_vol']:.3f}"
          f"  ({decomp['percentages_of_total']['stoch_vol']:.1f}%)")
    print(f"  Gamma_roughness        = {decomp['Gamma_roughness']:.3f}"
          f"  ({decomp['percentages_of_total']['roughness']:.1f}%)")
    print(f"  Gamma_interaction      = {decomp['Gamma_interaction_total']:.3f}"
          f"  ({decomp['percentages_of_total']['interaction']:.1f}%)")
    print(f"  Closure residual       = {decomp['closure_residual']:.2e}"
          f"  (must be < 1e-9)")

    print(f"\n  Results saved to {OUTPUT_JSON}")

    # Generate figure
    print("\n  Generating decomposition figure...")
    plot_decomposition(decomp, FIGURE_DIR / "fig_diagnostic_decomposition_closed.png")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
