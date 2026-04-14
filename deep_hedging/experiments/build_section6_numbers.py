#!/usr/bin/env python
"""
Aggregate every number cited in Sections 6.2 and 6.3 of main.tex
into a single flat JSON, figures/section6_numbers.json, keyed by
human-readable label (matching what the LaTeX will write).

Source JSONs (all must exist on disk):
    figures/unified_baseline_results.json      (Prompt 02)
    figures/decomposition_closed.json           (Prompt 03)
    figures/diagnostic_controls_results.json    (Prompt 03)
    figures/h_sweep_results.json                (Prompt 05)
    figures/h2_grid_extension.json              (Prompt 04)

Run:
    python -u -m deep_hedging.experiments.build_section6_numbers
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
OUTPUT = FIGURE_DIR / "section6_numbers.json"


def _load(name: str) -> dict:
    p = FIGURE_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required source JSON not found: {p}. "
            f"Run the prompt that produces it first."
        )
    return json.loads(p.read_text())


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _count_deep_covered(h2: dict) -> int:
    n = 0
    for row in h2["grid"].values():
        for cell in row.values():
            deep = cell.get("Deep") or {}
            if deep.get("metrics") is not None and not deep.get("unavailable"):
                n += 1
    return n


def _collect_gbm_benchmark_numbers() -> dict:
    """Read Section 6.2 aggregate CSVs if present."""
    agg_dir = (Path(__file__).resolve().parents[2]
               / "results" / "gbm_deephedge" / "benchmark_6_2" / "aggregate")
    scenario_csv = agg_dir / "scenario_summary.csv"
    if not scenario_csv.exists():
        return {
            "source_missing": True,
            "note": "Run `python -m src.run_benchmark_gbm_grid` to populate.",
            "expected_path": str(scenario_csv),
        }
    rows = list(csv.DictReader(scenario_csv.open()))
    return {
        "scenario_summary": rows,
        "source": str(scenario_csv),
    }


def build() -> dict[str, Any]:
    unified = _load("unified_baseline_results.json")
    decomp  = _load("decomposition_closed.json")
    diagct  = _load("diagnostic_controls_results.json")
    hsweep  = _load("h_sweep_results.json")
    h2grid  = _load("h2_grid_extension.json")

    # --- Section 6.2 ---
    section_6_2 = _collect_gbm_benchmark_numbers()

    # --- Section 6.3 ---
    r0   = unified["results"]["0.0"]
    r001 = unified["results"]["0.001"]
    meta_u = unified["meta"]

    # Observation 6.1 (baseline comparison)
    obs61 = {
        "test_set": {
            "n_test":      meta_u["n_test"],
            "master_seed": meta_u["master_test_seed"],
            "source":      "unified_baseline_results.json",
        },
        "lambda_0": {
            "BS_Delta":          r0["BS Delta"]["metrics"],
            "Plug_in_Delta":     r0["Plug-in Delta"]["metrics"],
            "DH_full_budget":    r0["DH full-budget"]["metrics"],
            "DH_GBM_pretrained": r0["DH GBM-pretrained"]["metrics"],
            "turnovers": {
                "BS_Delta":          r0["BS Delta"]["mean_turnover"],
                "Plug_in_Delta":     r0["Plug-in Delta"]["mean_turnover"],
                "DH_full_budget":    r0["DH full-budget"]["mean_turnover"],
                "DH_GBM_pretrained": r0["DH GBM-pretrained"]["mean_turnover"],
            },
        },
        "lambda_0p001": {
            "BS_Delta":          r001["BS Delta"]["metrics"],
            "Plug_in_Delta":     r001["Plug-in Delta"]["metrics"],
            "DH_full_budget":    r001["DH full-budget"]["metrics"],
            "DH_GBM_pretrained": r001["DH GBM-pretrained"]["metrics"],
        },
        "derived": {
            "gamma_baseline": (r0["BS Delta"]["metrics"]["es_95"]
                             - r0["DH full-budget"]["metrics"]["es_95"]),
            "gamma_transfer": (r0["BS Delta"]["metrics"]["es_95"]
                             - r0["DH GBM-pretrained"]["metrics"]["es_95"]),
            "pct_reduction_full_budget":
                100.0 * (r0["BS Delta"]["metrics"]["es_95"]
                       - r0["DH full-budget"]["metrics"]["es_95"])
                      / r0["BS Delta"]["metrics"]["es_95"],
            "pct_reduction_transfer":
                100.0 * (r0["BS Delta"]["metrics"]["es_95"]
                       - r0["DH GBM-pretrained"]["metrics"]["es_95"])
                      / r0["BS Delta"]["metrics"]["es_95"],
            "plugin_turnover_ratio_vs_BS":
                r0["Plug-in Delta"]["mean_turnover"]
                / r0["BS Delta"]["mean_turnover"],
        },
    }

    # Observation 6.2 (H-sweep + bootstrap)
    hsweep_results = (hsweep["results"]
                      if isinstance(hsweep, dict) and "results" in hsweep
                      else hsweep)
    obs62 = {
        "test_set": {
            "note": "Per-H independent rough-Bergomi test sets",
            "source": "h_sweep_results.json",
        },
        "per_H": [
            {"H":       row["H"],
             "es95_bs": row.get("bs_metrics", {}).get("es_95"),
             "es95_dh": row.get("dh_metrics", {}).get("es_95"),
             "gamma":   row.get("gamma")}
            for row in hsweep_results
        ],
        "bootstrap": hsweep.get("bootstrap", {}),
    }

    # Observation 6.3 (closed decomposition)
    obs63 = {
        "test_set": {
            "note": decomp["meta"]["baseline_disclaimer"],
            "source": "decomposition_closed.json",
        },
        "factorial_2x2": decomp["factorial_2x2"],
        "grid_3x3_anova": decomp["grid_3x3_anova"],
        "decomposition": decomp["decomposition"],
        "raw_C_experiment": {
            "es95_bs":     diagct["C"]["bs"]["es95"],
            "es95_dh_mse": diagct["C"]["dh_mse"]["es95"],
            "es95_dh_es":  diagct["C"]["dh_es"]["es95"],
        },
    }

    # Observation 6.5 (h2 grid + Leland)
    h2_cfg = h2grid["config"]
    obs65 = {
        "test_set": {
            "n_test":      h2_cfg["n_test"],
            "master_seed": h2_cfg["master_seed"],
            "source":      "h2_grid_extension.json",
        },
        "grid_config": {
            "freq_values": h2_cfg["freq_values"],
            "cost_values": h2_cfg["cost_values"],
            "strategies":  h2_cfg.get("strategies", ["BS"]),
        },
        "es95_table": {
            str(n): {
                str(c): {
                    s: (h2grid["grid"][str(n)][str(c)].get(s, {})
                        .get("metrics", {}).get("es_95"))
                    for s in h2_cfg.get("strategies", ["BS"])
                    if s in h2grid["grid"][str(n)][str(c)]
                }
                for c in h2_cfg["cost_values"]
            }
            for n in h2_cfg["freq_values"]
        },
        "sigma_leland_table": {
            str(n): {
                str(c): (h2grid["grid"][str(n)][str(c)]
                         .get("Leland", {}).get("sigma_leland"))
                for c in h2_cfg["cost_values"]
            }
            for n in h2_cfg["freq_values"]
        },
        "deep_coverage": {
            "total_cells":   len(h2_cfg["freq_values"]) * len(h2_cfg["cost_values"]),
            "covered_cells": _count_deep_covered(h2grid),
            "missing_note":  "Deep hedger available only where pareto_front.py cached PnL tensors.",
        },
        "detection": h2grid.get("detection", {}),
    }

    # Observation 6.7 (transfer learning)
    obs67 = {
        "source": "unified_baseline_results.json",
        "es95_bs":                r0["BS Delta"]["metrics"]["es_95"],
        "es95_dh_full_budget":    r0["DH full-budget"]["metrics"]["es_95"],
        "es95_dh_gbm_pretrained": r0["DH GBM-pretrained"]["metrics"]["es_95"],
        "gap_bs_minus_dh_gbm":    obs61["derived"]["gamma_transfer"],
        "gap_dh_minus_dh_gbm":    (r0["DH GBM-pretrained"]["metrics"]["es_95"]
                                 - r0["DH full-budget"]["metrics"]["es_95"]),
        "pct_reduction_vs_bs":    obs61["derived"]["pct_reduction_transfer"],
    }

    return {
        "meta": {
            "built_at":      datetime.now(timezone.utc).isoformat(),
            "source_script": "deep_hedging/experiments/build_section6_numbers.py",
            "source_commit": _git_commit(),
            "note":          "Single source of truth for every body-text number in main.tex Sections 6.2 and 6.3.",
            "sources_aggregated": [
                "unified_baseline_results.json",
                "decomposition_closed.json",
                "diagnostic_controls_results.json",
                "h_sweep_results.json",
                "h2_grid_extension.json",
            ],
        },
        "section_6_2_gbm_benchmark":      section_6_2,
        "observation_6_1_baseline":        obs61,
        "observation_6_2_roughness_null":  obs62,
        "observation_6_3_decomposition":   obs63,
        "observation_6_5_frequency_cost":  obs65,
        "observation_6_7_transfer":        obs67,
    }


def main() -> None:
    out = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUTPUT}")

    # Headline numbers
    d = out["observation_6_1_baseline"]["derived"]
    print(f"  Gamma_baseline (DH-full vs BS):  {d['gamma_baseline']:+.3f} "
          f"({d['pct_reduction_full_budget']:+.1f}% reduction)")
    print(f"  Gamma_transfer (DH-GBM vs BS):   {d['gamma_transfer']:+.3f} "
          f"({d['pct_reduction_transfer']:+.1f}% reduction)")
    print(f"  Plug-in turnover ratio vs BS:    {d['plugin_turnover_ratio_vs_BS']:.2f}x")

    b = out["observation_6_2_roughness_null"]["bootstrap"]
    if "panel_slope" in b:
        ps = b["panel_slope"]
        ci = ps["beta_ci_bootstrap_95"]
        print(f"  Beta_hat:                        {ps['beta_hat']:+.4f} "
              f"[95% CI: {ci[0]:+.3f}, {ci[1]:+.3f}]")
    if "noise_floor" in b:
        nf = b["noise_floor"]
        ratio = nf["beta_noise_floor"] / max(abs(ps["beta_hat"]), 1e-12)
        print(f"  Beta_noise_floor:                {nf['beta_noise_floor']:.3f} "
              f"(|beta_hat|/noise = 1:{ratio:.0f})")

    dec = out["observation_6_3_decomposition"]["decomposition"]
    pct = dec["percentages_of_total"]
    print(f"  Decomposition (Gamma_total={dec['Gamma_total']:+.3f}):")
    for k in ("objective", "interaction", "stoch_vol", "roughness", "architecture"):
        v = pct.get(k)
        if v is not None:
            print(f"    {k:14s}  {v:+6.1f}%")


if __name__ == "__main__":
    main()
