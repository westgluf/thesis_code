#!/usr/bin/env python
"""Consolidate reproducibility subprocess outputs into *_5seeds.json files.

Reads:
  - results/canonical_v2/repro_run1.json, repro_run2.json  (baseline repro, seed 2024)
  - results/canonical_v2/decomp_repro_run1.json, decomp_repro_run2.json (decomp repro, seed 3024)

Updates:
  - results/canonical_v2/baseline_5seeds.json (reproducibility_check field)
  - results/canonical_v2/decomposition_5seeds.json (reproducibility_check field)

Run:
    python -u -m deep_hedging.experiments.consolidate_repro
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "canonical_v2"

def _baseline_repro() -> dict[str, Any] | None:
    r1_path = RESULTS_DIR / "repro_run1.json"
    r2_path = RESULTS_DIR / "repro_run2.json"
    if not (r1_path.exists() and r2_path.exists()):
        print(f"  Baseline repro incomplete: {r1_path.exists()=} {r2_path.exists()=}")
        return None
    with open(r1_path) as f:
        r1 = json.load(f)
    with open(r2_path) as f:
        r2 = json.load(f)

    g1 = r1["0.0"]["gamma"]
    g2 = r2["0.0"]["gamma"]
    es1 = r1["0.0"]["es95_dh"]
    es2 = r2["0.0"]["es95_dh"]
    fw1 = r1["0.0"]["first_weight_sum"]
    fw2 = r2["0.0"]["first_weight_sum"]

    gamma_match = abs(g1 - g2) < 1e-9
    es_match = abs(es1 - es2) < 1e-9
    fw_match = abs(fw1 - fw2) < 1e-9
    all_match = gamma_match and es_match and fw_match

    return {
        "seed": 2024,
        "run1": {"gamma": g1, "es95_dh": es1, "first_weight_sum": fw1},
        "run2": {"gamma": g2, "es95_dh": es2, "first_weight_sum": fw2},
        "gamma_match": gamma_match,
        "es95_dh_match": es_match,
        "first_weight_sum_match": fw_match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }

def _decomp_repro() -> dict[str, Any] | None:
    r1_path = RESULTS_DIR / "decomp_repro_run1.json"
    r2_path = RESULTS_DIR / "decomp_repro_run2.json"
    if not (r1_path.exists() and r2_path.exists()):
        print(f"  Decomp repro incomplete: {r1_path.exists()=} {r2_path.exists()=}")
        return None
    with open(r1_path) as f:
        r1 = json.load(f)
    with open(r2_path) as f:
        r2 = json.load(f)

    g1 = r1["decomposition"]["Gamma_total"]
    g2 = r2["decomposition"]["Gamma_total"]
    obj1 = r1["decomposition"]["percentages_of_total"]["objective"]
    obj2 = r2["decomposition"]["percentages_of_total"]["objective"]
    esA_1 = r1["raw_experiments"]["A"]["es95_dh"]
    esA_2 = r2["raw_experiments"]["A"]["es95_dh"]

    gamma_match = abs(g1 - g2) < 1e-9
    pct_match = abs(obj1 - obj2) < 1e-9
    esA_match = abs(esA_1 - esA_2) < 1e-9
    all_match = gamma_match and pct_match and esA_match

    return {
        "seed": 3024,
        "run1": {"gamma_total": g1, "objective_pct": obj1, "esA_dh": esA_1},
        "run2": {"gamma_total": g2, "objective_pct": obj2, "esA_dh": esA_2},
        "gamma_match": gamma_match,
        "objective_pct_match": pct_match,
        "experiment_A_match": esA_match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }

def main() -> None:
    baseline_path = RESULTS_DIR / "baseline_5seeds.json"
    decomp_path = RESULTS_DIR / "decomposition_5seeds.json"

    # Baseline
    print("\n=== Baseline reproducibility check ===")
    br = _baseline_repro()
    if br is not None:
        print(f"  seed={br['seed']}")
        print(f"  Γ:                run1={br['run1']['gamma']:.6f} run2={br['run2']['gamma']:.6f} match={br['gamma_match']}")
        print(f"  ES_0.95_DH:       run1={br['run1']['es95_dh']:.6f} run2={br['run2']['es95_dh']:.6f} match={br['es95_dh_match']}")
        print(f"  first_weight_sum: run1={br['run1']['first_weight_sum']:.6f} run2={br['run2']['first_weight_sum']:.6f} match={br['first_weight_sum_match']}")
        print(f"  VERDICT: {br['verdict']}")

        if baseline_path.exists():
            with open(baseline_path) as f:
                data = json.load(f)
            data["reproducibility_check"] = br
            with open(baseline_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {baseline_path}")

    # Decomp
    print("\n=== Decomposition reproducibility check ===")
    dr = _decomp_repro()
    if dr is not None:
        print(f"  seed={dr['seed']}")
        print(f"  Γ_total:      run1={dr['run1']['gamma_total']:.6f} run2={dr['run2']['gamma_total']:.6f} match={dr['gamma_match']}")
        print(f"  Objective %:  run1={dr['run1']['objective_pct']:.4f} run2={dr['run2']['objective_pct']:.4f} match={dr['objective_pct_match']}")
        print(f"  ES_A_dh:      run1={dr['run1']['esA_dh']:.6f} run2={dr['run2']['esA_dh']:.6f} match={dr['experiment_A_match']}")
        print(f"  VERDICT: {dr['verdict']}")

        if decomp_path.exists():
            with open(decomp_path) as f:
                data = json.load(f)
            data["reproducibility_check"] = dr
            with open(decomp_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {decomp_path}")

if __name__ == "__main__":
    main()
