#!/usr/bin/env python
"""Consolidate P01.6 and P01.7 Cell A reproducibility subprocess outputs.

Reads:
  - results/block1_v2/p016_seed7401.json, p016_seed7401_rerun.json
  - results/block1_v2/p017_cellA.json, p017_cellA_seed7711_rerun.json

Updates `results/block1_v2/p016_5seeds.json` and `p017_results.json` with
`reproducibility_check` fields, and regenerates the markdown reports.

Run:
    python -u -m deep_hedging.experiments.consolidate_p016_p017_repro
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "block1_v2"


def build_p016_repro() -> dict[str, Any] | None:
    orig_p = OUT_DIR / "p016_seed7401.json"
    rerun_p = OUT_DIR / "p016_seed7401_rerun.json"
    if not (orig_p.exists() and rerun_p.exists()):
        print(f"  P01.6 repro incomplete: orig={orig_p.exists()} rerun={rerun_p.exists()}")
        return None
    with open(orig_p) as f:
        orig = json.load(f)
    with open(rerun_p) as f:
        rerun = json.load(f)
    fields = ("gamma", "es95_bs", "es95_dh", "first_weight_sum")
    match = {f: orig[f] == rerun[f] for f in fields}
    all_match = all(match.values())
    return {
        "seed": 7401,
        "original": {f: orig[f] for f in fields},
        "rerun": {f: rerun[f] for f in fields},
        "match": match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }


def build_p017_repro() -> dict[str, Any] | None:
    cellA_p = OUT_DIR / "p017_cellA.json"
    rerun_p = OUT_DIR / "p017_cellA_seed7711_rerun.json"
    if not (cellA_p.exists() and rerun_p.exists()):
        print(f"  P01.7 repro incomplete: cellA={cellA_p.exists()} rerun={rerun_p.exists()}")
        return None
    with open(cellA_p) as f:
        orig = json.load(f)
    with open(rerun_p) as f:
        rerun = json.load(f)
    orig_7711 = orig["n400_per_seed"]["7711"]
    rerun_7711 = rerun["n400_per_seed"]["7711"]
    fields = ("gamma", "es95_bs", "es95_dh")
    match = {f: orig_7711[f] == rerun_7711[f] for f in fields}
    all_match = all(match.values())
    return {
        "cell": "A",
        "seed": 7711,
        "original": {f: orig_7711[f] for f in fields},
        "rerun": {f: rerun_7711[f] for f in fields},
        "match_es95_bs": match["es95_bs"],
        "match_es95_dh": match["es95_dh"],
        "match_gamma": match["gamma"],
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }


def update_p016_5seeds(repro: dict[str, Any]) -> None:
    p = OUT_DIR / "p016_5seeds.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    data["reproducibility_check"] = repro
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Updated {p}")


def update_p017_results(repro: dict[str, Any]) -> None:
    p = OUT_DIR / "p017_results.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    data["reproducibility_check"] = repro
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Updated {p}")


def main() -> None:
    print("\n=== P01.6 reproducibility check (seed 7401) ===")
    p016_repro = build_p016_repro()
    if p016_repro is not None:
        for f, m in p016_repro["match"].items():
            print(f"  {f:20s}: "
                  f"orig={p016_repro['original'][f]:.6f} "
                  f"rerun={p016_repro['rerun'][f]:.6f} match={m}")
        print(f"  VERDICT: {p016_repro['verdict']}")
        update_p016_5seeds(p016_repro)

    print("\n=== P01.7 Cell A reproducibility check (seed 7711) ===")
    p017_repro = build_p017_repro()
    if p017_repro is not None:
        for f in ("gamma", "es95_bs", "es95_dh"):
            m = (p017_repro["match_gamma"] if f == "gamma"
                 else p017_repro["match_es95_bs"] if f == "es95_bs"
                 else p017_repro["match_es95_dh"])
            print(f"  {f:20s}: "
                  f"orig={p017_repro['original'][f]:.6f} "
                  f"rerun={p017_repro['rerun'][f]:.6f} match={m}")
        print(f"  VERDICT: {p017_repro['verdict']}")
        update_p017_results(p017_repro)


if __name__ == "__main__":
    main()
