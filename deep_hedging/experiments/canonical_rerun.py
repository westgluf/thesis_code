#!/usr/bin/env python
"""
Canonical Section 6.3 re-run across 5 seeds with the seeding fix applied.

Applies the 2-line seeding fix from Phase B to `run_section_6_3_baseline.py`
and re-runs the same experiment across seeds [2024, 2025, 2026, 2027, 2028]
for both the frictionless (lambda=0.0) and with-costs (lambda=0.001) cases.

Produces:
  - results/canonical_v2/baseline_5seeds.json (per-seed + aggregated)
  - results/canonical_v2/baseline_reproducibility.json (dual-subprocess check)

Run:
    python -u -m deep_hedging.experiments.canonical_rerun
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_hedging.experiments.run_section_6_3_baseline import Section63Experiment
from deep_hedging.utils.config import RoughBergomiParams, DatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "canonical_v2"
FIGURES_DIR = REPO_ROOT / "figures" / "canonical_v2"
OUTPUT_JSON = RESULTS_DIR / "baseline_5seeds.json"
REPRO_JSON = RESULTS_DIR / "baseline_reproducibility.json"

SEEDS = [2024, 2025, 2026, 2027, 2028]
COST_LAMBDAS = [0.0, 0.001]
DATASET_KW = dict(n_train=80_000, n_val=20_000, n_test=50_000)
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 2048
LR = 1e-3

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

def _metrics_to_dict(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
    return out

def _first_weight_sum(model: torch.nn.Module) -> float:
    """Sum of the first linear layer's weight tensor (determinism signature)."""
    for p in model.parameters():
        if p.ndim >= 2:
            return float(p.detach().flatten().sum().cpu())
    return 0.0

def run_one_seed(seed: int) -> dict[str, Any]:
    """Run the full Section 6.3 experiment for a single seed."""
    print(f"\n{'='*70}")
    print(f"  SEED {seed}")
    print(f"{'='*70}", flush=True)
    t0 = time.perf_counter()

    params = RoughBergomiParams()
    dataset_config = DatasetConfig(**DATASET_KW)
    exp = Section63Experiment(params=params, dataset_config=dataset_config)

    exp.generate_data(seed=seed)

    per_lam: dict[str, dict[str, Any]] = {}
    first_ws: dict[str, float] = {}
    for lam in COST_LAMBDAS:
        print(f"\n  -- lambda={lam} --", flush=True)
        r_bs = exp.run_bs_delta(lam)
        r_plugin = exp.run_plugin_delta(lam)
        r_dh = exp.run_deep_hedger(
            cost_lambda=lam, epochs=EPOCHS, patience=PATIENCE,
            batch_size=BATCH_SIZE, lr=LR, seed=seed,
        )

        fw = _first_weight_sum(r_dh["model"])
        first_ws[str(lam)] = fw

        bs_m = _metrics_to_dict(r_bs["metrics"])
        dh_m = _metrics_to_dict(r_dh["metrics"])
        plugin_m = _metrics_to_dict(r_plugin["metrics"])
        gamma = bs_m["es_95"] - dh_m["es_95"]

        per_lam[str(lam)] = {
            "es95_bs": bs_m["es_95"],
            "es95_dh": dh_m["es_95"],
            "es95_plugin": plugin_m["es_95"],
            "es99_bs": bs_m["es_99"],
            "es99_dh": dh_m["es_99"],
            "var95_bs": bs_m["var_95"],
            "var95_dh": dh_m["var_95"],
            "var99_bs": bs_m.get("var_99", 0.0),
            "var99_dh": dh_m.get("var_99", 0.0),
            "mean_pl_bs": bs_m["mean_pnl"],
            "mean_pl_dh": dh_m["mean_pnl"],
            "std_pl_bs": bs_m["std_pnl"],
            "std_pl_dh": dh_m["std_pnl"],
            "skew_bs": bs_m["skewness"],
            "skew_dh": dh_m["skewness"],
            "kurtosis_bs": bs_m["kurtosis"],
            "kurtosis_dh": dh_m["kurtosis"],
            "gamma": gamma,
            "first_weight_sum": fw,
            "dh_metrics_full": dh_m,
            "bs_metrics_full": bs_m,
            "plugin_metrics_full": plugin_m,
        }
        del r_bs, r_plugin, r_dh
        gc.collect()

    wall = time.perf_counter() - t0
    per_lam["wall_clock_s"] = wall
    per_lam["first_weight_sums"] = first_ws
    print(f"\n  Seed {seed} done in {wall/60:.1f} min")
    print(f"  lambda=0.0  Gamma = {per_lam['0.0']['gamma']:+.4f}  "
          f"(ES_BS={per_lam['0.0']['es95_bs']:.4f}, ES_DH={per_lam['0.0']['es95_dh']:.4f})")
    print(f"  lambda=0.001 Gamma = {per_lam['0.001']['gamma']:+.4f}")
    return per_lam

def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate across seeds: mean, std, ci_low/high for each key."""
    def agg_scalar(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se = std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
        return {
            "mean": mean,
            "std": std,
            "se": se,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(arr)),
        }

    aggregated: dict[str, Any] = {}
    keys_to_agg = [
        "es95_bs", "es95_dh", "es95_plugin",
        "es99_bs", "es99_dh",
        "var95_bs", "var95_dh",
        "mean_pl_bs", "mean_pl_dh",
        "std_pl_bs", "std_pl_dh",
        "skew_bs", "skew_dh",
        "kurtosis_bs", "kurtosis_dh",
        "gamma",
    ]
    for lam_str in ["0.0", "0.001"]:
        aggregated[lam_str] = {}
        for key in keys_to_agg:
            values = [per_seed[str(s)][lam_str][key] for s in SEEDS]
            aggregated[lam_str][key] = agg_scalar(values)
    return aggregated

def reproducibility_check() -> dict[str, Any]:
    """Re-run seed 2024 in a fresh subprocess and compare outputs."""
    print(f"\n{'='*70}")
    print("  REPRODUCIBILITY CHECK — seed 2024 in two fresh subprocesses")
    print(f"{'='*70}", flush=True)

    check_json1 = RESULTS_DIR / "repro_run1.json"
    check_json2 = RESULTS_DIR / "repro_run2.json"
    check_json1.parent.mkdir(parents=True, exist_ok=True)

    cmd_template = [
        sys.executable, "-u", "-m", "deep_hedging.experiments.canonical_rerun",
        "--single-seed", "2024",
        "--single-seed-output",
    ]

    for i, out_path in enumerate([check_json1, check_json2], 1):
        print(f"\n  Subprocess {i}: writing to {out_path}", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd_template + [str(out_path)],
            cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"  Subprocess {i} FAILED (exit {result.returncode})")
            print(f"  STDERR: {result.stderr[-500:]}")
            return {
                "error": f"subprocess {i} failed",
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:],
            }
        print(f"  Subprocess {i} done in {elapsed/60:.1f} min", flush=True)

    with open(check_json1) as f:
        r1 = json.load(f)
    with open(check_json2) as f:
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

    result: dict[str, Any] = {
        "seed": 2024,
        "run1": {"gamma": g1, "es95_dh": es1, "first_weight_sum": fw1},
        "run2": {"gamma": g2, "es95_dh": es2, "first_weight_sum": fw2},
        "gamma_match": gamma_match,
        "es95_dh_match": es_match,
        "first_weight_sum_match": fw_match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }
    print(f"\n  Gamma      : run1={g1:.6f}  run2={g2:.6f}  match={gamma_match}")
    print(f"  ES_0.95_DH : run1={es1:.6f}  run2={es2:.6f}  match={es_match}")
    print(f"  first-W sum: run1={fw1:.6f}  run2={fw2:.6f}  match={fw_match}")
    print(f"  VERDICT    : {result['verdict']}", flush=True)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-seed", type=int, default=None,
                        help="Run a single seed and exit (for reproducibility check).")
    parser.add_argument("--single-seed-output", type=str, default=None,
                        help="Output path for single-seed JSON.")
    parser.add_argument("--skip-reproducibility", action="store_true",
                        help="Skip the dual-subprocess reproducibility check.")
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None,
                        help="Only run these seeds (subset of canonical).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.single_seed is not None:
        r = run_one_seed(args.single_seed)
        out_path = Path(args.single_seed_output) if args.single_seed_output else (
            RESULTS_DIR / f"baseline_seed_{args.single_seed}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            # strip tensor-like objects
            def _clean(obj):
                if isinstance(obj, dict):
                    return {k: _clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_clean(v) for v in obj]
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                if isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                return None
            json.dump(_clean(r), f, indent=2)
        print(f"\n  Wrote {out_path}")
        return

    seeds_to_run = args.seeds_only if args.seeds_only else SEEDS

    print("=" * 70)
    print("  CANONICAL RE-RUN — Section 6.3 baseline, 5 seeds")
    print(f"  seeds: {seeds_to_run}")
    print(f"  commit: {_git_commit_sha()}")
    print("=" * 70, flush=True)

    per_seed: dict[str, dict[str, Any]] = {}
    total_t0 = time.perf_counter()
    for seed in seeds_to_run:
        r = run_one_seed(seed)
        per_seed[str(seed)] = r
        # Incremental save after each seed so partial results survive crashes.
        _partial = {
            "meta": {
                "script": "canonical_rerun.py",
                "git_commit": _git_commit_sha(),
                "fix_applied": True,
                "seeds_complete": list(per_seed.keys()),
                "seeds_planned": seeds_to_run,
            },
            "per_seed": per_seed,
        }

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            return None

        with open(OUTPUT_JSON, "w") as f:
            json.dump(_clean(_partial), f, indent=2)
        print(f"  (saved partial to {OUTPUT_JSON})", flush=True)

    total_wall = time.perf_counter() - total_t0
    print(f"\n  TOTAL wall-clock (5 seeds): {total_wall/60:.1f} min", flush=True)

    aggregated = aggregate(per_seed)

    repro = None
    if not args.skip_reproducibility:
        repro = reproducibility_check()

    # Old canonical values (from existing figures/section_63_metrics.json)
    old_canonical_path = REPO_ROOT / "figures" / "section_63_metrics.json"
    old_canonical: dict[str, Any] = {}
    if old_canonical_path.exists():
        with open(old_canonical_path) as f:
            old_data = json.load(f)
        old_canonical = {
            "source": str(old_canonical_path),
            "lambda_0.0": {
                "es95_bs": old_data["0.0"]["BS Delta"]["es_95"],
                "es95_dh": old_data["0.0"]["Deep Hedger"]["es_95"],
                "gamma": (
                    old_data["0.0"]["BS Delta"]["es_95"]
                    - old_data["0.0"]["Deep Hedger"]["es_95"]
                ),
                "mean_pl_dh": old_data["0.0"]["Deep Hedger"]["mean_pnl"],
                "std_pl_dh": old_data["0.0"]["Deep Hedger"]["std_pnl"],
            },
            "lambda_0.001": {
                "es95_bs": old_data["0.001"]["BS Delta"]["es_95"],
                "es95_dh": old_data["0.001"]["Deep Hedger"]["es_95"],
                "gamma": (
                    old_data["0.001"]["BS Delta"]["es_95"]
                    - old_data["0.001"]["Deep Hedger"]["es_95"]
                ),
            },
        }

    output: dict[str, Any] = {
        "meta": {
            "script": "canonical_rerun.py",
            "git_commit": _git_commit_sha(),
            "fix_applied": True,
            "seeds": seeds_to_run,
            "cost_lambdas": COST_LAMBDAS,
            "n_train": DATASET_KW["n_train"],
            "n_val": DATASET_KW["n_val"],
            "n_test": DATASET_KW["n_test"],
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "total_wall_clock_s": total_wall,
        },
        "per_seed": per_seed,
        "aggregated": aggregated,
        "reproducibility_check": repro,
        "old_canonical": old_canonical,
    }

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return None

    with open(OUTPUT_JSON, "w") as f:
        json.dump(_clean(output), f, indent=2)
    print(f"\n  Results saved to {OUTPUT_JSON}", flush=True)

    if repro is not None:
        with open(REPRO_JSON, "w") as f:
            json.dump(_clean(repro), f, indent=2)
        print(f"  Repro-check saved to {REPRO_JSON}", flush=True)

    print("\n  AGGREGATED (lambda=0.0):", flush=True)
    agg0 = aggregated["0.0"]
    print(f"    Gamma       : {agg0['gamma']['mean']:.4f} +- {agg0['gamma']['std']:.4f}  "
          f"(min={agg0['gamma']['min']:.4f}, max={agg0['gamma']['max']:.4f})")
    print(f"    ES_0.95_BS  : {agg0['es95_bs']['mean']:.4f} +- {agg0['es95_bs']['std']:.4f}")
    print(f"    ES_0.95_DH  : {agg0['es95_dh']['mean']:.4f} +- {agg0['es95_dh']['std']:.4f}")

if __name__ == "__main__":
    main()
