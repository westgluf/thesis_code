#!/usr/bin/env python
"""
Decomposition re-run across 5 seeds with the seeding fix applied.

Runs the diagnostic-controls pipeline end-to-end for each seed in
[3024, 3025, 3026, 3027, 3028], then feeds the outputs through
`build_decomposition.py`'s arithmetic to obtain a 5-bucket percentage
decomposition for each seed. Finally aggregates mean ± std across seeds.

Produces:
  - results/canonical_v2/decomposition_5seeds.json

Run:
    python -u -m deep_hedging.experiments.decomposition_rerun
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

from deep_hedging.experiments.diagnostic_controls import DiagnosticControlsExperiment
from deep_hedging.experiments.build_decomposition import (
    compute_2x2_factorial, decompose_2x2, compute_3x3_anova, build_five_components,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "canonical_v2"
OUTPUT_JSON = RESULTS_DIR / "decomposition_5seeds.json"
REPRO_JSON = RESULTS_DIR / "decomposition_reproducibility.json"

SEEDS = [3024, 3025, 3026, 3027, 3028]

# Smaller workload than the canonical baseline (5 experiments per seed)
N_TRAIN = 60_000
N_VAL = 10_000
N_TEST = 30_000
EPOCHS = 150
PATIENCE = 25

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

def _strip_for_json(results: dict[str, Any]) -> dict[str, Any]:
    """Strip torch.Tensor and history objects for JSON serialisation."""
    def recurse(obj):
        if isinstance(obj, torch.Tensor):
            return None
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items() if recurse(v) is not None}
        if isinstance(obj, list):
            return [recurse(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return None
    return recurse(results)

def _collect_for_decomp(results: dict[str, Any]) -> dict[str, Any]:
    """Keep just the keys needed by build_decomposition arithmetic."""
    out: dict[str, Any] = {}
    # Experiment A — need es95_bs, es95_dh, gamma
    A = results["A"]
    out["A"] = {
        "H": A["H"], "eta": A["eta"],
        "es95_bs": A["es95_bs"], "es95_dh": A["es95_dh"],
        "gamma": A["gamma"],
    }
    # Experiment A'
    Ap = results["A_prime"]
    out["A_prime"] = {
        "H": Ap["H"], "eta": Ap["eta"],
        "es95_bs": Ap["es95_bs"], "es95_dh": Ap["es95_dh"],
        "gamma": Ap["gamma"],
    }
    # Experiment B — list of dicts (optional)
    if "B" in results:
        out["B"] = [
            {"H": r["H"], "eta": r["eta"], "es95_bs": r["es95_bs"],
             "es95_dh": r["es95_dh"], "gamma": r["gamma"]}
            for r in results["B"]
        ]
    # Experiment C — need bs, dh_mse, dh_es, gamma_decomposition
    C = results["C"]
    out["C"] = {
        "bs": {"es95": C["bs"]["es95"]},
        "dh_mse": {"es95": C["dh_mse"]["es95"]},
        "dh_mean": {"es95": C["dh_mean"]["es95"]},
        "dh_es": {"es95": C["dh_es"]["es95"]},
        "gamma_decomposition": C["gamma_decomposition"],
    }
    # Experiment D — list of dicts
    out["D"] = [
        {"H": r["H"], "eta": r["eta"], "es95_bs": r["es95_bs"],
         "es95_dh": r["es95_dh"], "gamma": r["gamma"]}
        for r in results["D"]
    ]
    return out

def _aggregate_across_existing(seed_files: list[Path], skip_B: bool = False) -> dict[str, Any]:
    """Load per-seed JSON files, combine into one dict (for parallel aggregation)."""
    per_seed: dict[str, dict[str, Any]] = {}
    for p in seed_files:
        with open(p) as f:
            data = json.load(f)
        seed = str(data["seed"])
        per_seed[seed] = data
    return per_seed

def run_one_seed(seed: int, skip_B: bool = False) -> dict[str, Any]:
    """Run A, A', (B), C, D for a single seed and compute 5-bucket decomposition.

    B is optional because it is not used by build_decomposition.py's
    five-component split (only A, A', C, D are needed). Skipping B cuts
    ~20-25 min per seed.
    """
    print(f"\n{'='*70}")
    print(f"  SEED {seed}{' (skip B)' if skip_B else ''}")
    print(f"{'='*70}", flush=True)
    t0 = time.perf_counter()

    exp = DiagnosticControlsExperiment()
    # Each experiment uses `seed` internally, which now triggers proper seeding
    # of torch + numpy via the fix.
    exp.run_experiment_A(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                         epochs=EPOCHS, seed=seed)
    exp.run_experiment_A_prime(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                               epochs=EPOCHS, seed=seed)
    if not skip_B:
        exp.run_experiment_B(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                             epochs=EPOCHS, seed=seed)
    exp.run_experiment_C(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                         epochs=EPOCHS, seed=seed)
    exp.run_experiment_D(n_train=50_000, n_val=10_000, n_test=20_000,
                         epochs=100, seed=seed)

    collected = _collect_for_decomp(exp.results)

    # Run build_decomposition arithmetic
    factorial = compute_2x2_factorial(collected)
    decomp_2x2 = decompose_2x2(factorial)
    anova = compute_3x3_anova(collected["D"])
    decomp = build_five_components(decomp_2x2, anova)

    wall = time.perf_counter() - t0
    print(f"\n  Seed {seed} done in {wall/60:.1f} min")
    print(f"  Gamma_total = {decomp['Gamma_total']:+.4f}")
    pct = decomp["percentages_of_total"]
    print(f"  Objective:    {pct['objective']:+7.2f} %")
    print(f"  Interaction:  {pct['interaction']:+7.2f} %")
    print(f"  Stoch vol:    {pct['stoch_vol']:+7.2f} %")
    print(f"  Roughness:    {pct['roughness']:+7.2f} %")
    print(f"  Architecture: {pct['architecture']:+7.2f} %", flush=True)

    del exp
    gc.collect()

    return {
        "seed": seed,
        "wall_clock_s": wall,
        "raw_experiments": collected,
        "factorial_2x2": factorial,
        "decomp_2x2": decomp_2x2,
        "anova_3x3": anova,
        "decomposition": decomp,
    }

def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate percentages and total across seeds."""
    def agg_scalar(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se = std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
        return {
            "mean": mean, "std": std, "se": se,
            "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
            "min": float(arr.min()), "max": float(arr.max()),
            "n": int(len(arr)),
        }

    aggregated: dict[str, Any] = {"percentages": {}, "absolute": {}}
    pct_keys = ["objective", "interaction", "stoch_vol", "roughness", "architecture"]
    abs_keys = ["Gamma_total", "Gamma_architecture", "Gamma_objective",
                "Gamma_stoch_vol", "Gamma_roughness", "Gamma_interaction_total"]

    for key in pct_keys:
        values = [per_seed[str(s)]["decomposition"]["percentages_of_total"][key]
                  for s in SEEDS]
        aggregated["percentages"][key] = agg_scalar(values)
    for key in abs_keys:
        values = [per_seed[str(s)]["decomposition"][key] for s in SEEDS]
        aggregated["absolute"][key] = agg_scalar(values)

    return aggregated

def reproducibility_check() -> dict[str, Any]:
    """Re-run seed 3024 in two fresh subprocesses and compare."""
    print(f"\n{'='*70}")
    print("  REPRODUCIBILITY CHECK — seed 3024 in two fresh subprocesses")
    print(f"{'='*70}", flush=True)

    check_json1 = RESULTS_DIR / "decomp_repro_run1.json"
    check_json2 = RESULTS_DIR / "decomp_repro_run2.json"
    check_json1.parent.mkdir(parents=True, exist_ok=True)

    for i, out_path in enumerate([check_json1, check_json2], 1):
        print(f"\n  Subprocess {i}: writing to {out_path}", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-u", "-m",
             "deep_hedging.experiments.decomposition_rerun",
             "--single-seed", "3024",
             "--single-seed-output", str(out_path)],
            cwd=REPO_ROOT, capture_output=True, text=True,
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

    result: dict[str, Any] = {
        "seed": 3024,
        "run1": {"gamma_total": g1, "objective_pct": obj1, "esA_dh": esA_1},
        "run2": {"gamma_total": g2, "objective_pct": obj2, "esA_dh": esA_2},
        "gamma_match": gamma_match,
        "objective_pct_match": pct_match,
        "experiment_A_match": esA_match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }
    print(f"\n  Gamma_total : run1={g1:.6f}  run2={g2:.6f}  match={gamma_match}")
    print(f"  Objective % : run1={obj1:.4f}  run2={obj2:.4f}  match={pct_match}")
    print(f"  ES_A_dh     : run1={esA_1:.6f}  run2={esA_2:.6f}  match={esA_match}")
    print(f"  VERDICT     : {result['verdict']}", flush=True)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--single-seed-output", type=str, default=None)
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--skip-B", action="store_true",
                        help="Skip Experiment B (not needed for 5-bucket split).")
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip all training; load per-seed JSONs and aggregate.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.single_seed is not None:
        r = run_one_seed(args.single_seed, skip_B=args.skip_B)
        out_path = Path(args.single_seed_output) if args.single_seed_output else (
            RESULTS_DIR / f"decomp_seed_{args.single_seed}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_strip_for_json(r), f, indent=2)
        print(f"\n  Wrote {out_path}")
        return

    seeds_to_run = args.seeds_only if args.seeds_only else SEEDS

    if args.aggregate_only:
        print("=" * 70)
        print(f"  AGGREGATE-ONLY mode — loading per-seed JSONs for seeds {seeds_to_run}")
        print("=" * 70, flush=True)
        per_seed: dict[str, dict[str, Any]] = {}
        for s in seeds_to_run:
            p = RESULTS_DIR / f"decomp_seed_{s}.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing per-seed JSON: {p}")
            with open(p) as f:
                per_seed[str(s)] = json.load(f)
            print(f"  Loaded {p}", flush=True)
        aggregated = aggregate(per_seed)
        # Reload old canonical
        old_canonical_path = REPO_ROOT / "figures" / "decomposition_closed.json"
        old_canonical: dict[str, Any] = {}
        if old_canonical_path.exists():
            with open(old_canonical_path) as f:
                old = json.load(f)
            old_canonical = {
                "source": str(old_canonical_path),
                "Gamma_total": old["decomposition"]["Gamma_total"],
                "percentages": old["decomposition"]["percentages_of_total"],
            }
        output = {
            "meta": {
                "script": "decomposition_rerun.py (aggregate-only)",
                "git_commit": _git_commit_sha(),
                "fix_applied": True,
                "seeds": seeds_to_run,
                "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                "epochs": EPOCHS, "patience": PATIENCE,
                "skip_B": args.skip_B,
                "parallel_mode": True,
            },
            "per_seed": per_seed,
            "aggregated": aggregated,
            "reproducibility_check": None,
            "old_canonical": old_canonical,
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(_strip_for_json(output), f, indent=2)
        print(f"\n  Results saved to {OUTPUT_JSON}")
        return

    print("=" * 70)
    print("  DECOMPOSITION RE-RUN — diagnostic controls, 5 seeds")
    print(f"  seeds: {seeds_to_run}")
    print(f"  commit: {_git_commit_sha()}")
    print("=" * 70, flush=True)

    per_seed: dict[str, dict[str, Any]] = {}
    total_t0 = time.perf_counter()
    for seed in seeds_to_run:
        r = run_one_seed(seed, skip_B=args.skip_B)
        per_seed[str(seed)] = r
        # incremental save
        partial = {
            "meta": {
                "script": "decomposition_rerun.py",
                "git_commit": _git_commit_sha(),
                "fix_applied": True,
                "seeds_complete": list(per_seed.keys()),
                "seeds_planned": seeds_to_run,
            },
            "per_seed": per_seed,
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(_strip_for_json(partial), f, indent=2)
        print(f"  (saved partial to {OUTPUT_JSON})", flush=True)

    total_wall = time.perf_counter() - total_t0
    print(f"\n  TOTAL wall-clock (5 seeds): {total_wall/60:.1f} min", flush=True)

    aggregated = aggregate(per_seed)

    repro = None
    if not args.skip_reproducibility:
        repro = reproducibility_check()

    # Old canonical decomposition percentages
    old_canonical_path = REPO_ROOT / "figures" / "decomposition_closed.json"
    old_canonical: dict[str, Any] = {}
    if old_canonical_path.exists():
        with open(old_canonical_path) as f:
            old = json.load(f)
        old_canonical = {
            "source": str(old_canonical_path),
            "Gamma_total": old["decomposition"]["Gamma_total"],
            "percentages": old["decomposition"]["percentages_of_total"],
        }

    output = {
        "meta": {
            "script": "decomposition_rerun.py",
            "git_commit": _git_commit_sha(),
            "fix_applied": True,
            "seeds": seeds_to_run,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
            "epochs": EPOCHS, "patience": PATIENCE,
            "total_wall_clock_s": total_wall,
        },
        "per_seed": per_seed,
        "aggregated": aggregated,
        "reproducibility_check": repro,
        "old_canonical": old_canonical,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(_strip_for_json(output), f, indent=2)
    print(f"\n  Results saved to {OUTPUT_JSON}", flush=True)

    if repro is not None:
        with open(REPRO_JSON, "w") as f:
            json.dump(_strip_for_json(repro), f, indent=2)
        print(f"  Repro-check saved to {REPRO_JSON}", flush=True)

if __name__ == "__main__":
    main()
