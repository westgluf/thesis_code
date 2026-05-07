#!/usr/bin/env python
"""
Phase E2 — P01.7 rerun with fixed seeding (4 cells × 3 seeds each at n=400).

Runs the four decomposition-related cells (A: η=0 MSE, B: η=1.9 MSE, C: H2 with
λ=0.001, D: GBM-pretrained transfer eval) at the refined grid n=400 using the
seeding protocol fixed in Prompt B.

Seeds (disjoint from all existing codebase seeds):
  Cell A: [7711, 7712, 7713]
  Cell B: [7721, 7722, 7723]
  Cell C: [7731, 7732, 7733]
  Cell D: [7741, 7742, 7743]

The cell runner functions `run_cell_A/B/C/D` already exist in
`block1_extended_validation.py` and already apply the fix from Prompt B
(torch.manual_seed / np.random.seed before DeepHedgerFNN). This wrapper
simply re-targets them at a new output directory and provides parallel /
reproducibility helpers.

Run:
    python -u -m deep_hedging.experiments.p017_rerun
    # Single cell for parallelism:
    python -u -m deep_hedging.experiments.p017_rerun --single-cell A
    # Reproducibility subprocess (A with only seed 7711):
    python -u -m deep_hedging.experiments.p017_rerun \
        --single-cell A --seeds-only 7711 \
        --single-cell-output results/block1_v2/p017_cellA_seed7711_rerun.json
    # Aggregate per-cell JSONs into p017_results.json:
    python -u -m deep_hedging.experiments.p017_rerun --aggregate-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from deep_hedging.experiments.block1_extended_validation import (
    BUDGET_DIAG,
    BUDGET_H2,
    CANONICAL,
    N_BOOTSTRAP,
    N_TEST,
    RBERG,
    _global_verdict,
    run_cell_A,
    run_cell_B,
    run_cell_C,
    run_cell_D,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "block1_v2"

# Canonical seeds (new-protocol, disjoint from existing [7711,7712,7713] etc.
# in block1_extended_validation — we use the SAME integers, but the output
# directory is different so there is no file collision).
SEEDS = {
    "A": [7711, 7712, 7713],
    "B": [7721, 7722, 7723],
    "C": [7731, 7732, 7733],
    "D": [7741, 7742, 7743],
}
N_SIM = 400


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


def _run_cell(cell_name: str, seeds: list[int],
              n_sim: int = N_SIM, n_test: int = N_TEST,
              n_bootstrap: int = N_BOOTSTRAP) -> dict[str, Any]:
    """Dispatch to the appropriate cell runner."""
    print(f"\n{'='*70}", flush=True)
    print(f"  P01.7 Cell {cell_name}  seeds={seeds}  n_sim={n_sim}", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.perf_counter()
    if cell_name == "A":
        r = run_cell_A(n_sim, seeds, n_test, n_bootstrap, BUDGET_DIAG)
    elif cell_name == "B":
        r = run_cell_B(n_sim, seeds, n_test, n_bootstrap, BUDGET_DIAG)
    elif cell_name == "C":
        r = run_cell_C(n_sim, seeds, n_test, n_bootstrap, BUDGET_H2)
    elif cell_name == "D":
        r = run_cell_D(n_sim, seeds, n_test, n_bootstrap)
    else:
        raise ValueError(f"Unknown cell: {cell_name}")
    r["cell_name"] = cell_name
    r["seeds"] = seeds
    r["wall_clock_s"] = time.perf_counter() - t0
    return r


def _strip_for_json(obj):
    """Strip numpy/torch objects for JSON."""
    if hasattr(obj, "detach"):
        return None
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items()
                if _strip_for_json(v) is not None}
    if isinstance(obj, (list, tuple)):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return None  # skip arrays
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return None


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_strip_for_json(obj), f, indent=2)


# ---------------------------------------------------------------------------
# Reproducibility check
# ---------------------------------------------------------------------------


def run_reproducibility_check(original_cell_A: dict[str, Any]) -> dict[str, Any]:
    """Re-run Cell A with only seed 7711 in fresh subprocess."""
    print("\n" + "=" * 70, flush=True)
    print("  P01.7 REPRODUCIBILITY — Cell A seed 7711 in fresh subprocess",
          flush=True)
    print("=" * 70, flush=True)

    rerun_path = OUT_DIR / "p017_cellA_seed7711_rerun.json"
    cmd = [
        sys.executable, "-u", "-m",
        "deep_hedging.experiments.p017_rerun",
        "--single-cell", "A",
        "--seeds-only", "7711",
        "--single-cell-output", str(rerun_path),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    print(f"  Subprocess finished in {wall/60:.1f} min (exit={result.returncode})",
          flush=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1500:]}", flush=True)
        return {"error": "subprocess failed",
                "stderr_tail": result.stderr[-2000:]}

    with open(rerun_path) as f:
        rerun = json.load(f)

    # Extract seed-7711 data from both original & rerun
    original_7711 = original_cell_A["n400_per_seed"]["7711"]
    rerun_7711 = rerun["n400_per_seed"]["7711"]

    match_es_bs = original_7711["es95_bs"] == rerun_7711["es95_bs"]
    match_es_dh = original_7711["es95_dh"] == rerun_7711["es95_dh"]
    match_gamma = original_7711["gamma"] == rerun_7711["gamma"]
    all_match = match_es_bs and match_es_dh and match_gamma

    r = {
        "cell": "A", "seed": 7711,
        "original": {k: original_7711[k] for k in ("es95_bs", "es95_dh", "gamma")},
        "rerun": {k: rerun_7711[k] for k in ("es95_bs", "es95_dh", "gamma")},
        "match_es95_bs": match_es_bs,
        "match_es95_dh": match_es_dh,
        "match_gamma": match_gamma,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }
    print(f"  es95_bs: orig={original_7711['es95_bs']:.6f} "
          f"rerun={rerun_7711['es95_bs']:.6f} match={match_es_bs}", flush=True)
    print(f"  es95_dh: orig={original_7711['es95_dh']:.6f} "
          f"rerun={rerun_7711['es95_dh']:.6f} match={match_es_dh}", flush=True)
    print(f"  gamma:   orig={original_7711['gamma']:.6f} "
          f"rerun={rerun_7711['gamma']:.6f} match={match_gamma}", flush=True)
    print(f"  VERDICT: {r['verdict']}", flush=True)
    return r


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cell_aggregate(cell: dict[str, Any]) -> dict[str, Any]:
    """Extract per-seed Γ list and compute mean/std."""
    out = {"per_seed_gammas": [], "per_seed_es_bs": [], "per_seed_es_dh": []}
    if "n400_per_seed" in cell:
        for seed_str, r in cell["n400_per_seed"].items():
            if "error" in r:
                continue
            out["per_seed_gammas"].append(r["gamma"])
            out["per_seed_es_bs"].append(r["es95_bs"])
            out["per_seed_es_dh"].append(r["es95_dh"])
    elif "variant_C_match" in cell:
        # Cell C: only one seed; use variant_C_match (canonical grid match)
        out["per_seed_gammas"] = [cell["variant_C_match"]["gamma"]]
        out["per_seed_es_bs"] = [cell["variant_C_match"]["es95_bs"]]
        out["per_seed_es_dh"] = [cell["variant_C_match"]["es95_dh"]]
    elif "n400_results" in cell:
        # Cell D: eval only
        out["per_seed_gammas"] = [cell["n400_results"]["gamma"]]
        out["per_seed_es_bs"] = [cell["n400_results"]["es95_bs"]]
        out["per_seed_es_dh"] = [cell["n400_results"]["es95_dh"]]
    gs = np.array(out["per_seed_gammas"])
    if len(gs) > 0:
        out["gamma_mean"] = float(gs.mean())
        out["gamma_std"] = float(gs.std(ddof=1)) if len(gs) > 1 else 0.0
    else:
        out["gamma_mean"] = 0.0
        out["gamma_std"] = 0.0
    return out


def write_report(output: dict[str, Any], path: Path) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    meta = output["meta"]
    cells = output["cells"]
    repro = output.get("reproducibility_check")
    global_verdict = output.get("global_verdict")

    lines = []
    lines.append("# P01.7 Rerun Results — 4-Cell Extended Validation "
                 "with Fixed Seeding")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append(f"Git commit: {meta['git_commit']}")
    lines.append("Script: `deep_hedging/experiments/p017_rerun.py`")
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"Four validation cells at n={N_SIM} with fixed seeding:")
    lines.append("")
    lines.append(f"- **Cell A:** (η=0, MSE objective); canonical Γ = "
                 f"{CANONICAL['A']['gamma']:+.4f}")
    lines.append(f"- **Cell B:** (η=1.9, MSE objective); canonical Γ = "
                 f"{CANONICAL['B']['gamma']:+.4f}")
    lines.append(f"- **Cell C:** H2 with λ=0.001 (transaction-cost cell); "
                 f"canonical Γ = {CANONICAL['C']['gamma']:+.4f}")
    lines.append(f"- **Cell D:** GBM-pretrained transfer (eval only); "
                 f"canonical Γ = {CANONICAL['D']['gamma']:+.4f}")
    lines.append("")
    lines.append(f"Seeds per cell: A={SEEDS['A']}, B={SEEDS['B']}, "
                 f"C={SEEDS['C']}, D={SEEDS['D']}")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility check (Cell A, seed 7711)")
    lines.append("")
    if repro is not None and "error" not in repro:
        lines.append("| Metric | Original | Rerun | Match? |")
        lines.append("|---|---|---|---|")
        for k in ("gamma", "es95_bs", "es95_dh"):
            lbl = {"gamma": "Γ", "es95_bs": "ES_BS", "es95_dh": "ES_DH"}[k]
            m_key = {"gamma": "match_gamma", "es95_bs": "match_es95_bs",
                     "es95_dh": "match_es95_dh"}[k]
            m = "✓" if repro[m_key] else "✗"
            lines.append(f"| {lbl} | {repro['original'][k]:.6f} | "
                         f"{repro['rerun'][k]:.6f} | {m} |")
        lines.append("")
        lines.append(f"Verdict: **{repro['verdict']}**")
    else:
        lines.append("_Not available._")
    lines.append("")

    # Per-cell results
    lines.append("## Per-cell results")
    lines.append("")
    for cell_name in ("A", "B", "C", "D"):
        if cell_name not in cells:
            continue
        cell = cells[cell_name]
        agg = _cell_aggregate(cell)
        lines.append(f"### Cell {cell_name} — {cell.get('description', '')}")
        lines.append("")
        if cell.get("skipped", False):
            lines.append(f"_skipped: {cell.get('reason', 'unknown')}_")
            lines.append("")
            continue
        if cell_name in ("A", "B") and "n400_per_seed" in cell:
            lines.append("| Seed | ES_BS | ES_DH | Γ |")
            lines.append("|---|---|---|---|")
            for seed_str in sorted(cell["n400_per_seed"].keys(), key=int):
                r = cell["n400_per_seed"][seed_str]
                if "error" in r:
                    lines.append(f"| {seed_str} | ERROR | ERROR | ERROR |")
                    continue
                lines.append(f"| {seed_str} | {r['es95_bs']:.4f} | "
                             f"{r['es95_dh']:.4f} | {r['gamma']:+.4f} |")
            lines.append(f"| **Mean** | — | — | "
                         f"**{agg['gamma_mean']:+.4f} ± {agg['gamma_std']:.4f}** |")
        elif cell_name == "C" and "variant_C_match" in cell:
            lines.append("Cell C uses only the first seed by design "
                         "(evaluation-focused).")
            lines.append("")
            for variant in ("variant_C_match", "variant_C_subsample"):
                if variant not in cell:
                    continue
                v = cell[variant]
                lines.append(
                    f"- **{variant}** (n_rebal={v['n_rebal_effective']}): "
                    f"ES_BS={v['es95_bs']:.4f}, ES_DH={v['es95_dh']:.4f}, "
                    f"Γ={v['gamma']:+.4f}, 95%CI=[{v['ci_low']:+.4f}, "
                    f"{v['ci_high']:+.4f}]"
                )
        elif cell_name == "D" and "n400_results" in cell:
            r = cell["n400_results"]
            lines.append(f"ES_BS={r['es95_bs']:.4f}, ES_DH={r['es95_dh']:.4f}, "
                         f"Γ={r['gamma']:+.4f}, 95%CI=[{r['ci_low']:+.4f}, "
                         f"{r['ci_high']:+.4f}]")
        lines.append("")
        canonical_g = CANONICAL[cell_name]["gamma"]
        lines.append(f"Canonical Γ (n=100): {canonical_g:+.4f}  "
                     f"→ verdict: **{cell.get('cell_verdict', 'unknown')}**")
        lines.append("")

    # Combined verdict
    lines.append("## Combined verdict")
    lines.append("")
    for cell_name in ("A", "B", "C", "D"):
        if cell_name in cells:
            v = cells[cell_name].get("cell_verdict", "unknown")
            lines.append(f"- Cell {cell_name}: {v}")
    lines.append("")
    if global_verdict:
        lines.append(f"Overall: **{global_verdict}**")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"  Wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="P01.7 rerun (4 cells × 3 seeds)")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--single-cell", type=str, default=None,
                        help="Run just one cell (A/B/C/D).")
    parser.add_argument("--single-cell-output", type=str, default=None,
                        help="Output path for single-cell JSON.")
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None,
                        help="Override default seeds (e.g. for reproducibility).")
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Combine per-cell JSONs into p017_results.json.")
    parser.add_argument("--cells-only", nargs="+", type=str, default=None,
                        help="Run only these cells (space-separated).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_cell is not None:
        cell_name = args.single_cell
        seeds = args.seeds_only if args.seeds_only else SEEDS[cell_name]
        r = _run_cell(cell_name, seeds)
        out_path = Path(args.single_cell_output) if args.single_cell_output else (
            output_dir / f"p017_cell{cell_name}.json"
        )
        _save_json(r, out_path)
        print(f"\n  Wrote {out_path}", flush=True)
        return

    cells_to_run = args.cells_only if args.cells_only else ["A", "B", "C", "D"]

    if args.aggregate_only:
        print("=" * 70, flush=True)
        print(f"  AGGREGATE-ONLY MODE — loading per-cell JSONs for cells "
              f"{cells_to_run}", flush=True)
        print("=" * 70, flush=True)
        cells: dict[str, dict[str, Any]] = {}
        for c in cells_to_run:
            p = output_dir / f"p017_cell{c}.json"
            if not p.exists():
                print(f"  MISSING: {p} — skipping cell {c}", flush=True)
                continue
            with open(p) as f:
                cells[c] = json.load(f)
            print(f"  Loaded {p}", flush=True)

        verdicts = {name: cell.get("cell_verdict", "unknown")
                    for name, cell in cells.items()}
        gv = _global_verdict(verdicts)

        # Reproducibility
        repro = None
        if not args.skip_reproducibility and "A" in cells \
                and "n400_per_seed" in cells["A"] \
                and "7711" in cells["A"]["n400_per_seed"]:
            repro = run_reproducibility_check(cells["A"])

        output = {
            "meta": {
                "script": "deep_hedging/experiments/p017_rerun.py",
                "git_commit": _git_commit_sha(),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "n_sim": N_SIM, "seeds_by_cell": SEEDS,
                "n_test": N_TEST, "bootstrap_draws": N_BOOTSTRAP,
            },
            "cells": cells,
            "global_verdict": gv,
            "reproducibility_check": repro,
        }
        final_path = output_dir / "p017_results.json"
        _save_json(output, final_path)
        print(f"\n  Wrote {final_path}", flush=True)
        write_report(output, output_dir / "p017_report.md")

        print("\n" + "=" * 70, flush=True)
        print(f"  HEADLINE: global verdict = {gv}", flush=True)
        for c, v in verdicts.items():
            print(f"    Cell {c}: {v}", flush=True)
        print("=" * 70, flush=True)
        return

    # Serial mode: run all cells in this process
    print("=" * 70, flush=True)
    print(f"  P01.7 RERUN — cells {cells_to_run} × 3 seeds each (serial)",
          flush=True)
    print(f"  commit: {_git_commit_sha()}", flush=True)
    print("=" * 70, flush=True)

    cells: dict[str, dict[str, Any]] = {}
    total_t0 = time.perf_counter()
    for cell_name in cells_to_run:
        try:
            cells[cell_name] = _run_cell(cell_name, SEEDS[cell_name])
            _save_json(cells[cell_name], output_dir / f"p017_cell{cell_name}.json")
        except Exception as exc:
            print(f"\n  ERROR cell {cell_name}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            cells[cell_name] = {"error": str(exc), "cell_verdict": "ERROR"}

    total_wall = time.perf_counter() - total_t0

    verdicts = {name: cell.get("cell_verdict", "unknown")
                for name, cell in cells.items()}
    gv = _global_verdict(verdicts)

    repro = None
    if not args.skip_reproducibility and "A" in cells \
            and "n400_per_seed" in cells["A"] \
            and "7711" in cells["A"]["n400_per_seed"]:
        repro = run_reproducibility_check(cells["A"])

    output = {
        "meta": {
            "script": "deep_hedging/experiments/p017_rerun.py",
            "git_commit": _git_commit_sha(),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "n_sim": N_SIM, "seeds_by_cell": SEEDS,
            "n_test": N_TEST, "bootstrap_draws": N_BOOTSTRAP,
            "total_wall_clock_s": total_wall,
        },
        "cells": cells,
        "global_verdict": gv,
        "reproducibility_check": repro,
    }
    _save_json(output, output_dir / "p017_results.json")
    write_report(output, output_dir / "p017_report.md")

    print("\n" + "=" * 70, flush=True)
    print(f"  P01.7 done in {total_wall/60:.1f} min  global verdict = {gv}",
          flush=True)
    for c, v in verdicts.items():
        print(f"    Cell {c}: {v}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
