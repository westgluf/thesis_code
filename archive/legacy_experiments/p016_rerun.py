#!/usr/bin/env python
"""
Phase E1 — P01.6 rerun with fixed seeding (5 seeds at n=400).

Verifies that the hedging advantage Γ is preserved when the grid resolution
is refined from n=100 to n=400. Uses the seeding protocol fixed in Prompt B.

Runs with 5 seeds [7401, 7402, 7403, 7404, 7405] either:
  - serially via `run_experiment()` from `block1_validation_n400.py`, or
  - in parallel subprocesses (one per seed) via `--single-seed` / `--aggregate-only`.

Produces:
  - results/block1_v2/p016_5seeds.json           (per-seed + aggregated)
  - results/block1_v2/p016_seed{seed}.json       (per-seed artefacts)
  - results/block1_v2/p016_seed7401_rerun.json   (reproducibility subprocess)
  - results/block1_v2/p016_report.md             (human-readable verdict)

Run:
    # Serial (all 5 seeds in one process):
    python -u -m deep_hedging.experiments.p016_rerun
    # Single-seed mode (used by parallel driver or reproducibility):
    python -u -m deep_hedging.experiments.p016_rerun --single-seed 7401
    # Aggregate-only (combine per-seed JSONs into p016_5seeds.json):
    python -u -m deep_hedging.experiments.p016_rerun --aggregate-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import expected_shortfall

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "block1_v2"

SEEDS = [7401, 7402, 7403, 7404, 7405]

# Canonical rBergomi parameters
H = 0.07
ETA = 1.9
RHO = -0.7
XI0 = 0.235 ** 2
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 400
SIGMA_BS = 0.235

# Same training budget as canonical
N_TRAIN = 80_000
N_VAL = 20_000
N_TEST = 50_000
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 2048
LR = 1e-3
ALPHA = 0.95
COST_LAMBDA = 0.0

# Canonical n=100 reference (from Prompt B)
CANONICAL_N100_GAMMA_MEAN = 1.1479
CANONICAL_N100_GAMMA_STD = 0.0761


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _first_weight_sum(model: torch.nn.Module) -> float:
    for p in model.parameters():
        if p.ndim >= 2:
            return float(p.detach().flatten().sum().cpu())
    return 0.0


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------


def run_single_seed(seed: int, verbose: bool = True) -> dict[str, Any]:
    """Run P01.6 for a single seed at n=400.

    - Simulate rBergomi paths (n=400, canonical calibration)
    - Train DH with fixed seeding
    - Evaluate both BS delta and DH on common test set
    - Return per-seed record
    """
    print(f"\n{'='*70}", flush=True)
    print(f"  P01.6 seed {seed}  (n_steps={N_STEPS})", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.perf_counter()

    device = torch.device("cpu")

    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0,
    )

    # Shared test set across seeds: deterministic test_seed offset
    test_seed = 7499
    print(f"  Generating test set: {N_TEST} paths, test_seed={test_seed}...",
          flush=True)
    S_test, _, _ = sim.simulate(N_TEST, S0=S0, seed=test_seed, device=device)
    payoff_test = compute_payoff(S_test, K, "call")

    # p0 estimate from independent training batch
    p0_seed = test_seed + 1000
    print(f"  Estimating p0 from independent MC batch (seed={p0_seed})...",
          flush=True)
    S_p0, _, _ = sim.simulate(N_TRAIN, S0=S0, seed=p0_seed, device=device)
    payoff_p0 = compute_payoff(S_p0, K, "call")
    p0 = float(payoff_p0.mean())
    del S_p0, payoff_p0
    gc.collect()
    print(f"  p0 = {p0:.4f}", flush=True)

    # BS delta on test set
    bs_hedger = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
    deltas_bs = bs_hedger.hedge_paths(S_test)
    pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, COST_LAMBDA)
    es_bs = float(expected_shortfall(pnl_bs, ALPHA))
    print(f"  BS ES_0.95 = {es_bs:.4f}", flush=True)

    # Generate training data with per-seed train_seed
    train_seed = 8400 + (seed - 7400)
    print(f"  Generating training data: {N_TRAIN+N_VAL} paths, "
          f"train_seed={train_seed}...", flush=True)
    S_all, _, _ = sim.simulate(N_TRAIN + N_VAL, S0=S0, seed=train_seed, device=device)
    S_train = S_all[:N_TRAIN]
    S_val = S_all[N_TRAIN:]
    del S_all
    gc.collect()

    payoff_train = compute_payoff(S_train, K, "call")
    p0_train = float(payoff_train.mean())
    print(f"  p0_train = {p0_train:.4f}", flush=True)

    # Seeded DH creation
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)

    tt = time.perf_counter()
    history = train_deep_hedger(
        model, S_train, S_val,
        K=K, T=T, S0=S0, p0=p0_train,
        cost_lambda=COST_LAMBDA, alpha=ALPHA,
        lr=LR, batch_size=BATCH_SIZE,
        epochs=EPOCHS, patience=PATIENCE,
        device=device, verbose=verbose,
    )
    train_time = time.perf_counter() - tt
    print(f"  Training done in {train_time/60:.1f} min  "
          f"best_epoch={history['best_epoch']}", flush=True)

    del S_train, S_val, payoff_train
    gc.collect()

    # Evaluate DH on test set
    model.eval()
    with torch.no_grad():
        deltas_dh = model.hedge_paths(S_test.float(), T, S0).to(S_test.dtype)
        pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff_test, p0, COST_LAMBDA)
    es_dh = float(expected_shortfall(pnl_dh, ALPHA))
    gamma = es_bs - es_dh

    fw_sum = _first_weight_sum(model)
    wall = time.perf_counter() - t0

    record = {
        "seed": seed,
        "n_steps": N_STEPS,
        "es95_bs": es_bs,
        "es95_dh": es_dh,
        "gamma": gamma,
        "mean_pl_bs": float(pnl_bs.mean()),
        "std_pl_bs": float(pnl_bs.std()),
        "mean_pl_dh": float(pnl_dh.mean()),
        "std_pl_dh": float(pnl_dh.std()),
        "p0": p0,
        "p0_train": p0_train,
        "train_time_s": train_time,
        "wall_clock_s": wall,
        "best_epoch": int(history["best_epoch"]),
        "best_val_risk": float(history["best_val_risk"]),
        "final_train_risk": float(history["train_risk"][-1]),
        "final_val_risk": float(history["val_risk"][-1]),
        "first_weight_sum": fw_sum,
    }
    print(f"\n  seed {seed} summary:", flush=True)
    print(f"    ES_0.95 BS = {es_bs:.4f}", flush=True)
    print(f"    ES_0.95 DH = {es_dh:.4f}", flush=True)
    print(f"    Γ          = {gamma:+.4f}", flush=True)
    print(f"    wall       = {wall/60:.1f} min", flush=True)
    return record


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    t_crit = 2.776 if n == 5 else (3.182 if n == 4 else (4.303 if n == 3 else 1.96))
    half = t_crit * se
    return {
        "mean": mean, "std": std, "se": se,
        "ci95_lower": mean - half, "ci95_upper": mean + half,
        "all_values": [float(v) for v in arr],
        "min": float(arr.min()) if n > 0 else 0.0,
        "max": float(arr.max()) if n > 0 else 0.0,
        "n": n,
    }


def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid = {s: r for s, r in per_seed.items() if "error" not in r}
    if not valid:
        return {}
    keys = ["gamma", "es95_bs", "es95_dh",
            "mean_pl_bs", "std_pl_bs",
            "mean_pl_dh", "std_pl_dh", "p0"]
    return {k: _agg([valid[s][k] for s in valid]) for k in keys}


def verdict(agg_gamma: dict[str, float]) -> str:
    """Compare Γ(n=400) to canonical Γ(n=100)."""
    m = agg_gamma["mean"]
    s = agg_gamma["std"]
    canonical_m = CANONICAL_N100_GAMMA_MEAN
    canonical_s = CANONICAL_N100_GAMMA_STD

    # Sign check
    if m * canonical_m < 0:
        return "INCONSISTENT"

    # 2σ overlap check
    lo_400 = m - 2 * s
    hi_400 = m + 2 * s
    lo_100 = canonical_m - 2 * canonical_s
    hi_100 = canonical_m + 2 * canonical_s
    overlap = not (hi_400 < lo_100 or hi_100 < lo_400)

    if overlap:
        return "PRESERVED"
    # Same sign but outside 2σ
    return "SHIFTED"


# ---------------------------------------------------------------------------
# Reproducibility check
# ---------------------------------------------------------------------------


def run_reproducibility_check(original: dict[str, Any]) -> dict[str, Any]:
    """Re-run seed 7401 in fresh subprocess and compare outputs."""
    print("\n" + "=" * 70, flush=True)
    print("  P01.6 REPRODUCIBILITY — seed 7401 in fresh subprocess", flush=True)
    print("=" * 70, flush=True)

    out_path = OUT_DIR / "p016_seed7401_rerun.json"
    cmd = [
        sys.executable, "-u", "-m",
        "deep_hedging.experiments.p016_rerun",
        "--single-seed", "7401",
        "--single-seed-output", str(out_path),
        "--verbose",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    print(f"  Subprocess finished in {wall/60:.1f} min  "
          f"(exit={result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1500:]}", flush=True)
        return {"error": "subprocess failed", "stderr_tail": result.stderr[-2000:]}

    with open(out_path) as f:
        rerun = json.load(f)

    match = {
        "es95_bs": original["es95_bs"] == rerun["es95_bs"],
        "es95_dh": original["es95_dh"] == rerun["es95_dh"],
        "gamma": original["gamma"] == rerun["gamma"],
        "first_weight_sum": original["first_weight_sum"] == rerun["first_weight_sum"],
    }
    all_match = all(match.values())
    r = {
        "seed": 7401,
        "original": {k: original[k] for k in ("es95_bs", "es95_dh", "gamma",
                                                "first_weight_sum")},
        "rerun": {k: rerun[k] for k in ("es95_bs", "es95_dh", "gamma",
                                          "first_weight_sum")},
        "match": match,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }
    for k, v in match.items():
        print(f"  {k:20s}: "
              f"orig={original[k]:.6f} rerun={rerun[k]:.6f} match={v}",
              flush=True)
    print(f"  VERDICT: {r['verdict']}", flush=True)
    return r


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(output: dict[str, Any], path: Path) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    meta = output["meta"]
    per = output["per_seed"]
    agg = output["aggregated"]
    repro = output.get("reproducibility_check")

    lines = []
    lines.append("# P01.6 Rerun Results — n=400 Validation with Fixed Seeding")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append(f"Git commit: {meta['git_commit']}")
    lines.append("Script: `deep_hedging/experiments/p016_rerun.py`")
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Rough Bergomi at canonical calibration "
                 f"(H={H}, η={ETA}, ρ={RHO}, ξ₀={XI0:.6f})")
    lines.append(f"- Grid resolution: n={N_STEPS} (vs n=100 canonical)")
    lines.append(f"- Training: {N_TRAIN} train / {N_VAL} val / {N_TEST} test, "
                 f"{EPOCHS} epochs, patience={PATIENCE}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility check")
    lines.append("")
    if repro is not None and "error" not in repro:
        lines.append("Seed 7401 rerun in fresh subprocess:")
        lines.append("")
        lines.append("| Metric | Original | Rerun | Match? |")
        lines.append("|---|---|---|---|")
        for k in ("gamma", "es95_bs", "es95_dh", "first_weight_sum"):
            lbl = {"gamma": "Γ", "es95_bs": "ES_BS",
                   "es95_dh": "ES_DH",
                   "first_weight_sum": "first_weight_sum"}[k]
            m = "✓" if repro["match"][k] else "✗"
            lines.append(f"| {lbl} | {repro['original'][k]:.6f} | "
                         f"{repro['rerun'][k]:.6f} | {m} |")
        lines.append("")
        lines.append(f"Verdict: **{repro['verdict']}**")
    else:
        lines.append("_Not available._")
    lines.append("")

    # Per-seed
    lines.append("## Per-seed results")
    lines.append("")
    lines.append("| Seed | ES_BS | ES_DH | Γ | Mean P&L (DH) | Std P&L (DH) |")
    lines.append("|---|---|---|---|---|---|")
    for s in sorted(per.keys(), key=int):
        r = per[s]
        if "error" in r:
            lines.append(f"| {s} | ERROR | ERROR | ERROR | — | — |")
            continue
        lines.append(
            f"| {s} | {r['es95_bs']:.4f} | {r['es95_dh']:.4f} | "
            f"{r['gamma']:+.4f} | {r['mean_pl_dh']:+.4f} | {r['std_pl_dh']:.4f} |"
        )
    lines.append("")

    # Aggregate
    lines.append("## Aggregate")
    lines.append("")
    g = agg["gamma"]
    lines.append(f"- Γ(n={N_STEPS}) = {g['mean']:+.4f} ± {g['std']:.4f}")
    lines.append(f"- 95% CI: [{g['ci95_lower']:+.4f}, {g['ci95_upper']:+.4f}]")
    lines.append("")

    # Comparison
    lines.append("## Comparison to canonical (n=100)")
    lines.append("")
    lines.append(f"- Γ(n=100) from Prompt B baseline: "
                 f"{CANONICAL_N100_GAMMA_MEAN:+.4f} ± {CANONICAL_N100_GAMMA_STD:.4f}")
    lines.append(f"- Γ(n={N_STEPS}) from this run: "
                 f"{g['mean']:+.4f} ± {g['std']:.4f}")
    lines.append(f"- Absolute difference: "
                 f"{abs(g['mean'] - CANONICAL_N100_GAMMA_MEAN):.4f}")

    lo_400 = g["mean"] - 2 * g["std"]
    hi_400 = g["mean"] + 2 * g["std"]
    lo_100 = CANONICAL_N100_GAMMA_MEAN - 2 * CANONICAL_N100_GAMMA_STD
    hi_100 = CANONICAL_N100_GAMMA_MEAN + 2 * CANONICAL_N100_GAMMA_STD
    overlap = not (hi_400 < lo_100 or hi_100 < lo_400)
    lines.append(f"- Overlap of 95% CIs ([{lo_100:+.4f}, {hi_100:+.4f}] vs "
                 f"[{lo_400:+.4f}, {hi_400:+.4f}]): {'YES' if overlap else 'NO'}")
    lines.append("")

    # Verdict
    v = verdict(g)
    lines.append("## Verdict")
    lines.append("")
    explanations = {
        "PRESERVED": ("Γ(n=400) within 2σ of Γ(n=100) → coarse-grid canonical "
                      "Γ is robust under grid refinement."),
        "SHIFTED": ("Γ(n=400) outside 2σ but same sign → refinement produces "
                    "a different value but the qualitative claim (DH beats BS) "
                    "is preserved."),
        "INCONSISTENT": ("Γ(n=400) has opposite sign or implausibly different "
                         "→ investigate path simulator, training convergence, "
                         "or evaluation code."),
    }
    lines.append(f"**{v}** — {explanations[v]}")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"  Wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------


def _save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="P01.6 rerun (5 seeds at n=400)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None)
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--single-seed-output", type=str, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_seed is not None:
        r = run_single_seed(args.single_seed, verbose=args.verbose)
        out_path = Path(args.single_seed_output) if args.single_seed_output else (
            output_dir / f"p016_seed{args.single_seed}.json"
        )
        _save_json(r, out_path)
        print(f"\n  Wrote {out_path}", flush=True)
        return

    seeds_to_run = args.seeds_only if args.seeds_only else SEEDS

    if args.aggregate_only:
        print("=" * 70, flush=True)
        print(f"  AGGREGATE-ONLY MODE — loading per-seed JSONs for seeds "
              f"{seeds_to_run}", flush=True)
        print("=" * 70, flush=True)
        per_seed: dict[str, dict[str, Any]] = {}
        for s in seeds_to_run:
            p = output_dir / f"p016_seed{s}.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing per-seed JSON: {p}")
            with open(p) as f:
                per_seed[str(s)] = json.load(f)
            print(f"  Loaded {p}", flush=True)

        aggregated = aggregate(per_seed)

        # Reproducibility check via subprocess
        repro = None
        if not args.skip_reproducibility and "7401" in per_seed \
                and "error" not in per_seed["7401"]:
            repro = run_reproducibility_check(per_seed["7401"])

        meta = {
            "script": "deep_hedging/experiments/p016_rerun.py",
            "git_commit": _git_commit_sha(),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "seeds": seeds_to_run,
            "parameters": {
                "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
                "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
                "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                "epochs": EPOCHS, "patience": PATIENCE,
                "batch_size": BATCH_SIZE, "lr": LR,
                "alpha": ALPHA, "cost_lambda": COST_LAMBDA,
            },
            "canonical_n100": {
                "gamma_mean": CANONICAL_N100_GAMMA_MEAN,
                "gamma_std": CANONICAL_N100_GAMMA_STD,
                "source": "results/canonical_v2/baseline_5seeds.json",
            },
        }
        final = {
            "meta": meta,
            "per_seed": per_seed,
            "aggregated": aggregated,
            "reproducibility_check": repro,
        }
        final_path = output_dir / "p016_5seeds.json"
        _save_json(final, final_path)
        print(f"\n  Wrote {final_path}", flush=True)
        write_report(final, output_dir / "p016_report.md")

        g = aggregated["gamma"]
        v = verdict(g)
        print("\n" + "=" * 70, flush=True)
        print(f"  HEADLINE: Γ(n=400) = {g['mean']:+.4f} ± {g['std']:.4f}  "
              f"(verdict: {v})", flush=True)
        print(f"  Canonical (n=100): "
              f"{CANONICAL_N100_GAMMA_MEAN:+.4f} ± {CANONICAL_N100_GAMMA_STD:.4f}",
              flush=True)
        print("=" * 70, flush=True)
        return

    # Serial mode: run all seeds in this process
    print("=" * 70, flush=True)
    print("  P01.6 RERUN — 5 seeds at n=400 (serial)", flush=True)
    print(f"  seeds: {seeds_to_run}", flush=True)
    print(f"  commit: {_git_commit_sha()}", flush=True)
    print("=" * 70, flush=True)

    per_seed = {}
    total_t0 = time.perf_counter()
    for seed in seeds_to_run:
        try:
            r = run_single_seed(seed, verbose=args.verbose)
            per_seed[str(seed)] = r
            _save_json(r, output_dir / f"p016_seed{seed}.json")
        except Exception as exc:
            print(f"\n  ERROR seed {seed}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            per_seed[str(seed)] = {"seed": seed, "error": str(exc)}
        # intermediate save
        intermediate = {
            "meta": {
                "script": "deep_hedging/experiments/p016_rerun.py",
                "git_commit": _git_commit_sha(),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "seeds_complete": list(per_seed),
                "seeds_planned": seeds_to_run,
            },
            "per_seed": per_seed,
        }
        _save_json(intermediate, output_dir / "p016_5seeds.json")

    total_wall = time.perf_counter() - total_t0
    print(f"\n  Total wall (5 seeds): {total_wall/60:.1f} min", flush=True)
    aggregated = aggregate(per_seed)

    repro = None
    if not args.skip_reproducibility and "7401" in per_seed \
            and "error" not in per_seed["7401"]:
        repro = run_reproducibility_check(per_seed["7401"])

    meta = {
        "script": "deep_hedging/experiments/p016_rerun.py",
        "git_commit": _git_commit_sha(),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds_to_run,
        "total_wall_clock_s": total_wall,
        "parameters": {
            "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
            "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
            "epochs": EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR,
            "alpha": ALPHA, "cost_lambda": COST_LAMBDA,
        },
        "canonical_n100": {
            "gamma_mean": CANONICAL_N100_GAMMA_MEAN,
            "gamma_std": CANONICAL_N100_GAMMA_STD,
            "source": "results/canonical_v2/baseline_5seeds.json",
        },
    }
    final = {"meta": meta, "per_seed": per_seed,
             "aggregated": aggregated, "reproducibility_check": repro}
    _save_json(final, output_dir / "p016_5seeds.json")
    write_report(final, output_dir / "p016_report.md")

    g = aggregated["gamma"]
    v = verdict(g)
    print("\n" + "=" * 70, flush=True)
    print(f"  HEADLINE: Γ(n=400) = {g['mean']:+.4f} ± {g['std']:.4f}  "
          f"(verdict: {v})", flush=True)
    print(f"  Canonical (n=100): "
          f"{CANONICAL_N100_GAMMA_MEAN:+.4f} ± {CANONICAL_N100_GAMMA_STD:.4f}",
          flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
