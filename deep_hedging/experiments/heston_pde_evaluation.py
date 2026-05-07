#!/usr/bin/env python
"""
Phase 3 — Heston PDE delta evaluation on rough Bergomi test paths (Phase J).

For each of 5 seeds [6024..6028]:
  1. Generate rough Bergomi paths at canonical calibration.
  2. Evaluate three strategies on the same test set:
       BS delta (sigma=sqrt(xi0))
       PluginDelta (BS-functional with realised variance)
       HestonPDEDelta (true Heston PDE delta, calibrated in Phase 2)
  3. Record per-strategy ES_0.95, ES_0.99, VaR, std, mean PL, turnover.

Reproducibility: seed 6024 re-run in a fresh subprocess must produce
byte-identical outputs.

Run:
    python -u -m deep_hedging.experiments.heston_pde_evaluation
    python -u -m deep_hedging.experiments.heston_pde_evaluation \\
        --single-seed 6024 --output results/heston_pde/seed_6024_rerun.json
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
from deep_hedging.hedging.delta_hedger import BlackScholesDelta, PluginDelta
from deep_hedging.hedging.heston_pde_delta import HestonPDEDelta
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import compute_all_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "heston_pde"

SEEDS = [6024, 6025, 6026, 6027, 6028]

# Canonical rough Bergomi
H = 0.07
ETA = 1.9
RHO_RB = -0.7
XI0 = 0.235 ** 2
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
SIGMA_BS = math.sqrt(XI0)

# Dataset sizes (matching Phase B canonical)
N_TRAIN = 80_000
N_VAL = 20_000
N_TEST = 50_000
N_TOTAL = N_TRAIN + N_VAL + N_TEST

COST_LAMBDA = 0.0
ALPHA = 0.95

def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"

def _load_calibration() -> dict[str, Any]:
    path = OUT_DIR / "calibration_data.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Run phase2_calibrate.py first."
        )
    with open(path) as f:
        return json.load(f)

def _mean_turnover(deltas: torch.Tensor) -> float:
    batch = deltas.shape[0]
    dtype, device = deltas.dtype, deltas.device
    delta_prev = torch.cat(
        [torch.zeros(batch, 1, dtype=dtype, device=device), deltas[:, :-1]],
        dim=1,
    )
    return float((deltas - delta_prev).abs().sum(dim=1).mean())

def _first_delta_sum(deltas: torch.Tensor) -> float:
    """Sum of the first column of deltas — reproducibility signature."""
    return float(deltas[:, 0].sum())

# ---------------------------------------------------------------------------
# Cached PDE solver (one per process)
# ---------------------------------------------------------------------------

_PDE_CACHE: HestonPDEDelta | None = None

def _get_pde(cal: dict[str, Any]) -> HestonPDEDelta:
    global _PDE_CACHE
    if _PDE_CACHE is None:
        hp = cal["heston_params"]
        _PDE_CACHE = HestonPDEDelta(
            kappa=hp["kappa"],
            theta=hp["theta"],
            sigma_v=hp["sigma_v"],
            rho=hp["rho"],
            V0=hp["V0"],
            K=K, T=T,
            S_max=400.0, V_max=1.0,
            n_S=200, n_V=80, n_t=400,
        )
    return _PDE_CACHE

# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def run_single_seed(seed: int) -> dict[str, Any]:
    """Run BS / Plugin / HestonPDE on seed `seed`'s rough Bergomi test set."""
    print(f"\n{'='*70}", flush=True)
    print(f"  SEED {seed}", flush=True)
    print(f"{'='*70}", flush=True)
    t_all = time.time()

    cal = _load_calibration()
    pde = _get_pde(cal)

    # --- 1. rough Bergomi paths ---
    print(f"  [1/4] Simulating {N_TOTAL:,} rough Bergomi paths (seed={seed})...",
          flush=True)
    t0 = time.time()
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=ETA, rho=RHO_RB, xi0=XI0,
    )
    S_all, V_all, t_grid = sim.simulate(n_paths=N_TOTAL, S0=S0, seed=seed)
    # Use the last N_TEST paths as test set (matches canonical convention)
    S_test = S_all[N_TRAIN + N_VAL :]
    V_test = V_all[N_TRAIN + N_VAL :]
    del S_all, V_all
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s  S_test={tuple(S_test.shape)}  "
          f"V_test={tuple(V_test.shape)}", flush=True)

    # --- 2. p0 (MC call price estimate) ---
    # Use training-split payoffs for p0, matching canonical baseline.
    # For simplicity (and because N_TRAIN paths consume memory), re-generate
    # just the training split for p0 computation.
    print(f"  [2/4] p0 from training split...", flush=True)
    t0 = time.time()
    S_train_only, _, _ = sim.simulate(n_paths=N_TRAIN, S0=S0, seed=seed + 100000)
    payoff_train = compute_payoff(S_train_only, K, "call")
    p0 = float(payoff_train.mean())
    del S_train_only, payoff_train
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s  p0 = {p0:.4f}", flush=True)

    # --- 3. Evaluate all 3 strategies ---
    payoff_test = compute_payoff(S_test, K, "call")

    strategies_out = {}

    # BS Delta
    print(f"  [3/4] BS Delta (sigma=sqrt(xi0)={SIGMA_BS:.4f})...", flush=True)
    t0 = time.time()
    bs = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
    deltas_bs = bs.hedge_paths(S_test)
    pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, COST_LAMBDA)
    metrics_bs = compute_all_metrics(pnl_bs)
    metrics_bs["turnover"] = _mean_turnover(deltas_bs)
    metrics_bs["first_delta_sum"] = _first_delta_sum(deltas_bs)
    strategies_out["bs"] = metrics_bs
    print(f"  Done in {time.time()-t0:.2f}s  ES_0.95 = {metrics_bs['es_95']:.4f}",
          flush=True)

    # PluginDelta (BS with realised variance)
    print(f"  [3/4] PluginDelta (BS with realised V_t)...", flush=True)
    t0 = time.time()
    plugin = PluginDelta(K=K, T=T)
    deltas_pl = plugin.hedge_paths(S_test, V_test)
    pnl_pl = compute_hedging_pnl(S_test, deltas_pl, payoff_test, p0, COST_LAMBDA)
    metrics_pl = compute_all_metrics(pnl_pl)
    metrics_pl["turnover"] = _mean_turnover(deltas_pl)
    metrics_pl["first_delta_sum"] = _first_delta_sum(deltas_pl)
    strategies_out["plugin"] = metrics_pl
    print(f"  Done in {time.time()-t0:.2f}s  ES_0.95 = {metrics_pl['es_95']:.4f}",
          flush=True)

    # HestonPDE Delta (true PDE)
    print(f"  [3/4] HestonPDE (kappa={pde.kappa:.3f}, sigma_v={pde.sigma_v:.3f})...",
          flush=True)
    t0 = time.time()
    # Use uniform t_grid for compatibility with the PDE's cached grid
    t_vec = torch.linspace(0.0, T, N_STEPS + 1, dtype=S_test.dtype, device=S_test.device)
    deltas_h = pde.hedge_paths(S_test, V_test, t_grid=t_vec)
    pnl_h = compute_hedging_pnl(S_test, deltas_h, payoff_test, p0, COST_LAMBDA)
    metrics_h = compute_all_metrics(pnl_h)
    metrics_h["turnover"] = _mean_turnover(deltas_h)
    metrics_h["first_delta_sum"] = _first_delta_sum(deltas_h)
    strategies_out["heston_pde"] = metrics_h
    print(f"  Done in {time.time()-t0:.2f}s  ES_0.95 = {metrics_h['es_95']:.4f}",
          flush=True)

    wall = time.time() - t_all
    print(f"  [4/4] Seed {seed} complete in {wall/60:.1f} min", flush=True)

    record = {
        "seed": seed,
        "n_test": int(S_test.shape[0]),
        "p0": p0,
        "wall_clock_s": wall,
        "bs": strategies_out["bs"],
        "plugin": strategies_out["plugin"],
        "heston_pde": strategies_out["heston_pde"],
    }

    # Free memory before next seed
    del S_test, V_test, deltas_bs, deltas_pl, deltas_h
    del pnl_bs, pnl_pl, pnl_h
    gc.collect()

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
    t_crit = 2.776 if n == 5 else 1.96
    return {
        "mean": mean, "std": std, "se": se,
        "ci95_lower": mean - t_crit * se,
        "ci95_upper": mean + t_crit * se,
        "min": float(arr.min()), "max": float(arr.max()),
        "all_values": [float(v) for v in arr],
    }

def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed metrics into mean/std/CI per strategy per metric."""
    agg: dict[str, dict[str, Any]] = {}
    for strat in ("bs", "plugin", "heston_pde"):
        agg[strat] = {}
        for metric in ("es_95", "es_99", "std_pnl", "var_95", "var_99",
                        "mean_pnl", "turnover", "first_delta_sum"):
            values = [per_seed[s][strat].get(metric)
                      for s in per_seed if strat in per_seed[s]
                      and per_seed[s][strat].get(metric) is not None]
            if values:
                agg[strat][metric] = _agg(values)
    return agg

# ---------------------------------------------------------------------------
# Reproducibility check
# ---------------------------------------------------------------------------

def run_reproducibility_check(original: dict[str, Any]) -> dict[str, Any]:
    """Re-run seed 6024 in a fresh subprocess and compare outputs."""
    print("\n" + "=" * 70, flush=True)
    print("  REPRODUCIBILITY — seed 6024 in fresh subprocess", flush=True)
    print("=" * 70, flush=True)

    out_path = OUT_DIR / "seed_6024_rerun.json"
    cmd = [
        sys.executable, "-u", "-m",
        "deep_hedging.experiments.heston_pde_evaluation",
        "--single-seed", "6024",
        "--output", str(out_path),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    print(f"  Subprocess finished in {wall/60:.1f} min  (exit={result.returncode})",
          flush=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1500:]}", flush=True)
        return {"error": f"subprocess exit {result.returncode}"}

    with open(out_path) as f:
        rerun = json.load(f)

    r = {"seed": 6024}
    all_match = True
    for strat in ("bs", "plugin", "heston_pde"):
        for metric in ("es_95", "first_delta_sum"):
            orig_val = original[strat][metric]
            rerun_val = rerun[strat][metric]
            matches = orig_val == rerun_val
            r[f"{strat}__{metric}"] = {
                "original": orig_val, "rerun": rerun_val, "match": matches,
            }
            if not matches:
                all_match = False

    r["all_match"] = all_match
    r["verdict"] = "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE"

    print(f"\n  Match results:")
    for strat in ("bs", "plugin", "heston_pde"):
        for metric in ("es_95", "first_delta_sum"):
            k = f"{strat}__{metric}"
            v = r[k]
            print(f"    {k}: orig={v['original']:.6f}  rerun={v['rerun']:.6f}  "
                  f"match={v['match']}", flush=True)
    print(f"\n  VERDICT: {r['verdict']}", flush=True)
    return r

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Heston PDE evaluation (Phase J)")
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.single_seed is not None:
        r = run_single_seed(args.single_seed)
        out_path = Path(args.output) if args.output else (
            OUT_DIR / f"seed_{args.single_seed}.json"
        )
        _save_json(r, out_path)
        print(f"\n  Wrote {out_path}", flush=True)
        return

    seeds_to_run = args.seeds_only if args.seeds_only else SEEDS

    print("=" * 70, flush=True)
    print(f"  Phase 3 — Heston PDE evaluation on 5 seeds", flush=True)
    print(f"  seeds: {seeds_to_run}", flush=True)
    print(f"  commit: {_git_commit_sha()}", flush=True)
    print("=" * 70, flush=True)

    cal = _load_calibration()
    print(f"\n  Using calibrated Heston parameters:")
    hp = cal["heston_params"]
    print(f"    kappa   = {hp['kappa']:.4f}", flush=True)
    print(f"    sigma_v = {hp['sigma_v']:.4f}", flush=True)
    print(f"    theta   = {hp['theta']:.6f}", flush=True)
    print(f"    rho     = {hp['rho']:.4f}", flush=True)
    print(f"    V_0     = {hp['V0']:.6f}", flush=True)

    per_seed: dict[str, dict[str, Any]] = {}
    total_t0 = time.time()
    for seed in seeds_to_run:
        try:
            r = run_single_seed(seed)
            per_seed[str(seed)] = r
            _save_json(r, OUT_DIR / f"seed_{seed}.json")
        except Exception as exc:
            print(f"\n  ERROR seed {seed}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            per_seed[str(seed)] = {"seed": seed, "error": str(exc)}
        # Save partial
        partial = {
            "meta": {
                "script": "deep_hedging/experiments/heston_pde_evaluation.py",
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "git_commit": _git_commit_sha(),
                "seeds_complete": list(per_seed),
                "calibration": hp,
            },
            "per_seed": per_seed,
        }
        _save_json(partial, OUT_DIR / "heston_pde_5seeds.json")

    total_wall = time.time() - total_t0
    print(f"\n  Total wall (5 seeds): {total_wall/60:.1f} min", flush=True)

    aggregated = aggregate(per_seed)

    repro = None
    if not args.skip_reproducibility and "6024" in per_seed \
            and "error" not in per_seed["6024"]:
        repro = run_reproducibility_check(per_seed["6024"])

    final = {
        "meta": {
            "script": "deep_hedging/experiments/heston_pde_evaluation.py",
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit_sha(),
            "seeds": seeds_to_run,
            "total_wall_clock_s": total_wall,
            "calibration": cal["heston_params"],
            "calibration_source": "results/heston_pde/calibration_data.json",
            "rbergomi_params": {
                "H": H, "eta": ETA, "rho": RHO_RB, "xi0": XI0,
                "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
            },
            "dataset": {
                "n_test": N_TEST, "cost_lambda": COST_LAMBDA, "alpha": ALPHA,
            },
        },
        "per_seed": per_seed,
        "aggregated": aggregated,
        "reproducibility_check": repro,
    }
    _save_json(final, OUT_DIR / "heston_pde_5seeds.json")

    # Print headline
    print("\n" + "=" * 70, flush=True)
    print("  HEADLINE — ES_0.95 aggregates (mean ± std across 5 seeds)",
          flush=True)
    for strat in ("bs", "plugin", "heston_pde"):
        if strat in aggregated and "es_95" in aggregated[strat]:
            m = aggregated[strat]["es_95"]
            print(f"    {strat:12s}: {m['mean']:.4f} ± {m['std']:.4f}  "
                  f"(CI [{m['ci95_lower']:.4f}, {m['ci95_upper']:.4f}])",
                  flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
