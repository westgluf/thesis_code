#!/usr/bin/env python
"""
Lean H4 sweep: three fair points at full Stage 1 budget.

Runs the flat / sig-3 / sig-full ablation at H in {0.02, 0.25},
then combines with the already-computed H=0.05 results from
Stage 1 to produce a three-point trend check for the revised
Proposition 6.9.

Budget is FIXED at Stage 1 values:
    n_train    = 80_000
    n_val      = 20_000
    n_test     = 50_000
    epochs     = 200
    patience   = 30
    batch_size = 2048
    lr         = 1e-3

This budget is NOT configurable via argparse or env var. The whole
point of this script is that every data point uses the same
resources as Stage 1.

Run:
    python -u -m deep_hedging.experiments.run_lean_h4_sweep
"""
from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger, evaluate_deep_hedger
from deep_hedging.hedging.signature_hedger import SignatureDeepHedger
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics, expected_shortfall


# ─── FIXED BUDGET ─────────────────────────────────────────────────────
# Do NOT change any of these values. They must match Stage 1 exactly.
FIXED_BUDGET: dict[str, Any] = {
    "n_train": 80_000,
    "n_val": 20_000,
    "n_test": 50_000,
    "epochs": 200,
    "patience": 30,
    "batch_size": 2048,
    "lr": 1e-3,
    "hidden_dim": 128,
    "n_res_blocks": 2,
    "weight_decay": 1e-5,
}

# Model defaults (match Stage 1 exactly)
ETA = 1.9
RHO = -0.7
XI0 = 0.235 ** 2
SIGMA = 0.235
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100

# The two new H values to run (H=0.05 comes from Stage 1 cache)
NEW_H_VALUES: list[float] = [0.02, 0.25]

# Output directory
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"
STAGE_1_JSON = OUT_DIR / "signature_ablation_stage_1.json"


# =======================================================================
# Path generation and BS delta
# =======================================================================

def generate_paths(
    H: float, seed: int, device: torch.device,
    budget: dict | None = None,
) -> dict:
    """Generate rBergomi paths and split into train/val/test.

    Parameters
    ----------
    H : float
        Hurst parameter.
    seed : int
        Simulation seed.
    device : torch.device
        Torch device (unused for simulation which runs in float64 on CPU).
    budget : dict, optional
        Override budget (testing only).

    Returns
    -------
    dict with keys: S_train, S_val, S_test, p0.
    """
    b = budget if budget is not None else FIXED_BUDGET
    total = b["n_train"] + b["n_val"] + b["n_test"]
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0,
    )
    S_all, _, _ = sim.simulate(n_paths=total, S0=S0, seed=seed)

    n1 = b["n_train"]
    n2 = n1 + b["n_val"]
    S_train = S_all[:n1]
    S_val = S_all[n1:n2]
    S_test = S_all[n2:]
    del S_all
    gc.collect()

    payoff_train = compute_payoff(S_train, K, "call")
    p0 = float(payoff_train.mean())

    return {
        "S_train": S_train,
        "S_val": S_val,
        "S_test": S_test,
        "p0": p0,
    }


def run_bs_delta(
    S_test: torch.Tensor, p0: float, sigma: float = SIGMA,
) -> dict:
    """Evaluate BS delta on test set."""
    bs = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas = bs.hedge_paths(S_test)
    payoff = compute_payoff(S_test, K, "call")
    pnl = compute_hedging_pnl(S_test, deltas, payoff, p0, 0.0)
    return {"metrics": compute_all_metrics(pnl), "pnl": pnl}


# =======================================================================
# Training one hedger
# =======================================================================

def train_and_evaluate_hedger(
    hedger: torch.nn.Module,
    data: dict,
    tag: str,
    p0: float,
    budget: dict | None = None,
) -> dict:
    """Train a hedger with the fixed full budget and evaluate on test."""
    b = budget if budget is not None else FIXED_BUDGET

    t0 = time.time()
    history = train_deep_hedger(
        hedger, data["S_train"], data["S_val"],
        K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
        alpha=0.95,
        lr=b["lr"],
        batch_size=b["batch_size"],
        epochs=b["epochs"],
        patience=b["patience"],
        verbose=False,
    )
    elapsed = time.time() - t0

    pnl = evaluate_deep_hedger(
        hedger, data["S_test"], K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
    )
    metrics = compute_all_metrics(pnl)

    return {
        "metrics": metrics,
        "pnl": pnl,
        "history": history,
        "time_s": elapsed,
        "tag": tag,
    }


# =======================================================================
# One H value — all four strategies
# =======================================================================

def run_single_H(
    H: float, seed: int, device: torch.device,
    budget: dict | None = None,
) -> dict:
    """Full-budget run at one H value with BS + flat + sig-3 + sig-full.

    All three deep hedgers share the SAME NN init seed so the comparison
    is controlled for random initialisation.
    """
    b = budget if budget is not None else FIXED_BUDGET

    print(f"\n  Generating paths at H={H} (seed={seed}) ...", flush=True)
    t0 = time.time()
    data = generate_paths(H=H, seed=seed, device=device, budget=b)
    print(f"    {b['n_train']} train + {b['n_val']} val + {b['n_test']} test "
          f"({time.time()-t0:.1f}s), p0={data['p0']:.4f}", flush=True)

    # BS delta baseline
    print("  BS delta ...", flush=True)
    bs_result = run_bs_delta(data["S_test"], p0=data["p0"])
    es_bs = bs_result["metrics"]["es_95"]
    print(f"    ES_95 = {es_bs:.3f}", flush=True)

    nn_init_seed = seed + 1000  # Same across feature sets within one H

    results: dict[str, Any] = {
        "H": H,
        "seed": seed,
        "nn_init_seed": nn_init_seed,
        "p0": data["p0"],
        "bs_metrics": bs_result["metrics"],
        "bs_pnl": bs_result["pnl"],
        "budget": dict(b),
    }

    # Flat hedger (DeepHedgerFNN, the Prompt 3 baseline)
    print("  Flat (DeepHedgerFNN, 4d) ...", flush=True)
    torch.manual_seed(nn_init_seed)
    np.random.seed(nn_init_seed)
    flat_hedger = DeepHedgerFNN(
        input_dim=4, hidden_dim=b["hidden_dim"], n_res_blocks=b["n_res_blocks"],
    )
    flat_out = train_and_evaluate_hedger(flat_hedger, data, "flat", data["p0"], budget=b)
    print(f"    trained in {flat_out['time_s']:.0f}s  |  "
          f"ES_95 = {flat_out['metrics']['es_95']:.3f}  "
          f"(best epoch {flat_out['history']['best_epoch']})", flush=True)
    results["flat_metrics"] = flat_out["metrics"]
    results["flat_pnl"] = flat_out["pnl"]
    results["flat_history"] = flat_out["history"]
    results["flat_time_s"] = flat_out["time_s"]
    del flat_hedger
    gc.collect()

    # Sig-3 hedger
    print("  Sig-3 (SignatureDeepHedger, 7d) ...", flush=True)
    torch.manual_seed(nn_init_seed)
    np.random.seed(nn_init_seed)
    sig3_hedger = SignatureDeepHedger(
        feature_set="sig-3",
        hidden_dim=b["hidden_dim"], n_res_blocks=b["n_res_blocks"],
        xi0=XI0, eta_ref=ETA, T=T,
    )
    sig3_out = train_and_evaluate_hedger(sig3_hedger, data, "sig-3", data["p0"], budget=b)
    print(f"    trained in {sig3_out['time_s']:.0f}s  |  "
          f"ES_95 = {sig3_out['metrics']['es_95']:.3f}  "
          f"(best epoch {sig3_out['history']['best_epoch']})", flush=True)
    results["sig3_metrics"] = sig3_out["metrics"]
    results["sig3_pnl"] = sig3_out["pnl"]
    results["sig3_history"] = sig3_out["history"]
    results["sig3_time_s"] = sig3_out["time_s"]
    del sig3_hedger
    gc.collect()

    # Sig-full hedger
    print("  Sig-full (SignatureDeepHedger, 12d) ...", flush=True)
    torch.manual_seed(nn_init_seed)
    np.random.seed(nn_init_seed)
    sigfull_hedger = SignatureDeepHedger(
        feature_set="sig-full",
        hidden_dim=b["hidden_dim"], n_res_blocks=b["n_res_blocks"],
        xi0=XI0, eta_ref=ETA, T=T,
    )
    sigfull_out = train_and_evaluate_hedger(sigfull_hedger, data, "sig-full", data["p0"], budget=b)
    print(f"    trained in {sigfull_out['time_s']:.0f}s  |  "
          f"ES_95 = {sigfull_out['metrics']['es_95']:.3f}  "
          f"(best epoch {sigfull_out['history']['best_epoch']})", flush=True)
    results["sigfull_metrics"] = sigfull_out["metrics"]
    results["sigfull_pnl"] = sigfull_out["pnl"]
    results["sigfull_history"] = sigfull_out["history"]
    results["sigfull_time_s"] = sigfull_out["time_s"]
    del sigfull_hedger
    gc.collect()

    # Derived Gamma values
    results["gamma_flat"] = es_bs - results["flat_metrics"]["es_95"]
    results["gamma_sig3"] = es_bs - results["sig3_metrics"]["es_95"]
    results["gamma_sigfull"] = es_bs - results["sigfull_metrics"]["es_95"]
    results["roughness_adv"] = results["gamma_sigfull"] - results["gamma_flat"]

    # Drop training data to free memory
    del data
    gc.collect()

    return results


# =======================================================================
# Persistence
# =======================================================================

def _strip_tensors(obj: Any) -> Any:
    """Recursively strip torch tensors and nn.Modules from a dict for JSON."""
    import torch.nn as nn
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, nn.Module):
        return None
    if isinstance(obj, dict):
        return {k: _strip_tensors(v) for k, v in obj.items()
                if _strip_tensors(v) is not None}
    if isinstance(obj, (list, tuple)):
        return [_strip_tensors(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    try:
        return float(obj)
    except Exception:
        return None


def save_single_H_result(result: dict, H: float, out_dir: Path) -> None:
    """Save per-H scalar metrics and PnL tensors immediately (crash safety)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scalar JSON
    scalar = _strip_tensors(result)
    json_path = out_dir / f"lean_h4_H{H:.2f}.json"
    with open(json_path, "w") as f:
        json.dump(scalar, f, indent=2)
    print(f"    Saved {json_path.name}", flush=True)

    # PnL tensors (for tail figures)
    pnl_dict = {
        "bs_pnl": result["bs_pnl"].detach().float().cpu(),
        "flat_pnl": result["flat_pnl"].detach().float().cpu(),
        "sig3_pnl": result["sig3_pnl"].detach().float().cpu(),
        "sigfull_pnl": result["sigfull_pnl"].detach().float().cpu(),
    }
    pt_path = out_dir / f"lean_h4_H{H:.2f}_pnl.pt"
    torch.save(pnl_dict, pt_path)
    print(f"    Saved {pt_path.name}", flush=True)


# =======================================================================
# Stage 1 loader — adapts to our schema
# =======================================================================

def load_stage_1_results(stage1_path: Path) -> dict:
    """Load the H=0.05 Stage 1 results and reshape to match run_single_H output.

    Stage 1 JSON schema::
        { 'H': 0.05, 'p0': float,
          'bs':      {'metrics': {...}},
          'flat':    {'metrics': {...}, 'history': {...}, 'feature_set': 'flat'},
          'sig3':    {'metrics': {...}, ...},
          'sigfull': {'metrics': {...}, ...},
          'gamma_flat': float, 'gamma_sig3': float, 'gamma_sigfull': float,
          ... }

    Target schema (matches run_single_H)::
        { 'H': 0.05, 'p0': float,
          'bs_metrics': {...}, 'flat_metrics': {...}, 'sig3_metrics': {...},
          'sigfull_metrics': {...},
          'gamma_flat': float, ..., 'roughness_adv': float,
          'flat_history': {...}, 'sig3_history': {...}, 'sigfull_history': {...} }
    """
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 results not found at {stage1_path}")

    with open(stage1_path) as f:
        s1 = json.load(f)

    result: dict[str, Any] = {
        "H": s1["H"],
        "p0": s1["p0"],
        "seed": 2024,  # Stage 1 default
        "nn_init_seed": 2024 + 1000,
        "bs_metrics": s1["bs"]["metrics"],
        "flat_metrics": s1["flat"]["metrics"],
        "sig3_metrics": s1["sig3"]["metrics"],
        "sigfull_metrics": s1["sigfull"]["metrics"],
        "flat_history": s1["flat"].get("history", {}),
        "sig3_history": s1["sig3"].get("history", {}),
        "sigfull_history": s1["sigfull"].get("history", {}),
        "gamma_flat": s1["gamma_flat"],
        "gamma_sig3": s1["gamma_sig3"],
        "gamma_sigfull": s1["gamma_sigfull"],
        "roughness_adv": s1["gamma_sigfull"] - s1["gamma_flat"],
        "source": "stage_1",
    }

    # Carry through training times if present
    if "training_times_s" in s1:
        tt = s1["training_times_s"]
        result["flat_time_s"] = tt.get("flat", 0.0)
        result["sig3_time_s"] = tt.get("sig-3", 0.0)
        result["sigfull_time_s"] = tt.get("sig-full", 0.0)

    return result


# =======================================================================
# Trend verdict
# =======================================================================

def compute_trend_verdict(results_by_H: dict[float, dict]) -> dict:
    """Analyse the three (H, Gamma) points and return a verdict dict."""
    H_sorted = sorted(results_by_H.keys())
    rough_advs = [results_by_H[h]["roughness_adv"] for h in H_sorted]

    max_adv = max(rough_advs)
    max_H = H_sorted[rough_advs.index(max_adv)]
    any_positive = any(v > 0.2 for v in rough_advs)

    if any_positive and max_H <= 0.1:
        verdict = "weak_h4_support"
        message = (
            "weak H4 support - roughness advantage appears at small H\n"
            f"           max adv = {max_adv:+.3f} at H={max_H}\n"
            "           RECOMMEND: extend the sweep to confirm the trend"
        )
    else:
        verdict = "h4_refuted"
        message = (
            "H4 refuted at all three points\n"
            f"           max adv = {max_adv:+.3f}  (not meaningfully positive)\n"
            "           RECOMMEND: proceed to Scenario C narrative - flat features suffice"
        )

    return {
        "verdict": verdict,
        "message": message,
        "max_adv": max_adv,
        "max_adv_H": max_H,
        "roughness_adv_by_H": dict(zip(H_sorted, rough_advs)),
    }


# =======================================================================
# Main
# =======================================================================

def main() -> None:
    from deep_hedging.utils.config import get_device
    device = get_device()

    print("=" * 65, flush=True)
    print("  LEAN H4 SWEEP — Full Stage 1 Budget", flush=True)
    print("=" * 65, flush=True)
    print(f"  Fixed budget: {FIXED_BUDGET}", flush=True)
    print(f"  New H values: {NEW_H_VALUES}", flush=True)
    print(f"  Existing H=0.05 from: {STAGE_1_JSON.name}", flush=True)

    all_results: dict[float, dict] = {}

    for i, H in enumerate(NEW_H_VALUES):
        print(f"\n{'=' * 65}", flush=True)
        print(f"  LEAN H4 SWEEP  —  H = {H}  ({i+1}/{len(NEW_H_VALUES)})", flush=True)
        print(f"{'=' * 65}", flush=True)

        t0 = time.time()
        seed = 2024 + int(H * 1000)  # H=0.02 -> 2044, H=0.25 -> 2274
        result = run_single_H(H=H, seed=seed, device=device)
        elapsed = time.time() - t0

        print(f"\n  H={H} completed in {elapsed/60:.1f} min", flush=True)
        print(f"  ES_95 BS:       {result['bs_metrics']['es_95']:.3f}", flush=True)
        print(f"  ES_95 flat:     {result['flat_metrics']['es_95']:.3f}", flush=True)
        print(f"  ES_95 sig-3:    {result['sig3_metrics']['es_95']:.3f}", flush=True)
        print(f"  ES_95 sig-full: {result['sigfull_metrics']['es_95']:.3f}", flush=True)
        print(f"  Gamma_flat     = {result['gamma_flat']:+.3f}", flush=True)
        print(f"  Gamma_sig-3    = {result['gamma_sig3']:+.3f}", flush=True)
        print(f"  Gamma_sig-full = {result['gamma_sigfull']:+.3f}", flush=True)
        print(f"  Roughness adv  = {result['roughness_adv']:+.3f}", flush=True)

        save_single_H_result(result, H, OUT_DIR)
        all_results[H] = result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Load Stage 1 H=0.05
    print("\n  Loading Stage 1 H=0.05 ...", flush=True)
    stage1 = load_stage_1_results(STAGE_1_JSON)
    all_results[0.05] = stage1
    print(f"    Gamma_flat     = {stage1['gamma_flat']:+.3f}", flush=True)
    print(f"    Gamma_sig-full = {stage1['gamma_sigfull']:+.3f}", flush=True)
    print(f"    Roughness adv  = {stage1['roughness_adv']:+.3f}", flush=True)

    # Combined summary JSON (scalars only, string keys for JSON compat)
    summary: dict[str, Any] = {
        f"{H:.2f}": _strip_tensors(all_results[H])
        for H in sorted(all_results.keys())
    }
    summary_path = OUT_DIR / "lean_h4_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {summary_path}", flush=True)

    # Final summary table
    print("\n" + "=" * 65, flush=True)
    print("  LEAN H4 SWEEP COMPLETE", flush=True)
    print("=" * 65, flush=True)
    print(f"  {'H':>6}  {'Gamma_flat':>12}  {'Gamma_sig3':>12}  "
          f"{'Gamma_sigfull':>14}  {'rough_adv':>10}", flush=True)
    for H in sorted(all_results.keys()):
        r = all_results[H]
        print(
            f"  {H:>6.2f}  {r['gamma_flat']:>+12.3f}  {r['gamma_sig3']:>+12.3f}  "
            f"{r['gamma_sigfull']:>+14.3f}  {r['roughness_adv']:>+10.3f}",
            flush=True,
        )

    # Trend verdict
    verdict = compute_trend_verdict(all_results)
    print("\n--- Trend check ---", flush=True)
    for H, adv in verdict["roughness_adv_by_H"].items():
        print(f"  H={H:.2f}:  roughness_adv = {adv:+.3f}", flush=True)
    print(f"\n  VERDICT: {verdict['message']}", flush=True)


if __name__ == "__main__":
    main()
