#!/usr/bin/env python
"""
Unified master test-set evaluation for Section 6.3.

Generates a single frozen rough Bergomi test set (50 000 paths,
seed=2024) and evaluates four strategies on it:

  1. BS Delta
  2. Plug-in Delta (observes realised variance)
  3. DH full-budget (trained from scratch on rBergomi)
  4. DH GBM-pretrained (loaded from transfer-learning checkpoint)

Emits figures/unified_baseline_results.json — the single source of
truth for every number cited in Observations 6.1, 6.3, and 6.7.

Run:
    python -u -m deep_hedging.experiments.run_unified_baseline
    python -u -m deep_hedging.experiments.run_unified_baseline --skip-train
    python -u -m deep_hedging.experiments.run_unified_baseline --force-regen-test-set
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta, PluginDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    hedge_paths_deep,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics

# ---------------------------------------------------------------------------
# Frozen constants — DO NOT parameterise
# ---------------------------------------------------------------------------

# Rough Bergomi dissertation calibration
H       = 0.07
ETA     = 1.9
RHO     = -0.7
XI0     = 0.235 ** 2          # 0.055225
S0      = 100.0
K       = 100.0
T       = 1.0
N_STEPS = 100

# Master test set
N_TEST           = 50_000
MASTER_TEST_SEED = 2024

# Training set (independent from test set: TRAIN_SEED != MASTER_TEST_SEED)
N_TRAIN    = 80_000
N_VAL      = 10_000
TRAIN_SEED = 42

# Training protocol
DH_EPOCHS     = 200
DH_PATIENCE   = 30
DH_LR         = 1e-3
DH_BATCH_SIZE = 2048
DH_ALPHA      = 0.95

# Cost levels
COST_LAMBDAS = [0.0, 0.001]

# Strategy names (exact strings for downstream consumers)
STRAT_BS       = "BS Delta"
STRAT_PLUGIN   = "Plug-in Delta"
STRAT_DH_FULL  = "DH full-budget"
STRAT_DH_GBM   = "DH GBM-pretrained"

# Output paths
FIGURE_DIR           = Path(__file__).resolve().parents[2] / "figures"
MASTER_TEST_SET_PATH = FIGURE_DIR / "unified_master_test_set.pt"
DH_CHECKPOINT_PATH   = FIGURE_DIR / "unified_dh_rbergomi_hedger.pt"
GBM_PRETRAINED_PATH  = FIGURE_DIR / "gbm_pretrained_hedger.pt"
OUTPUT_JSON          = FIGURE_DIR / "unified_baseline_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_turnover(deltas: Tensor) -> float:
    """Mean over paths of sum_k |delta_k - delta_{k-1}|, delta_{-1} = 0."""
    batch = deltas.shape[0]
    dtype, device = deltas.dtype, deltas.device
    delta_prev = torch.cat(
        [torch.zeros(batch, 1, dtype=dtype, device=device), deltas[:, :-1]],
        dim=1,
    )
    return float((deltas - delta_prev).abs().sum(dim=1).mean())


def _git_commit_sha() -> str:
    """Return HEAD SHA (7+ chars), suffixed with -dirty if working tree is unclean."""
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


def _make_simulator() -> DifferentiableRoughBergomi:
    return DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0,
    )


# ---------------------------------------------------------------------------
# Master test set
# ---------------------------------------------------------------------------

def load_or_generate_master_test_set(
    *, force: bool = False,
) -> tuple[Tensor, Tensor]:
    """Load cached master test set or generate and cache it.

    Returns (S, V) both of shape (N_TEST, N_STEPS+1).
    """
    path = MASTER_TEST_SET_PATH

    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached["seed"] == MASTER_TEST_SEED
                and cached["n_paths"] == N_TEST
                and cached["H"] == H
                and cached["eta"] == ETA):
            print(f"  Loaded cached master test set "
                  f"(seed={MASTER_TEST_SEED}, n={N_TEST})")
            return cached["S"], cached["V"]
        print("  WARNING: cached test set params mismatch — regenerating.")

    print(f"  Generating master test set: {N_TEST} paths, seed={MASTER_TEST_SEED}...")
    sim = _make_simulator()
    S, V, _ = sim.simulate(n_paths=N_TEST, S0=S0, seed=MASTER_TEST_SEED)
    payload = {
        "S": S, "V": V,
        "seed": MASTER_TEST_SEED, "n_paths": N_TEST,
        "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
        "n_steps": N_STEPS, "T": T, "S0": S0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  Cached to {path}")
    return S, V


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_full_budget_model(
    S_test: Tensor, V_test: Tensor,
) -> tuple[DeepHedgerFNN, float]:
    """Train a DH from scratch on fresh rBergomi paths.

    Returns (model, p0) where p0 is the MC premium estimate on training paths.
    """
    print(f"\n  Generating training paths: {N_TRAIN + N_VAL} paths, seed={TRAIN_SEED}...")
    sim = _make_simulator()
    S_all, V_all, _ = sim.simulate(
        n_paths=N_TRAIN + N_VAL, S0=S0, seed=TRAIN_SEED,
    )
    S_train = S_all[:N_TRAIN]
    S_val   = S_all[N_TRAIN:]

    payoff_train = compute_payoff(S_train, K, "call")
    p0 = float(payoff_train.mean())
    print(f"  MC option price p0 = {p0:.4f}")

    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)

    print(f"\n  Training DH full-budget ({DH_EPOCHS} epochs, patience={DH_PATIENCE})...")
    t0 = time.perf_counter()
    train_deep_hedger(
        model, S_train, S_val,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=0.0, alpha=DH_ALPHA,
        lr=DH_LR, batch_size=DH_BATCH_SIZE,
        epochs=DH_EPOCHS, patience=DH_PATIENCE,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s")

    DH_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DH_CHECKPOINT_PATH)
    print(f"  Checkpoint saved to {DH_CHECKPOINT_PATH}")

    return model, p0


def load_full_budget_model() -> DeepHedgerFNN:
    """Load a previously trained DH full-budget checkpoint."""
    if not DH_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"--skip-train requires {DH_CHECKPOINT_PATH} to exist. "
            "Run without --skip-train first."
        )
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    state = torch.load(DH_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded DH full-budget from {DH_CHECKPOINT_PATH}")
    return model


def load_gbm_pretrained_model(
    *, allow_missing: bool = False,
) -> DeepHedgerFNN | None:
    """Load the GBM-pretrained checkpoint from transfer-learning."""
    if not GBM_PRETRAINED_PATH.exists():
        msg = (
            f"GBM-pretrained checkpoint not found at {GBM_PRETRAINED_PATH}. "
            "Run the transfer-learning experiment first."
        )
        if allow_missing:
            print(f"  WARNING: {msg}")
            return None
        raise FileNotFoundError(msg)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    state = torch.load(GBM_PRETRAINED_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded DH GBM-pretrained from {GBM_PRETRAINED_PATH}")
    return model


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    name: str,
    deltas: Tensor,
    S_test: Tensor,
    p0: float,
    cost_lambda: float,
) -> dict[str, Any]:
    """Compute metrics and turnover for a strategy's deltas."""
    payoff = compute_payoff(S_test, K, "call")
    pnl = compute_hedging_pnl(S_test, deltas, payoff, p0, cost_lambda)
    metrics = compute_all_metrics(pnl)
    turnover = _mean_turnover(deltas)
    return {"metrics": metrics, "mean_turnover": turnover}


def run_all_strategies(
    S_test: Tensor,
    V_test: Tensor,
    p0: float,
    dh_full: DeepHedgerFNN,
    dh_gbm: DeepHedgerFNN | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all strategies at all cost levels. Returns nested dict."""
    # Pre-compute deltas (cost-independent for analytical hedgers and
    # our DH models which were trained at lambda=0)
    print("\n  Computing deltas...")
    sigma_flat = math.sqrt(XI0)
    bs_hedger = BlackScholesDelta(sigma=sigma_flat, K=K, T=T)
    deltas_bs = bs_hedger.hedge_paths(S_test)

    plugin_hedger = PluginDelta(K=K, T=T)
    deltas_plugin = plugin_hedger.hedge_paths(S_test, V_test)

    dh_full.eval()
    with torch.no_grad():
        deltas_dh_full = hedge_paths_deep(dh_full, S_test, T, S0)

    deltas_dh_gbm = None
    if dh_gbm is not None:
        dh_gbm.eval()
        with torch.no_grad():
            deltas_dh_gbm = hedge_paths_deep(dh_gbm, S_test, T, S0)

    results: dict[str, dict[str, Any]] = {}
    for lam in COST_LAMBDAS:
        print(f"\n  Evaluating at lambda={lam}...")
        row: dict[str, Any] = {}
        row[STRAT_BS] = evaluate_strategy(
            STRAT_BS, deltas_bs, S_test, p0, lam)
        row[STRAT_PLUGIN] = evaluate_strategy(
            STRAT_PLUGIN, deltas_plugin, S_test, p0, lam)
        row[STRAT_DH_FULL] = evaluate_strategy(
            STRAT_DH_FULL, deltas_dh_full, S_test, p0, lam)
        if deltas_dh_gbm is not None:
            row[STRAT_DH_GBM] = evaluate_strategy(
                STRAT_DH_GBM, deltas_dh_gbm, S_test, p0, lam)
        else:
            row[STRAT_DH_GBM] = {"unavailable": True}
        results[str(lam)] = row
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output_json(
    results: dict, p0: float,
) -> dict:
    """Assemble the full output dict with metadata."""
    return {
        "meta": {
            "master_test_seed": MASTER_TEST_SEED,
            "n_test": N_TEST,
            "train_seed": TRAIN_SEED,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "rbergomi": {
                "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
                "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
            },
            "dh_protocol": {
                "input_dim": 4, "hidden_dim": 128, "n_res_blocks": 2,
                "epochs": DH_EPOCHS, "patience": DH_PATIENCE,
                "lr": DH_LR, "batch_size": DH_BATCH_SIZE,
                "alpha": DH_ALPHA,
                "loss": "smooth_cvar_rockafellar_uryasev",
            },
            "p0_mc_estimate": p0,
            "train_set_path": str(MASTER_TEST_SET_PATH),
            "checkpoints": {
                "dh_full_budget":    str(DH_CHECKPOINT_PATH),
                "dh_gbm_pretrained": str(GBM_PRETRAINED_PATH),
            },
            "source_script": "deep_hedging/experiments/run_unified_baseline.py",
            "source_commit": _git_commit_sha(),
        },
        "results": results,
    }


def print_summary_table(results: dict) -> None:
    """Print plain-text summary for human sanity-checking."""
    strategies = [STRAT_BS, STRAT_PLUGIN, STRAT_DH_FULL, STRAT_DH_GBM]
    header = (f"{'Strategy':25s} {'Mean':>8s} {'Std':>8s} "
              f"{'VaR95':>8s} {'ES95':>8s} {'ES99':>8s} {'Turn':>8s}")

    for cost_key in sorted(results.keys(), key=float):
        lam = float(cost_key)
        tag = "Frictionless" if lam == 0 else f"lambda={lam}"
        print(f"\n  [{tag}]")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for name in strategies:
            entry = results[cost_key][name]
            if entry.get("unavailable"):
                print(f"  {name:25s}  (checkpoint missing)")
                continue
            m = entry["metrics"]
            t = entry["mean_turnover"]
            print(f"  {name:25s} {m['mean_pnl']:8.3f} {m['std_pnl']:8.3f} "
                  f"{m['var_95']:8.3f} {m['es_95']:8.3f} "
                  f"{m['es_99']:8.3f} {t:8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    *,
    skip_train: bool = False,
    force_regen: bool = False,
    allow_missing_pretrained: bool = False,
) -> None:
    print("=" * 65)
    print("  Unified Section 6.3 Baseline")
    print("=" * 65)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Master test set
    print("\n--- Master test set ---")
    S_test, V_test = load_or_generate_master_test_set(force=force_regen)

    # 2. DH full-budget
    print("\n--- DH full-budget ---")
    if skip_train:
        dh_full = load_full_budget_model()
        # Still need p0: compute from fresh training paths
        sim = _make_simulator()
        S_train_for_p0, _, _ = sim.simulate(
            n_paths=N_TRAIN, S0=S0, seed=TRAIN_SEED,
        )
        p0 = float(compute_payoff(S_train_for_p0, K, "call").mean())
        print(f"  MC option price p0 = {p0:.4f}")
        del S_train_for_p0
    else:
        dh_full, p0 = train_full_budget_model(S_test, V_test)

    # 3. DH GBM-pretrained
    print("\n--- DH GBM-pretrained ---")
    dh_gbm = load_gbm_pretrained_model(allow_missing=allow_missing_pretrained)

    # 4. Evaluate all strategies
    print("\n--- Evaluation ---")
    results = run_all_strategies(S_test, V_test, p0, dh_full, dh_gbm)

    # 5. Build and save JSON
    output = build_output_json(results, p0)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_JSON}")

    # 6. Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print_summary_table(results)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Section 6.3 baseline evaluation.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Reuse existing DH checkpoint; error if missing.",
    )
    parser.add_argument(
        "--force-regen-test-set", action="store_true",
        help="Overwrite cached master test set.",
    )
    args = parser.parse_args()
    main(skip_train=args.skip_train, force_regen=args.force_regen_test_set)


if __name__ == "__main__":
    cli()
