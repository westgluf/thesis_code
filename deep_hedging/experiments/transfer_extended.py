#!/usr/bin/env python
"""
Phase L — Transfer Learning Comprehensive Extension (Block 4).

Five sub-experiments testing whether the deep hedging policy is
dynamics-agnostic across source models, pretraining budgets, fine-tuning
budgets, target families, and rough-Bergomi calibrations.

Sub-experiments:
  L.1  Multi-source zero-shot       (3 sources × 5 seeds)
  L.2  Pretraining budget sweep     (3 sources × 6 budgets × 3 seeds)
  L.3  Extended fine-tuning curve   (11 n_ft × 3 seeds × 2 regimes)
  L.4  Reverse transfer             (3 seeds × 2 targets)
  L.5  Cross-calibration transfer   (3 H values × 3 seeds)

Run from repo root:
    python -u -m deep_hedging.experiments.transfer_extended --setup
    python -u -m deep_hedging.experiments.transfer_extended --L1
    python -u -m deep_hedging.experiments.transfer_extended --L2
    python -u -m deep_hedging.experiments.transfer_extended --L3
    python -u -m deep_hedging.experiments.transfer_extended --L4
    python -u -m deep_hedging.experiments.transfer_extended --L5
    python -u -m deep_hedging.experiments.transfer_extended --synthesise
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

from deep_hedging.core.gbm import GBM
from deep_hedging.core.heston import Heston
from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    evaluate_deep_hedger,
    hedge_paths_deep,
    train_deep_hedger,
)
from deep_hedging.hedging.heston_pde_delta import HestonPDEDelta
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import compute_all_metrics, expected_shortfall

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "transfer_v2"
FIG_DIR = REPO_ROOT / "figures" / "transfer_v2"

SHARED_TEST_PATH = OUT_DIR / "shared_test_set.pt"
GBM_PRETRAINED_PATH = REPO_ROOT / "figures" / "gbm_pretrained_hedger.pt"
CANONICAL_DH_PATH = REPO_ROOT / "figures" / "unified_dh_rbergomi_hedger.pt"
HESTON_CAL_PATH = REPO_ROOT / "results" / "heston_pde" / "calibration_data.json"

# --------------------------------------------------------------------------
# Canonical parameters
# --------------------------------------------------------------------------

H_RB = 0.07
ETA_RB = 1.9
RHO_RB = -0.7
XI0 = 0.235 ** 2  # 0.055225
SIGMA_BS = math.sqrt(XI0)  # 0.235
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100

# Test set (shared canonical rough Bergomi)
N_TEST = 50_000
TEST_SEED = 2024

# Reference values (from Phase B)
CANONICAL_BS_ES95 = 11.6307
CANONICAL_DH_ES95 = 10.4463
CANONICAL_DH_5SEED_MEAN = 10.4442
CANONICAL_DH_5SEED_STD = 0.0748
CANONICAL_BS_5SEED_MEAN = 11.5921
CANONICAL_BS_5SEED_STD = 0.0316

# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _strip_for_json(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() < 20 else None
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

def _first_weight_sum(model: torch.nn.Module) -> float:
    for p in model.parameters():
        if p.ndim >= 2:
            return float(p.detach().flatten().sum().cpu())
    return 0.0

def _mean_turnover(deltas: torch.Tensor) -> float:
    batch = deltas.shape[0]
    dtype, device = deltas.dtype, deltas.device
    delta_prev = torch.cat(
        [torch.zeros(batch, 1, dtype=dtype, device=device), deltas[:, :-1]],
        dim=1,
    )
    return float((deltas - delta_prev).abs().sum(dim=1).mean())

def _agg(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    t_crit = 2.776 if n == 5 else (4.303 if n == 3 else 1.96)
    return {
        "mean": mean, "std": std, "se": se,
        "ci95_lower": mean - t_crit * se,
        "ci95_upper": mean + t_crit * se,
        "min": float(arr.min()), "max": float(arr.max()),
        "all_values": [float(v) for v in arr], "n": n,
    }

# --------------------------------------------------------------------------
# Source simulator factory
# --------------------------------------------------------------------------

def build_source_simulator(source: str) -> Any:
    """Return a simulator with .simulate(n_paths, S0, seed) interface."""
    if source == "gbm":
        return GBM(n_steps=N_STEPS, T=T, sigma=SIGMA_BS)
    elif source == "heston":
        with open(HESTON_CAL_PATH) as f:
            cal = json.load(f)
        hp = cal["heston_params"]
        return Heston(
            n_steps=N_STEPS, T=T,
            v0=hp["V0"],
            kappa=hp["kappa"],
            theta=hp["theta"],
            sigma_v=hp["sigma_v"],
            rho=hp["rho"],
        )
    elif source == "rbergomi_H03":
        return DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T,
            H=0.3, eta=ETA_RB, rho=RHO_RB, xi0=XI0,
        )
    elif source == "rbergomi_canonical":
        return DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T,
            H=H_RB, eta=ETA_RB, rho=RHO_RB, xi0=XI0,
        )
    else:
        raise ValueError(f"Unknown source: {source}")

def build_rbergomi_at_H(H: float) -> DifferentiableRoughBergomi:
    return DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T,
        H=H, eta=ETA_RB, rho=RHO_RB, xi0=XI0,
    )

# --------------------------------------------------------------------------
# Setup: shared canonical test set
# --------------------------------------------------------------------------

def setup_shared_test_set() -> dict[str, Any]:
    """Generate (or load cached) canonical rough Bergomi test set."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SHARED_TEST_PATH.exists():
        cache = torch.load(SHARED_TEST_PATH, weights_only=False)
        if (cache.get("seed") == TEST_SEED
                and cache["S"].shape[0] == N_TEST
                and cache.get("H") == H_RB):
            print(f"  Loaded cached shared test set from {SHARED_TEST_PATH}", flush=True)
            return cache
    print(f"  Generating canonical rough Bergomi test set "
          f"({N_TEST} paths, seed={TEST_SEED})...", flush=True)
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H_RB, eta=ETA_RB, rho=RHO_RB, xi0=XI0,
    )
    S, V, t_grid = sim.simulate(n_paths=N_TEST, S0=S0, seed=TEST_SEED)
    payoff = compute_payoff(S, K, "call")

    # p0 from independent training-set MC (matches Phase B convention)
    sim_p0 = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H_RB, eta=ETA_RB, rho=RHO_RB, xi0=XI0,
    )
    S_p0, _, _ = sim_p0.simulate(n_paths=80_000, S0=S0, seed=TEST_SEED + 100000)
    p0 = float(compute_payoff(S_p0, K, "call").mean())

    cache = {
        "S": S, "V": V, "t_grid": t_grid,
        "payoff": payoff, "p0": p0,
        "seed": TEST_SEED, "H": H_RB, "eta": ETA_RB, "rho": RHO_RB, "xi0": XI0,
        "n_paths": N_TEST,
    }
    torch.save(cache, SHARED_TEST_PATH)
    print(f"  Cached to {SHARED_TEST_PATH}  (p0 = {p0:.4f})", flush=True)
    return cache

# --------------------------------------------------------------------------
# Hedge evaluation helpers
# --------------------------------------------------------------------------

def evaluate_hedger_on_paths(
    model: DeepHedgerFNN, S_test: torch.Tensor, p0: float, payoff: torch.Tensor,
) -> dict[str, Any]:
    """Run model.hedge_paths and return metrics."""
    model.eval()
    with torch.no_grad():
        deltas = model.hedge_paths(S_test.float(), T, S0).to(S_test.dtype)
        pnl = compute_hedging_pnl(S_test, deltas, payoff, p0, 0.0)
    metrics = compute_all_metrics(pnl)
    metrics["turnover"] = _mean_turnover(deltas)
    metrics["first_weight_sum"] = _first_weight_sum(model)
    return metrics

def train_dh_on_source(
    source: str, n_train: int, n_val: int, seed: int,
    epochs: int = 200, patience: int = 30, batch_size: int = 2048, lr: float = 1e-3,
) -> tuple[DeepHedgerFNN, dict[str, Any], float]:
    """Train one DH on source-model paths.

    Returns (model, history, train_time_s).
    """
    sim = build_source_simulator(source)

    # Generate training paths with deterministic per-seed train_seed
    train_seed = 8000 + seed - 7000  # offsets keep seeds disjoint
    S_all, _, _ = sim.simulate(n_paths=n_train + n_val, S0=S0, seed=train_seed)
    S_tr = S_all[:n_train]
    S_va = S_all[n_train:]

    payoff_tr = compute_payoff(S_tr, K, "call")
    p0 = float(payoff_tr.mean())

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)

    t0 = time.time()
    history = train_deep_hedger(
        model, S_tr, S_va,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=0.0, alpha=0.95,
        lr=lr, batch_size=batch_size,
        epochs=epochs, patience=patience,
        device=torch.device("cpu"), verbose=False,
    )
    train_time = time.time() - t0

    del S_all, S_tr, S_va, payoff_tr
    gc.collect()
    return model, history, train_time

# --------------------------------------------------------------------------
# L.1 — Multi-source zero-shot transfer
# --------------------------------------------------------------------------

def run_L1_multi_source(seeds: list[int] | None = None,
                          sources: list[str] | None = None,
                          out_path: Path | None = None) -> dict[str, Any]:
    """L.1: train on each source, evaluate on canonical rough Bergomi test set."""
    if seeds is None:
        seeds = [7001, 7002, 7003, 7004, 7005]
    if sources is None:
        sources = ["gbm", "heston", "rbergomi_H03"]

    print("=" * 70, flush=True)
    print(f"  L.1 — Multi-source zero-shot ({len(sources)} sources × {len(seeds)} seeds)",
          flush=True)
    print("=" * 70, flush=True)

    # Shared test set
    test = setup_shared_test_set()
    S_test = test["S"]
    payoff_test = test["payoff"]
    p0 = test["p0"]
    print(f"  Test set: N={S_test.shape[0]}, p0={p0:.4f}", flush=True)

    out: dict[str, Any] = {}
    intermediate_path = out_path or (OUT_DIR / "L1_multi_source_5seeds.json")

    for src in sources:
        print(f"\n  --- Source: {src} ---", flush=True)
        out[src] = {"per_seed": {}, "aggregate": None}
        for seed in seeds:
            print(f"    seed={seed} training...", flush=True)
            try:
                model, hist, ttrain = train_dh_on_source(
                    src, n_train=80_000, n_val=20_000, seed=seed,
                    epochs=200, patience=30,
                )
                metrics = evaluate_hedger_on_paths(model, S_test, p0, payoff_test)
                metrics["best_epoch"] = int(hist["best_epoch"])
                metrics["train_time_s"] = ttrain
                out[src]["per_seed"][str(seed)] = metrics
                print(f"      ES_0.95 = {metrics['es_95']:.4f}  "
                      f"first_weight_sum = {metrics['first_weight_sum']:.6f}  "
                      f"({ttrain/60:.1f} min)", flush=True)
            except Exception as exc:
                print(f"      ERROR seed {seed}: {exc}", flush=True)
                out[src]["per_seed"][str(seed)] = {"error": str(exc)}
            del model
            gc.collect()
            # Incremental save
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)

        # Aggregate per source
        es_values = [r["es_95"] for r in out[src]["per_seed"].values()
                     if "error" not in r]
        out[src]["aggregate"] = {"es_95": _agg(es_values)}

    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)

    print("\n  --- L.1 SUMMARY ---", flush=True)
    for src in sources:
        ag = out[src].get("aggregate", {}).get("es_95", {})
        if ag:
            print(f"    {src:18s}: ES_0.95 = {ag['mean']:.4f} ± {ag['std']:.4f}",
                  flush=True)
    print(f"    BS reference:       {CANONICAL_BS_5SEED_MEAN:.4f} ± "
          f"{CANONICAL_BS_5SEED_STD:.4f}", flush=True)
    print(f"    Canonical DH:       {CANONICAL_DH_5SEED_MEAN:.4f} ± "
          f"{CANONICAL_DH_5SEED_STD:.4f}", flush=True)
    return final

# --------------------------------------------------------------------------
# L.2 — Pretraining budget sweep
# --------------------------------------------------------------------------

def run_L2_budget_sweep(
    seeds: list[int] | None = None,
    sources: list[str] | None = None,
    budgets: list[int] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """L.2: vary N_train per source, find min N to beat BS."""
    if seeds is None:
        seeds = [7101, 7102, 7103]
    if sources is None:
        sources = ["gbm", "heston", "rbergomi_H03"]
    if budgets is None:
        budgets = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]

    print("=" * 70, flush=True)
    print(f"  L.2 — Budget sweep ({len(sources)} sources × {len(budgets)} budgets × "
          f"{len(seeds)} seeds = {len(sources)*len(budgets)*len(seeds)} trainings)",
          flush=True)
    print("=" * 70, flush=True)

    test = setup_shared_test_set()
    S_test = test["S"]
    payoff_test = test["payoff"]
    p0 = test["p0"]

    intermediate_path = out_path or (OUT_DIR / "L2_budget_sweep.json")

    # RESUME: load existing data so already-done cells are preserved.
    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            existing_results = existing.get("results", {})
            for src_key, src_cells in existing_results.items():
                out[src_key] = {}
                for N_key, cell in src_cells.items():
                    out[src_key][N_key] = cell
            n_existing = sum(1 for src in out.values() for c in src.values()
                              for s, m in c.get("per_seed", {}).items()
                              if isinstance(m, dict) and "es_95" in m)
            print(f"  RESUME: loaded {n_existing} existing seeds from "
                  f"{intermediate_path.name}", flush=True)
        except Exception as exc:
            print(f"  RESUME failed ({exc}); starting fresh", flush=True)
            out = {}

    for src in sources:
        if src not in out:
            out[src] = {}
        for N in budgets:
            # Adjust epochs/patience to N (avoid overtraining at small N)
            if N <= 10_000:
                epochs = 100
                patience = 20
            elif N <= 40_000:
                epochs = 150
                patience = 25
            else:
                epochs = 200
                patience = 30
            existing_cell = out.get(src, {}).get(str(N))
            existing_per_seed = (existing_cell or {}).get("per_seed", {}) if existing_cell else {}
            print(f"\n  {src}  N={N}  (epochs={epochs}, patience={patience})",
                  flush=True)
            cell = existing_cell if existing_cell else {
                "N": N, "epochs": epochs, "patience": patience,
                "per_seed": {}, "aggregate": None,
            }
            for seed in seeds:
                if str(seed) in existing_per_seed and \
                        isinstance(existing_per_seed[str(seed)], dict) and \
                        "es_95" in existing_per_seed[str(seed)]:
                    print(f"    seed={seed} SKIP (already done, "
                          f"ES={existing_per_seed[str(seed)]['es_95']:.4f})",
                          flush=True)
                    continue
                print(f"    seed={seed} ...", flush=True)
                try:
                    model, hist, ttrain = train_dh_on_source(
                        src, n_train=N, n_val=max(N // 4, 1000), seed=seed,
                        epochs=epochs, patience=patience,
                    )
                    metrics = evaluate_hedger_on_paths(model, S_test, p0, payoff_test)
                    metrics["train_time_s"] = ttrain
                    metrics["best_epoch"] = int(hist["best_epoch"])
                    cell["per_seed"][str(seed)] = metrics
                    print(f"      ES_0.95 = {metrics['es_95']:.4f}  "
                          f"({ttrain/60:.1f} min)", flush=True)
                except Exception as exc:
                    print(f"      ERROR: {exc}", flush=True)
                    cell["per_seed"][str(seed)] = {"error": str(exc)}
                del model
                gc.collect()

            es_values = [r["es_95"] for r in cell["per_seed"].values()
                         if isinstance(r, dict) and "es_95" in r]
            cell["aggregate"] = {"es_95": _agg(es_values)}
            out[src][str(N)] = cell

            # Incremental save after each (src, N) cell
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)

    return {"meta": _meta(), "results": _strip_for_json(out)}

# --------------------------------------------------------------------------
# L.3 — Extended fine-tuning curve
# --------------------------------------------------------------------------

def _load_or_train_gbm_pretrained(seed: int = 2024) -> DeepHedgerFNN:
    """Load GBM-pretrained checkpoint or train one if missing.

    Note: the canonical GBM-pretrained checkpoint at
    `figures/gbm_pretrained_hedger.pt` is the deterministic reference for
    L.3. Use it for all L.3 fine-tune runs to share the same starting point.
    """
    if GBM_PRETRAINED_PATH.exists():
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        state = torch.load(GBM_PRETRAINED_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
    # Fallback: train from scratch on GBM
    print("    [no GBM checkpoint; training one]", flush=True)
    model, _, _ = train_dh_on_source(
        "gbm", n_train=80_000, n_val=20_000, seed=seed,
        epochs=200, patience=30,
    )
    return model

def fine_tune_dh(
    base_model_state: dict[str, torch.Tensor],
    S_ft: torch.Tensor, S_va: torch.Tensor,
    p0: float, seed: int,
    epochs: int = 30, patience: int = 5, lr: float = 5e-4,
) -> tuple[DeepHedgerFNN, dict[str, Any]]:
    """Fine-tune DH starting from base_model_state, with short budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    model.load_state_dict(base_model_state)
    history = train_deep_hedger(
        model, S_ft, S_va,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=0.0, alpha=0.95,
        lr=lr, batch_size=min(2048, max(64, len(S_ft) // 4)),
        epochs=epochs, patience=patience,
        verbose=False,
    )
    return model, history

def train_from_scratch_rbergomi(
    n_train: int, seed: int,
    epochs: int = 100, patience: int = 20,
) -> tuple[DeepHedgerFNN, dict[str, Any]]:
    """Train fresh DH on n_train rough Bergomi paths."""
    return train_dh_on_source(
        "rbergomi_canonical", n_train=n_train,
        n_val=max(n_train // 4, 200),
        seed=seed,
        epochs=epochs, patience=patience,
    )[:2]

def run_L3_fine_tuning_extended(
    seeds: list[int] | None = None,
    n_ft_values: list[int] | None = None,
) -> dict[str, Any]:
    """L.3: GBM-pretrained → fine-tune on n_ft canonical paths vs from-scratch."""
    if seeds is None:
        seeds = [7201, 7202, 7203]
    if n_ft_values is None:
        n_ft_values = [0, 100, 250, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 80_000]

    print("=" * 70, flush=True)
    print(f"  L.3 — Extended fine-tuning curve ({len(n_ft_values)} n_ft × "
          f"{len(seeds)} seeds × 2 regimes)", flush=True)
    print("=" * 70, flush=True)

    test = setup_shared_test_set()
    S_test = test["S"]
    payoff_test = test["payoff"]
    p0_test = test["p0"]
    print(f"  Test set: N={S_test.shape[0]}, p0={p0_test:.4f}", flush=True)

    # Load or train GBM-pretrained
    print("  Loading GBM-pretrained checkpoint ...", flush=True)
    base_model = _load_or_train_gbm_pretrained(seed=2024)
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    base_zero_shot = evaluate_hedger_on_paths(base_model, S_test, p0_test, payoff_test)
    print(f"  GBM-pretrained zero-shot ES_0.95 = {base_zero_shot['es_95']:.4f}",
          flush=True)
    del base_model
    gc.collect()

    out = {
        "base_zero_shot": base_zero_shot,
        "fine_tune": {},
        "from_scratch": {},
    }
    intermediate_path = OUT_DIR / "L3_fine_tuning_extended.json"

    sim = build_source_simulator("rbergomi_canonical")

    for n_ft in n_ft_values:
        print(f"\n  n_ft = {n_ft}", flush=True)
        ft_cell = {"n_ft": n_ft, "per_seed": {}, "aggregate": None}
        fs_cell = {"n_ft": n_ft, "per_seed": {}, "aggregate": None}

        for seed in seeds:
            data_seed = 8000 + seed
            # ---- Fine-tune branch ----
            if n_ft == 0:
                # zero-shot: just load GBM pretrained
                ft_metrics = dict(base_zero_shot)
                ft_metrics["best_epoch"] = 0
                ft_metrics["train_time_s"] = 0.0
            else:
                S_ft_all, _, _ = sim.simulate(
                    n_paths=int(n_ft + max(n_ft // 4, 100)),
                    S0=S0, seed=data_seed,
                )
                S_ft = S_ft_all[:n_ft]
                S_va = S_ft_all[n_ft:]
                p0_ft = float(compute_payoff(S_ft, K, "call").mean())
                t0 = time.time()
                ft_model, _ = fine_tune_dh(
                    base_state, S_ft, S_va, p0_ft, seed=seed,
                    epochs=30, patience=5, lr=5e-4,
                )
                ttrain = time.time() - t0
                ft_metrics = evaluate_hedger_on_paths(
                    ft_model, S_test, p0_test, payoff_test,
                )
                ft_metrics["train_time_s"] = ttrain
                del ft_model, S_ft_all, S_ft, S_va
                gc.collect()
            ft_cell["per_seed"][str(seed)] = ft_metrics
            print(f"    seed={seed}  fine-tune ES_0.95 = {ft_metrics['es_95']:.4f}",
                  flush=True)

            # ---- From-scratch branch ----
            if n_ft == 0:
                # No data → can't train from scratch; report BS reference (BS is fixed)
                bs_hedger = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
                deltas = bs_hedger.hedge_paths(S_test)
                pnl = compute_hedging_pnl(S_test, deltas, payoff_test, p0_test, 0.0)
                fs_metrics = compute_all_metrics(pnl)
                fs_metrics["turnover"] = _mean_turnover(deltas)
                fs_metrics["first_weight_sum"] = float("nan")
                fs_metrics["best_epoch"] = 0
                fs_metrics["train_time_s"] = 0.0
                fs_metrics["note"] = "no data; reporting BS reference"
            else:
                # Choose epochs proportional to n_ft (prevent overfit at small N)
                epochs = 100 if n_ft >= 5000 else 60
                patience = 20 if n_ft >= 5000 else 10
                t0 = time.time()
                fs_model, _ = train_from_scratch_rbergomi(
                    n_train=n_ft, seed=seed + 1000,
                    epochs=epochs, patience=patience,
                )
                ttrain = time.time() - t0
                fs_metrics = evaluate_hedger_on_paths(
                    fs_model, S_test, p0_test, payoff_test,
                )
                fs_metrics["train_time_s"] = ttrain
                del fs_model
                gc.collect()
            fs_cell["per_seed"][str(seed)] = fs_metrics
            print(f"    seed={seed}  from-scratch ES_0.95 = {fs_metrics['es_95']:.4f}",
                  flush=True)

        ft_es = [r["es_95"] for r in ft_cell["per_seed"].values() if "error" not in r]
        fs_es = [r["es_95"] for r in fs_cell["per_seed"].values() if "error" not in r]
        ft_cell["aggregate"] = {"es_95": _agg(ft_es)}
        fs_cell["aggregate"] = {"es_95": _agg(fs_es)}
        out["fine_tune"][str(n_ft)] = ft_cell
        out["from_scratch"][str(n_ft)] = fs_cell

        _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                    intermediate_path)

    return {"meta": _meta(), "results": _strip_for_json(out)}

# --------------------------------------------------------------------------
# L.4 — Reverse transfer
# --------------------------------------------------------------------------

def _load_canonical_dh() -> DeepHedgerFNN:
    """Load the canonical rough-Bergomi-trained DH checkpoint."""
    if not CANONICAL_DH_PATH.exists():
        raise FileNotFoundError(
            f"Canonical DH checkpoint not found: {CANONICAL_DH_PATH}. "
            "Need to retrain or use Phase B output."
        )
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    state = torch.load(CANONICAL_DH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model

def run_L4_reverse_transfer(
    seeds: list[int] | None = None,
    targets: list[str] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """L.4: rBergomi-trained DH → evaluate on GBM and Heston test sets."""
    if seeds is None:
        seeds = [7301, 7302, 7303]
    if targets is None:
        targets = ["gbm", "heston"]

    print("=" * 70, flush=True)
    print(f"  L.4 — Reverse transfer (rBergomi-trained → {targets})", flush=True)
    print("=" * 70, flush=True)

    print("  Loading canonical DH checkpoint ...", flush=True)
    dh = _load_canonical_dh()
    out = {"per_target": {}}

    # Need Heston PDE for Heston-target reference
    heston_pde = None
    if "heston" in targets:
        with open(HESTON_CAL_PATH) as f:
            cal = json.load(f)
        hp = cal["heston_params"]
        print("  Building Heston PDE delta (calibrated) for reference ...", flush=True)
        heston_pde = HestonPDEDelta(
            kappa=hp["kappa"], theta=hp["theta"], sigma_v=hp["sigma_v"],
            rho=hp["rho"], V0=hp["V0"],
            K=K, T=T,
            S_max=400.0, V_max=1.0,
            n_S=200, n_V=80, n_t=400,
        )

    for target in targets:
        print(f"\n  --- Target: {target} ---", flush=True)
        sim = build_source_simulator(target)
        cell = {"target": target, "per_seed": {}, "aggregate": None}

        for seed in seeds:
            print(f"    seed={seed} ...", flush=True)
            try:
                S_test, V_test, t_grid = sim.simulate(n_paths=N_TEST, S0=S0, seed=seed)
                payoff = compute_payoff(S_test, K, "call")
                # MC p0 from independent batch
                S_p0, _, _ = sim.simulate(
                    n_paths=N_TEST, S0=S0, seed=seed + 100000,
                )
                p0 = float(compute_payoff(S_p0, K, "call").mean())
                del S_p0
                gc.collect()

                # DH eval
                dh.eval()
                with torch.no_grad():
                    deltas_dh = dh.hedge_paths(S_test.float(), T, S0).to(S_test.dtype)
                pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff, p0, 0.0)
                m_dh = compute_all_metrics(pnl_dh)
                m_dh["turnover"] = _mean_turnover(deltas_dh)
                m_dh["first_weight_sum"] = _first_weight_sum(dh)

                # Reference
                if target == "gbm":
                    bs_hedger = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
                    deltas_ref = bs_hedger.hedge_paths(S_test)
                    pnl_ref = compute_hedging_pnl(S_test, deltas_ref, payoff, p0, 0.0)
                    m_ref = compute_all_metrics(pnl_ref)
                    m_ref["turnover"] = _mean_turnover(deltas_ref)
                    ref_label = "bs_delta"
                elif target == "heston":
                    deltas_ref = heston_pde.hedge_paths(
                        S_test, V_test,
                        t_grid=torch.linspace(0, T, N_STEPS + 1, dtype=S_test.dtype),
                    )
                    pnl_ref = compute_hedging_pnl(S_test, deltas_ref, payoff, p0, 0.0)
                    m_ref = compute_all_metrics(pnl_ref)
                    m_ref["turnover"] = _mean_turnover(deltas_ref)
                    ref_label = "heston_pde"

                gap = m_dh["es_95"] - m_ref["es_95"]
                cell["per_seed"][str(seed)] = {
                    "p0": p0,
                    "dh": m_dh,
                    ref_label: m_ref,
                    "ref_label": ref_label,
                    "gap_dh_minus_ref_es95": gap,
                }
                print(f"      DH ES_0.95 = {m_dh['es_95']:.4f}  "
                      f"{ref_label} ES_0.95 = {m_ref['es_95']:.4f}  "
                      f"gap = {gap:+.4f}", flush=True)
            except Exception as exc:
                print(f"      ERROR: {exc}", flush=True)
                cell["per_seed"][str(seed)] = {"error": str(exc)}

        dh_es = [r["dh"]["es_95"] for r in cell["per_seed"].values() if "error" not in r]
        ref_es = [r[r["ref_label"]]["es_95"] for r in cell["per_seed"].values()
                   if "error" not in r]
        gap_values = [r["gap_dh_minus_ref_es95"] for r in cell["per_seed"].values()
                       if "error" not in r]
        cell["aggregate"] = {
            "dh_es95": _agg(dh_es),
            "ref_es95": _agg(ref_es),
            "gap_dh_minus_ref": _agg(gap_values),
        }
        out["per_target"][target] = cell

    intermediate_path = out_path or (OUT_DIR / "L4_reverse_transfer.json")
    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# L.5 — Cross-calibration transfer
# --------------------------------------------------------------------------

def run_L5_cross_calibration(
    seeds: list[int] | None = None,
    target_H_values: list[float] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """L.5: rBergomi-trained DH → evaluate at different H values."""
    if seeds is None:
        seeds = [7401, 7402, 7403]
    if target_H_values is None:
        target_H_values = [0.07, 0.2, 0.4]

    print("=" * 70, flush=True)
    print(f"  L.5 — Cross-calibration ({target_H_values} H values × {len(seeds)} seeds)",
          flush=True)
    print("=" * 70, flush=True)

    print("  Loading canonical DH checkpoint ...", flush=True)
    dh = _load_canonical_dh()
    out = {"per_H": {}}

    for tH in target_H_values:
        print(f"\n  --- Target H = {tH} ---", flush=True)
        sim = build_rbergomi_at_H(tH)
        cell = {"target_H": tH, "per_seed": {}, "aggregate": None}

        for seed in seeds:
            print(f"    seed={seed} ...", flush=True)
            try:
                S_test, V_test, _ = sim.simulate(n_paths=N_TEST, S0=S0, seed=seed)
                payoff = compute_payoff(S_test, K, "call")
                S_p0, _, _ = sim.simulate(n_paths=N_TEST, S0=S0, seed=seed + 100000)
                p0 = float(compute_payoff(S_p0, K, "call").mean())
                del S_p0; gc.collect()

                dh.eval()
                with torch.no_grad():
                    deltas_dh = dh.hedge_paths(S_test.float(), T, S0).to(S_test.dtype)
                pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff, p0, 0.0)
                m_dh = compute_all_metrics(pnl_dh)
                m_dh["turnover"] = _mean_turnover(deltas_dh)
                m_dh["first_weight_sum"] = _first_weight_sum(dh)

                # BS reference (constant σ = √ξ_0)
                bs = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
                deltas_bs = bs.hedge_paths(S_test)
                pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff, p0, 0.0)
                m_bs = compute_all_metrics(pnl_bs)
                m_bs["turnover"] = _mean_turnover(deltas_bs)

                cell["per_seed"][str(seed)] = {
                    "p0": p0,
                    "dh": m_dh, "bs": m_bs,
                    "gap_dh_minus_bs_es95": m_dh["es_95"] - m_bs["es_95"],
                }
                print(f"      DH ES = {m_dh['es_95']:.4f}  "
                      f"BS ES = {m_bs['es_95']:.4f}  "
                      f"gap = {m_dh['es_95'] - m_bs['es_95']:+.4f}", flush=True)
            except Exception as exc:
                print(f"      ERROR: {exc}", flush=True)
                cell["per_seed"][str(seed)] = {"error": str(exc)}

        dh_es = [r["dh"]["es_95"] for r in cell["per_seed"].values() if "error" not in r]
        bs_es = [r["bs"]["es_95"] for r in cell["per_seed"].values() if "error" not in r]
        gap = [r["gap_dh_minus_bs_es95"] for r in cell["per_seed"].values()
                if "error" not in r]
        cell["aggregate"] = {
            "dh_es95": _agg(dh_es),
            "bs_es95": _agg(bs_es),
            "gap_dh_minus_bs": _agg(gap),
        }
        out["per_H"][str(tH)] = cell

    intermediate_path = out_path or (OUT_DIR / "L5_cross_calibration.json")
    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# Reproducibility check helper
# --------------------------------------------------------------------------

def reproducibility_check_L(
    sub_experiment: str, output_path: Path,
) -> dict[str, Any]:
    """Run a single seed in fresh subprocess and compare key values."""
    cmd = [sys.executable, "-u", "-m",
           "deep_hedging.experiments.transfer_extended",
           f"--repro-{sub_experiment}",
           "--repro-output", str(output_path)]
    print(f"  Running reproducibility subprocess for {sub_experiment} ...",
          flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    if result.returncode != 0:
        return {"error": "subprocess failed",
                "stderr_tail": result.stderr[-1500:],
                "wall_clock_s": wall}
    return {"ok": True, "wall_clock_s": wall, "output_path": str(output_path)}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _meta() -> dict[str, Any]:
    return {
        "script": "deep_hedging/experiments/transfer_extended.py",
        "git_commit": _git_commit_sha(),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "rbergomi_canonical": {
            "H": H_RB, "eta": ETA_RB, "rho": RHO_RB, "xi0": XI0,
            "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true",
                        help="Generate shared test set + verify infrastructure.")
    parser.add_argument("--L1", action="store_true")
    parser.add_argument("--L2", action="store_true")
    parser.add_argument("--L3", action="store_true")
    parser.add_argument("--L4", action="store_true")
    parser.add_argument("--L5", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repro-L2", action="store_true",
                        help="Reproducibility mode: gbm N=80000 seed 7101.")
    parser.add_argument("--repro-L1", action="store_true",
                        help="Reproducibility: gbm src, seed 7001 only.")
    parser.add_argument("--repro-L4", action="store_true",
                        help="Reproducibility: target=gbm, seed 7301 only.")
    parser.add_argument("--repro-L5", action="store_true",
                        help="Reproducibility: target_H=0.2, seed 7401 only.")
    parser.add_argument("--repro-output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--sources", nargs="+", type=str, default=None)
    parser.add_argument("--ft-values", nargs="+", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.setup:
        setup_shared_test_set()
        # Verify Heston cal accessible
        if HESTON_CAL_PATH.exists():
            print(f"  Heston calibration: OK ({HESTON_CAL_PATH})", flush=True)
        else:
            print(f"  Heston calibration: MISSING ({HESTON_CAL_PATH})", flush=True)
        # Verify checkpoints
        print(f"  GBM-pretrained checkpoint: "
              f"{'OK' if GBM_PRETRAINED_PATH.exists() else 'MISSING'}", flush=True)
        print(f"  Canonical DH checkpoint:   "
              f"{'OK' if CANONICAL_DH_PATH.exists() else 'MISSING'}", flush=True)
        return

    # Reproducibility subprocess modes
    # IMPORTANT: each repro call passes its own out_path so the main file
    # (e.g. L1_multi_source_5seeds.json) is NOT overwritten by the single-seed
    # incremental save inside run_L*().
    if args.repro_L1:
        out_path = Path(args.repro_output or (OUT_DIR / "L1_repro.json"))
        res = run_L1_multi_source(seeds=[7001], sources=["gbm"],
                                   out_path=out_path)
        _save_json(res, out_path)
        print(f"  Wrote {out_path}", flush=True)
        return
    if args.repro_L2:
        out_path = Path(args.repro_output or (OUT_DIR / "L2_repro.json"))
        res = run_L2_budget_sweep(seeds=[7101], sources=["gbm"],
                                    budgets=[80_000], out_path=out_path)
        _save_json(res, out_path)
        print(f"  Wrote {out_path}", flush=True)
        return
    if args.repro_L4:
        out_path = Path(args.repro_output or (OUT_DIR / "L4_repro.json"))
        res = run_L4_reverse_transfer(seeds=[7301], targets=["gbm"],
                                        out_path=out_path)
        _save_json(res, out_path)
        print(f"  Wrote {out_path}", flush=True)
        return
    if args.repro_L5:
        out_path = Path(args.repro_output or (OUT_DIR / "L5_repro.json"))
        res = run_L5_cross_calibration(seeds=[7401], target_H_values=[0.2],
                                         out_path=out_path)
        _save_json(res, out_path)
        print(f"  Wrote {out_path}", flush=True)
        return

    if args.all or args.L4:
        res = run_L4_reverse_transfer(seeds=args.seeds)
    if args.all or args.L5:
        res = run_L5_cross_calibration(seeds=args.seeds)
    if args.all or args.L1:
        res = run_L1_multi_source(seeds=args.seeds, sources=args.sources)
    if args.all or args.L3:
        res = run_L3_fine_tuning_extended(seeds=args.seeds, n_ft_values=args.ft_values)
    if args.all or args.L2:
        res = run_L2_budget_sweep(seeds=args.seeds, sources=args.sources,
                                    budgets=args.budgets)

if __name__ == "__main__":
    main()
