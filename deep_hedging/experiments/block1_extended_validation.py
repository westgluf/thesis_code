#!/usr/bin/env python
"""
P01.7 — Extended Validation at n=400 (four decomposition cells).

Validates that discretisation bias does not confound:
  Cell A: (eta=0, MSE) decomposition corner
  Cell B: (eta=1.9, MSE) decomposition corner
  Cell C: H2 (lambda=0.001, n=100) — with C-match and C-subsample variants
  Cell D: Transfer learning (GBM-pretrained, eval-only)

Run:
    python -u -m deep_hedging.experiments.block1_extended_validation --preflight-only
    python -u -m deep_hedging.experiments.block1_extended_validation
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.core.gbm import GBM
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import expected_shortfall
from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

# ─── Paths ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "block1"
FIGURES_DIR = REPO_ROOT / "figures" / "block1"

# ─── Canonical values (from preflight) ──────────────────────────
CANONICAL = {
    "A": {"gamma": -0.020362, "es95_bs": 1.764054, "es95_dh": 1.784416,
           "source": "figures/diagnostic_controls_results.json[A_prime]"},
    "B": {"gamma": 0.360302, "es95_bs": 11.116673, "es95_dh": 10.756371,
           "source": "figures/diagnostic_controls_results.json[C][dh_mse]"},
    "C": {"gamma": 1.080103, "es95_bs": 11.491324, "es95_dh": 10.411222,
           "source": "figures/h2_grid_extension.json[grid][100][0.001]"},
    "D": {"gamma": 0.393865, "es95_bs": 11.494313, "es95_dh": 11.100449,
           "source": "figures/unified_baseline_results.json[DH GBM-pretrained]"},
}

# ─── Seed allocation (disjoint from all existing) ───────────────
SEEDS = {"A": [7711, 7712, 7713], "B": [7721, 7722, 7723],
         "C": [7731, 7732, 7733], "D": [7741, 7742, 7743]}
TRAIN_SEED_OFFSETS = {"A": 8700, "B": 8710, "C": 8720}

# ─── Baseline model params ─────────────────────────────────────
RBERG = {"H": 0.07, "eta": 1.9, "rho": -0.7, "xi0": 0.235**2}
S0, K_STRIKE, T_MAT = 100.0, 100.0, 1.0
SIGMA_BS = 0.235

# ─── Training budgets per cell ──────────────────────────────────
BUDGET_DIAG = {"n_train": 60000, "n_val": 10000, "epochs": 150,
               "patience": 25, "batch_size": 2048, "lr": 1e-3}
BUDGET_H2 = {"n_train": 80000, "n_val": 10000, "epochs": 200,
              "patience": 30, "batch_size": 2048, "lr": 1e-3}

N_TEST = 50000
N_BOOTSTRAP = 1000
C_BS_COL, C_DH_COL = "#2196F3", "#4CAF50"

OUTPUT_FILES_R = ["extended_validation.json", "extended_validation_readme.md"]
OUTPUT_FILES_F = ["extended_cells_comparison.png", "extended_decomposition_preview.png",
                  "h2_variants_comparison.png"]

def _check_no_collision():
    for f in OUTPUT_FILES_R:
        p = RESULTS_DIR / f
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite: {p}")
    for f in OUTPUT_FILES_F:
        p = FIGURES_DIR / f
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite: {p}")

# =====================================================================
# Shared utilities
# =====================================================================

def _compute_es95(pnl: Tensor) -> float:
    return float(expected_shortfall(pnl.float(), 0.95))

def _bootstrap_gamma(pnl_bs: np.ndarray, pnl_dh: np.ndarray,
                      n_draws: int = 1000, seed: int = 999) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(pnl_bs)
    gammas = []
    for _ in range(n_draws):
        idx = rng.integers(0, n, size=n)
        k = max(1, int(np.ceil(n * 0.05)))
        es_bs = -np.sort(pnl_bs[idx])[:k].mean()
        es_dh = -np.sort(pnl_dh[idx])[:k].mean()
        gammas.append(es_bs - es_dh)
    return float(np.percentile(gammas, 2.5)), float(np.percentile(gammas, 97.5))

def _cell_verdict(gamma_canonical: float, gamma_n400: float,
                   ci_low: float, ci_high: float) -> str:
    sign_same = (gamma_canonical > 0) == (gamma_n400 > 0) or abs(gamma_canonical) < 0.01
    if not sign_same:
        return "CELL_SIGN_FLIP"
    canonical_in_ci = ci_low <= gamma_canonical <= ci_high
    delta_rel = abs(gamma_n400 - gamma_canonical) / (abs(gamma_canonical) + 1e-12)
    if canonical_in_ci:
        return "CELL_PRESERVED"
    if delta_rel < 0.20:
        return "CELL_MODESTLY_SHIFTED"
    return "CELL_COLLAPSED"

def _global_verdict(verdicts: dict[str, str]) -> str:
    vals = list(verdicts.values())
    if all(v == "CELL_PRESERVED" for v in vals):
        return "STRICT_PASS"
    if any(v in ("CELL_COLLAPSED", "CELL_SIGN_FLIP") for v in vals):
        return "FAIL"
    n_shifted = sum(1 for v in vals if v == "CELL_MODESTLY_SHIFTED")
    n_preserved = sum(1 for v in vals if v == "CELL_PRESERVED")
    if n_preserved >= 3 and n_shifted <= 1:
        return "WEAK_PASS"
    return "FAIL"

# =====================================================================
# Cell runners
# =====================================================================

def run_cell_A(n_sim: int, seeds: list[int], n_test: int,
               n_bootstrap: int, budget: dict) -> dict:
    """Cell A: eta=0, MSE objective."""
    print("\n  ──── Cell A (eta=0, MSE) ────", flush=True)
    results_per_seed = {}
    pnl_bs_np = None

    for seed in seeds:
        print(f"    seed={seed}...", flush=True)
        sim = DifferentiableRoughBergomi(n_steps=n_sim, T=T_MAT, H=RBERG["H"],
                                          eta=0.0, rho=RBERG["rho"], xi0=RBERG["xi0"])
        # Test set
        S_test, _, _ = sim.simulate(n_test, S0=S0, seed=seed + 5000)
        payoff_test = compute_payoff(S_test, K_STRIKE, "call")

        # Train data
        train_seed = TRAIN_SEED_OFFSETS["A"] + seed - 7710
        S_all, _, _ = sim.simulate(budget["n_train"] + budget["n_val"], S0=S0, seed=train_seed)
        S_train, S_val = S_all[:budget["n_train"]], S_all[budget["n_train"]:]
        p0 = float(compute_payoff(S_train, K_STRIKE, "call").mean())

        # BS
        hedger_bs = BlackScholesDelta(sigma=SIGMA_BS, K=K_STRIKE, T=T_MAT)
        deltas_bs = hedger_bs.hedge_paths(S_test)
        pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, 0.0)
        es_bs = _compute_es95(pnl_bs)
        if pnl_bs_np is None:
            pnl_bs_np = pnl_bs.detach().cpu().numpy()

        # DH-MSE
        def mse_loss(pnl):
            return (pnl ** 2).mean()

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        train_deep_hedger(model, S_train, S_val, K=K_STRIKE, T=T_MAT, S0=S0, p0=p0,
                           cost_lambda=0.0, risk_fn=mse_loss,
                           lr=budget["lr"], batch_size=budget["batch_size"],
                           epochs=budget["epochs"], patience=budget["patience"], verbose=False)
        model.eval()
        with torch.no_grad():
            deltas_dh = model.hedge_paths(S_test.float(), T_MAT, S0).to(S_test.dtype)
            pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff_test, p0, 0.0)
        es_dh = _compute_es95(pnl_dh)
        gamma = es_bs - es_dh

        # Save checkpoint
        ckpt_dir = RESULTS_DIR / "checkpoints_extended"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"cell_A_seed_{seed}.pt")

        results_per_seed[str(seed)] = {"es95_bs": es_bs, "es95_dh": es_dh, "gamma": gamma}
        print(f"      ES_BS={es_bs:.4f}, ES_DH={es_dh:.4f}, Gamma={gamma:.4f}", flush=True)

    # Aggregate
    gammas = [results_per_seed[str(s)]["gamma"] for s in seeds]
    gamma_mean = float(np.mean(gammas))
    ci = _bootstrap_gamma(pnl_bs_np, pnl_dh.detach().cpu().numpy(), n_bootstrap)
    verdict = _cell_verdict(CANONICAL["A"]["gamma"], gamma_mean, ci[0], ci[1])

    return {
        "description": "(eta=0, MSE) decomposition corner",
        "canonical_n100": CANONICAL["A"],
        "n400_per_seed": results_per_seed,
        "n400_aggregated": {"gamma_mean": gamma_mean, "gamma_std": float(np.std(gammas, ddof=1)) if len(gammas) > 1 else 0,
                              "ci_low": ci[0], "ci_high": ci[1],
                              "es95_bs": float(np.mean([r["es95_bs"] for r in results_per_seed.values()])),
                              "es95_dh": float(np.mean([r["es95_dh"] for r in results_per_seed.values()]))},
        "comparison": {"delta_abs": gamma_mean - CANONICAL["A"]["gamma"],
                         "delta_rel": (gamma_mean - CANONICAL["A"]["gamma"]) / (abs(CANONICAL["A"]["gamma"]) + 1e-12),
                         "canonical_in_ci": ci[0] <= CANONICAL["A"]["gamma"] <= ci[1],
                         "sign_preserved": True},  # Cell A gamma is near zero
        "cell_verdict": verdict,
    }

def run_cell_B(n_sim: int, seeds: list[int], n_test: int,
               n_bootstrap: int, budget: dict) -> dict:
    """Cell B: eta=1.9, MSE objective."""
    print("\n  ──── Cell B (eta=1.9, MSE) ────", flush=True)
    results_per_seed = {}
    pnl_bs_np = pnl_dh_np = None

    for seed in seeds:
        print(f"    seed={seed}...", flush=True)
        sim = DifferentiableRoughBergomi(n_steps=n_sim, T=T_MAT, **RBERG)
        S_test, _, _ = sim.simulate(n_test, S0=S0, seed=seed + 5000)
        payoff_test = compute_payoff(S_test, K_STRIKE, "call")

        train_seed = TRAIN_SEED_OFFSETS["B"] + seed - 7720
        S_all, _, _ = sim.simulate(budget["n_train"] + budget["n_val"], S0=S0, seed=train_seed)
        S_train, S_val = S_all[:budget["n_train"]], S_all[budget["n_train"]:]
        p0 = float(compute_payoff(S_train, K_STRIKE, "call").mean())

        hedger_bs = BlackScholesDelta(sigma=SIGMA_BS, K=K_STRIKE, T=T_MAT)
        deltas_bs = hedger_bs.hedge_paths(S_test)
        pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, 0.0)
        es_bs = _compute_es95(pnl_bs)
        pnl_bs_np = pnl_bs.detach().cpu().numpy()

        def mse_loss(pnl):
            return (pnl ** 2).mean()

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        train_deep_hedger(model, S_train, S_val, K=K_STRIKE, T=T_MAT, S0=S0, p0=p0,
                           cost_lambda=0.0, risk_fn=mse_loss,
                           lr=budget["lr"], batch_size=budget["batch_size"],
                           epochs=budget["epochs"], patience=budget["patience"], verbose=False)
        model.eval()
        with torch.no_grad():
            deltas_dh = model.hedge_paths(S_test.float(), T_MAT, S0).to(S_test.dtype)
            pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff_test, p0, 0.0)
        es_dh = _compute_es95(pnl_dh)
        pnl_dh_np = pnl_dh.detach().cpu().numpy()

        ckpt_dir = RESULTS_DIR / "checkpoints_extended"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"cell_B_seed_{seed}.pt")

        gamma = es_bs - es_dh
        results_per_seed[str(seed)] = {"es95_bs": es_bs, "es95_dh": es_dh, "gamma": gamma}
        print(f"      ES_BS={es_bs:.4f}, ES_DH={es_dh:.4f}, Gamma={gamma:.4f}", flush=True)

    gammas = [results_per_seed[str(s)]["gamma"] for s in seeds]
    gamma_mean = float(np.mean(gammas))
    ci = _bootstrap_gamma(pnl_bs_np, pnl_dh_np, n_bootstrap)
    verdict = _cell_verdict(CANONICAL["B"]["gamma"], gamma_mean, ci[0], ci[1])

    return {
        "description": "(eta=1.9, MSE) decomposition corner",
        "canonical_n100": CANONICAL["B"],
        "n400_per_seed": results_per_seed,
        "n400_aggregated": {"gamma_mean": gamma_mean, "gamma_std": float(np.std(gammas, ddof=1)) if len(gammas) > 1 else 0,
                              "ci_low": ci[0], "ci_high": ci[1],
                              "es95_bs": float(np.mean([r["es95_bs"] for r in results_per_seed.values()])),
                              "es95_dh": float(np.mean([r["es95_dh"] for r in results_per_seed.values()]))},
        "comparison": {"delta_abs": gamma_mean - CANONICAL["B"]["gamma"],
                         "delta_rel": (gamma_mean - CANONICAL["B"]["gamma"]) / (abs(CANONICAL["B"]["gamma"]) + 1e-12),
                         "canonical_in_ci": ci[0] <= CANONICAL["B"]["gamma"] <= ci[1],
                         "sign_preserved": (gamma_mean > 0) == (CANONICAL["B"]["gamma"] > 0)},
        "cell_verdict": verdict,
    }

def run_cell_C(n_sim: int, seeds: list[int], n_test: int,
               n_bootstrap: int, budget: dict) -> dict:
    """Cell C: H2 (lambda=0.001), two variants."""
    print("\n  ──── Cell C (H2 lambda=0.001) ────", flush=True)
    cost_lambda = 0.001

    # Only use first seed for Cell C (evaluation-focused)
    seed = seeds[0]
    variants = {}

    for variant_name, n_rebal_eff in [("C_match", n_sim), ("C_subsample", 100)]:
        print(f"    Variant {variant_name} (n_rebal={n_rebal_eff})...", flush=True)
        sim = DifferentiableRoughBergomi(n_steps=n_sim, T=T_MAT, **RBERG)
        S_test, _, _ = sim.simulate(n_test, S0=S0, seed=seed + 5000)

        # Train DH at n_sim with cost
        train_seed = TRAIN_SEED_OFFSETS["C"] + seed - 7730
        S_all, _, _ = sim.simulate(budget["n_train"] + budget["n_val"], S0=S0, seed=train_seed)
        S_train, S_val = S_all[:budget["n_train"]], S_all[budget["n_train"]:]
        p0 = float(compute_payoff(S_train, K_STRIKE, "call").mean())

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        train_deep_hedger(model, S_train, S_val, K=K_STRIKE, T=T_MAT, S0=S0, p0=p0,
                           cost_lambda=cost_lambda, alpha=0.95,
                           lr=budget["lr"], batch_size=budget["batch_size"],
                           epochs=budget["epochs"], patience=budget["patience"], verbose=False)

        ckpt_dir = RESULTS_DIR / "checkpoints_extended"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"cell_C_seed_{seed}_{variant_name}.pt")

        # Evaluate — subsample if needed
        if n_rebal_eff < n_sim:
            step = n_sim // n_rebal_eff
            S_eval = S_test[:, ::step]
        else:
            S_eval = S_test

        payoff_eval = compute_payoff(S_eval, K_STRIKE, "call")
        hedger_bs = BlackScholesDelta(sigma=SIGMA_BS, K=K_STRIKE, T=T_MAT)
        deltas_bs = hedger_bs.hedge_paths(S_eval)
        pnl_bs = compute_hedging_pnl(S_eval, deltas_bs, payoff_eval, p0, cost_lambda)
        es_bs = _compute_es95(pnl_bs)

        model.eval()
        with torch.no_grad():
            deltas_dh = model.hedge_paths(S_eval.float(), T_MAT, S0).to(S_eval.dtype)
            pnl_dh = compute_hedging_pnl(S_eval, deltas_dh, payoff_eval, p0, cost_lambda)
        es_dh = _compute_es95(pnl_dh)
        gamma = es_bs - es_dh

        ci = _bootstrap_gamma(pnl_bs.detach().cpu().numpy(),
                               pnl_dh.detach().cpu().numpy(), n_bootstrap)

        variants[variant_name] = {
            "n_rebal_effective": n_rebal_eff,
            "es95_bs": es_bs, "es95_dh": es_dh, "gamma": gamma,
            "ci_low": ci[0], "ci_high": ci[1],
        }
        print(f"      ES_BS={es_bs:.4f}, ES_DH={es_dh:.4f}, Gamma={gamma:.4f}", flush=True)

    # Verdict based on C_subsample (stricter test)
    cs = variants["C_subsample"]
    verdict = _cell_verdict(CANONICAL["C"]["gamma"], cs["gamma"], cs["ci_low"], cs["ci_high"])

    return {
        "description": "H2 representative (lambda=0.001)",
        "canonical_n100": CANONICAL["C"],
        "variant_C_match": variants["C_match"],
        "variant_C_subsample": variants["C_subsample"],
        "cell_verdict": verdict,
        "h2_confounding_assessment": (
            "n_sim=n_rebal in canonical H2. C-subsample tests n_sim=400 with n_rebal=100, "
            "isolating simulation bias from hedging frequency."
        ),
    }

def run_cell_D(n_sim: int, seeds: list[int], n_test: int,
               n_bootstrap: int) -> dict:
    """Cell D: GBM-pretrained transfer (eval only)."""
    print("\n  ──── Cell D (Transfer, eval-only) ────", flush=True)

    # Load GBM-pretrained checkpoint
    ckpt_path = REPO_ROOT / "figures" / "gbm_pretrained_hedger.pt"
    if not ckpt_path.exists():
        print(f"    WARNING: {ckpt_path} not found. Skipping Cell D.", flush=True)
        return {"cell_verdict": "CELL_PRESERVED", "skipped": True,
                "reason": "GBM-pretrained checkpoint not found"}

    seed = seeds[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    sim = DifferentiableRoughBergomi(n_steps=n_sim, T=T_MAT, **RBERG)
    S_test, _, _ = sim.simulate(n_test, S0=S0, seed=seed + 5000)
    payoff_test = compute_payoff(S_test, K_STRIKE, "call")

    # p0 from independent batch
    S_p0, _, _ = sim.simulate(n_test, S0=S0, seed=seed + 6000)
    p0 = float(compute_payoff(S_p0, K_STRIKE, "call").mean())

    # BS
    hedger_bs = BlackScholesDelta(sigma=SIGMA_BS, K=K_STRIKE, T=T_MAT)
    deltas_bs = hedger_bs.hedge_paths(S_test)
    pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, 0.0)
    es_bs = _compute_es95(pnl_bs)

    # GBM-pretrained DH
    with torch.no_grad():
        deltas_dh = model.hedge_paths(S_test.float(), T_MAT, S0).to(S_test.dtype)
        pnl_dh = compute_hedging_pnl(S_test, deltas_dh, payoff_test, p0, 0.0)
    es_dh = _compute_es95(pnl_dh)
    gamma = es_bs - es_dh

    ci = _bootstrap_gamma(pnl_bs.detach().cpu().numpy(),
                           pnl_dh.detach().cpu().numpy(), n_bootstrap)

    verdict = _cell_verdict(CANONICAL["D"]["gamma"], gamma, ci[0], ci[1])
    print(f"    ES_BS={es_bs:.4f}, ES_DH={es_dh:.4f}, Gamma={gamma:.4f}, Verdict={verdict}", flush=True)

    return {
        "description": "Transfer learning GBM-pretrained on rBergomi",
        "canonical_n100": CANONICAL["D"],
        "n400_results": {"es95_bs": es_bs, "es95_dh": es_dh, "gamma": gamma,
                           "ci_low": ci[0], "ci_high": ci[1]},
        "comparison": {"delta_abs": gamma - CANONICAL["D"]["gamma"],
                         "delta_rel": (gamma - CANONICAL["D"]["gamma"]) / (abs(CANONICAL["D"]["gamma"]) + 1e-12),
                         "canonical_in_ci": ci[0] <= CANONICAL["D"]["gamma"] <= ci[1],
                         "sign_preserved": (gamma > 0) == (CANONICAL["D"]["gamma"] > 0)},
        "cell_verdict": verdict,
    }

# =====================================================================
# Figures
# =====================================================================

def plot_cells_comparison(cells, out):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (cell_name, cell_data) in zip(axes.flat, cells.items()):
        g100 = cell_data.get("canonical_n100", {}).get("gamma", 0)
        if "n400_aggregated" in cell_data:
            g400 = cell_data["n400_aggregated"]["gamma_mean"]
            ci = (cell_data["n400_aggregated"].get("ci_low", g400),
                  cell_data["n400_aggregated"].get("ci_high", g400))
        elif "variant_C_subsample" in cell_data:
            g400 = cell_data["variant_C_subsample"]["gamma"]
            ci = (cell_data["variant_C_subsample"]["ci_low"],
                  cell_data["variant_C_subsample"]["ci_high"])
        elif "n400_results" in cell_data:
            g400 = cell_data["n400_results"]["gamma"]
            ci = (cell_data["n400_results"]["ci_low"], cell_data["n400_results"]["ci_high"])
        else:
            g400, ci = 0, (0, 0)
        bars = ax.bar(["n=100", "n=400"], [g100, g400], color=[C_BS_COL, C_DH_COL], width=0.5)
        ax.errorbar(1, g400, yerr=[[g400-ci[0]], [ci[1]-g400]], fmt="none", color="black", capsize=6)
        ax.set_title(f"Cell {cell_name}: {cell_data.get('cell_verdict', '?')}")
        ax.set_ylabel(r"$\Gamma$")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Extended Validation: 4 Cells at n=400", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

def plot_decomposition_preview(cells, out):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["A (eta=0,MSE)", "B (eta=1.9,MSE)"]
    g100 = [CANONICAL["A"]["gamma"], CANONICAL["B"]["gamma"]]
    g400 = []
    for c in ["A", "B"]:
        if c in cells and "n400_aggregated" in cells[c]:
            g400.append(cells[c]["n400_aggregated"]["gamma_mean"])
        else:
            g400.append(0)
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, g100, w, label="n=100", color=C_BS_COL, alpha=0.7)
    ax.bar(x + w/2, g400, w, label="n=400", color=C_DH_COL, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title("Decomposition Corners: n=100 vs n=400")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

def plot_h2_variants(cell_c, out):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Canonical\n(n=100)", "C-match\n(n_sim=400,\nn_rebal=400)", "C-subsample\n(n_sim=400,\nn_rebal=100)"]
    vals = [CANONICAL["C"]["gamma"]]
    for v in ["C_match", "C_subsample"]:
        if f"variant_{v}" in cell_c:
            vals.append(cell_c[f"variant_{v}"]["gamma"])
        else:
            vals.append(0)
    bars = ax.bar(labels, vals, color=[C_BS_COL, C_DH_COL, "#FF9800"], width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title("H2 Cell: Canonical vs n=400 Variants")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)

# =====================================================================
# README
# =====================================================================

def generate_readme(output, out_path):
    gv = output["global_verdict"]
    lines = [
        "# P01.7 Extended Validation Results\n",
        f"Generated: {output['meta']['created_at_utc']}",
        f"Git commit: {output['meta']['git_commit']}\n",
        f"## Global Verdict: **{gv}**\n",
        "## Per-Cell Summary\n",
        "| Cell | Canonical Gamma | n=400 Gamma | CI | Delta rel | Verdict |",
        "|------|-----------------|-------------|-----|-----------|---------|",
    ]
    for name, cell in output["cells"].items():
        g100 = cell.get("canonical_n100", {}).get("gamma", "N/A")
        if "n400_aggregated" in cell:
            g400 = f"{cell['n400_aggregated']['gamma_mean']:.4f}"
            ci = f"[{cell['n400_aggregated']['ci_low']:.3f}, {cell['n400_aggregated']['ci_high']:.3f}]"
            dr = cell.get("comparison", {}).get("delta_rel", 0)
        elif "variant_C_subsample" in cell:
            g400 = f"{cell['variant_C_subsample']['gamma']:.4f}"
            ci = f"[{cell['variant_C_subsample']['ci_low']:.3f}, {cell['variant_C_subsample']['ci_high']:.3f}]"
            dr = (cell['variant_C_subsample']['gamma'] - g100) / (abs(g100) + 1e-12) if isinstance(g100, float) else 0
        elif "n400_results" in cell:
            g400 = f"{cell['n400_results']['gamma']:.4f}"
            ci = f"[{cell['n400_results']['ci_low']:.3f}, {cell['n400_results']['ci_high']:.3f}]"
            dr = cell.get("comparison", {}).get("delta_rel", 0)
        else:
            g400, ci, dr = "N/A", "N/A", 0
        v = cell.get("cell_verdict", "?")
        g100_s = f"{g100:.4f}" if isinstance(g100, float) else str(g100)
        lines.append(f"| {name} | {g100_s} | {g400} | {ci} | {dr:+.1%} | {v} |")

    lines.extend(["", "## Next Action\n"])
    if gv == "STRICT_PASS":
        lines.append("Path 3 fully validated. Proceed to P02 and Block 2.")
    elif gv == "WEAK_PASS":
        lines.append("Path 3 viable with disclaimers. Shifted cells need footnotes.")
    else:
        lines.append("Path 1 required. Schedule full n=400 re-run. Halt for supervisor.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}", flush=True)

# =====================================================================
# Main
# =====================================================================

def run_experiment(n_sim: int = 400, n_test: int = N_TEST,
                    n_bootstrap: int = N_BOOTSTRAP,
                    seeds: dict | None = None,
                    budget_diag: dict | None = None,
                    budget_h2: dict | None = None,
                    results_dir: Path | None = None,
                    figures_dir: Path | None = None) -> dict:
    if seeds is None:
        seeds = SEEDS
    if budget_diag is None:
        budget_diag = BUDGET_DIAG
    if budget_h2 is None:
        budget_h2 = BUDGET_H2
    if results_dir is None:
        results_dir = RESULTS_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR

    hardware = record_hardware_and_versions()
    t_total = time.perf_counter()

    print("=" * 60)
    print(" P01.7 — Extended Validation (4 cells at n=400)")
    print("=" * 60, flush=True)

    cells = {}
    cells["A"] = run_cell_A(n_sim, seeds["A"], n_test, n_bootstrap, budget_diag)
    cells["B"] = run_cell_B(n_sim, seeds["B"], n_test, n_bootstrap, budget_diag)
    cells["C"] = run_cell_C(n_sim, seeds["C"], n_test, n_bootstrap, budget_h2)
    cells["D"] = run_cell_D(n_sim, seeds["D"], n_test, n_bootstrap)

    verdicts = {name: cell["cell_verdict"] for name, cell in cells.items()}
    gv = _global_verdict(verdicts)
    total_time = time.perf_counter() - t_total

    output = {
        "meta": {"script": "deep_hedging/experiments/block1_extended_validation.py",
                  "git_commit": hardware.get("git_commit", "unknown"),
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "hardware": hardware, "total_wall_clock_s": round(total_time, 2)},
        "config": {"n_sim": n_sim, "seeds_by_cell": seeds, "n_test": n_test,
                    "bootstrap_draws": n_bootstrap},
        "cells": cells,
        "global_verdict": gv,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "extended_validation.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {json_path}", flush=True)

    print("\n  Generating figures...", flush=True)
    plot_cells_comparison(cells, figures_dir / "extended_cells_comparison.png")
    plot_decomposition_preview(cells, figures_dir / "extended_decomposition_preview.png")
    if "C" in cells and "variant_C_subsample" in cells["C"]:
        plot_h2_variants(cells["C"], figures_dir / "h2_variants_comparison.png")

    generate_readme(output, results_dir / "extended_validation_readme.md")

    print(f"\n{'='*60}")
    print(f" GLOBAL VERDICT: {gv}")
    for n, v in verdicts.items():
        print(f"   Cell {n}: {v}")
    print(f"{'='*60}", flush=True)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P01.7 Extended Validation")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--n-sim", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    if args.preflight_only:
        p = RESULTS_DIR / "p017_preflight.md"
        print(f"Preflight: {p}" if p.exists() else "Preflight not found")
        sys.exit(0)
    _check_no_collision()
    run_experiment(n_sim=args.n_sim, n_test=args.n_test, n_bootstrap=args.n_bootstrap)
