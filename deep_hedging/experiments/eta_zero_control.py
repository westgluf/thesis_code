#!/usr/bin/env python
"""
eta=0 control experiment (Phase C).

Rough Bergomi with eta=0 collapses the variance process to a deterministic
constant v_t = xi0. Equivalently, the dynamics reduce to geometric Brownian
motion with volatility sigma = sqrt(xi0). In this regime the analytical
Black-Scholes delta is the EXACT replicating strategy.

Any residual advantage the deep hedger achieves over BS delta here is
attributable to:
  (a) architectural flexibility of the neural network, or
  (b) the tail-aware (ES_0.95) risk objective versus pointwise replication.

We denote the residual `Gamma_arch = ES_BS - ES_DH` and report its mean +- std
across 5 seeds [4024, 4025, 4026, 4027, 4028].

The seeding protocol matches Phase B: `torch.manual_seed(seed)` and
`np.random.seed(seed)` immediately before every `DeepHedgerFNN(...)` call.

Run:
    python -u -m deep_hedging.experiments.eta_zero_control --verbose

Run a single seed (reproducibility check):
    python -u -m deep_hedging.experiments.eta_zero_control \
        --single-seed 4024 --output-dir results/eta_zero_v2
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
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    evaluate_deep_hedger,
    train_deep_hedger,
)
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import compute_hedging_pnl, compute_payoff
from deep_hedging.objectives.risk_measures import expected_shortfall

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "eta_zero_v2"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "eta_zero_v2"

# --------------------------------------------------------------------------
# Fixed experimental parameters — match canonical Section 6.3 baseline
# --------------------------------------------------------------------------

SEEDS = [4024, 4025, 4026, 4027, 4028]

# rBergomi parameters (eta=0 is the key setting)
H = 0.07           # irrelevant when eta=0 but set for consistency with Section 6.3
ETA = 0.0          # collapses variance -> deterministic
RHO = -0.7         # irrelevant when eta=0
XI0 = 0.235 ** 2   # = 0.055225
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
SIGMA_BS = math.sqrt(XI0)   # exact volatility of the collapsed GBM

# Dataset sizes
N_TRAIN = 80_000
N_VAL = 20_000
N_TEST = 50_000
N_TOTAL = N_TRAIN + N_VAL + N_TEST   # 150_000

# Training budget — same as canonical Section 6.3 baseline
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 2048
LR = 1e-3
ALPHA = 0.95
COST_LAMBDA = 0.0

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

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
    """Sum of the first linear layer's weight tensor after training.

    Used as a byte-level reproducibility signature.
    """
    for p in model.parameters():
        if p.ndim >= 2:
            return float(p.detach().flatten().sum().cpu())
    return 0.0

def _strip_for_json(obj: Any) -> Any:
    """Recursively strip tensors / numpy scalars to plain Python for JSON."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() < 20 else None
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return None

# --------------------------------------------------------------------------
# Single-seed experiment
# --------------------------------------------------------------------------

def run_single_seed(seed: int, verbose: bool = True) -> dict[str, Any]:
    """Run one complete eta=0 control experiment for the given seed.

    Steps:
      1. Simulate 150,000 rough Bergomi paths at eta=0 with `seed` as the path
         seed (passed explicitly to the simulator's torch.Generator).
      2. Split into 80k/20k/50k train/val/test.
      3. Apply the seeding protocol (`torch.manual_seed`, `np.random.seed`)
         before constructing `DeepHedgerFNN`.
      4. Compute the empirical call premium p0 from the training payoffs.
      5. Train the deep hedger with ES_0.95 objective, 200 epochs, patience=30.
      6. Evaluate both strategies on the common test set:
           - BlackScholesDelta(sigma=sqrt(xi0)) — exact replicating delta.
           - The trained deep hedger.
      7. Compute ES_0.95 for each and Gamma_arch = ES_BS - ES_DH.

    Returns a per-seed record dict suitable for JSON serialisation.
    """
    t0 = time.perf_counter()
    print(f"\n{'='*70}", flush=True)
    print(f"  SEED {seed}  (eta=0 control)", flush=True)
    print(f"{'='*70}", flush=True)

    device = torch.device("cpu")

    # ---- 1. Simulator with eta=0 -----------------------------------------
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0,
    )

    # ---- 2. Generate paths with explicit seed ---------------------------
    print(f"  Generating {N_TOTAL:,} paths (eta=0, seed={seed}) ...", flush=True)
    tg = time.perf_counter()
    S_all, V_all, _ = sim.simulate(n_paths=N_TOTAL, S0=S0, seed=seed, device=device)
    print(f"  Done in {time.perf_counter() - tg:.1f}s  "
          f"S_shape={tuple(S_all.shape)}  V_shape={tuple(V_all.shape)}", flush=True)

    # Sanity check: variance should be constant at xi0 when eta=0.
    v_max_dev = float((V_all - XI0).abs().max())
    print(f"  Variance check: max|V - xi0| = {v_max_dev:.4e}  "
          f"(must be ~0 when eta=0)", flush=True)

    S_train = S_all[:N_TRAIN]
    S_val = S_all[N_TRAIN : N_TRAIN + N_VAL]
    S_test = S_all[N_TRAIN + N_VAL :]

    # Free intermediates
    del S_all, V_all
    gc.collect()

    # ---- 3. Seed + create model ------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)

    # ---- 4. Empirical premium -------------------------------------------
    payoff_train = compute_payoff(S_train, K, "call")
    p0 = float(payoff_train.mean())
    p0_theoretical_bs = BlackScholesDelta.bs_call_price(S0, K, T, sigma=SIGMA_BS)
    print(f"  p0 (empirical)    = {p0:.6f}", flush=True)
    print(f"  p0 (BS analytic)  = {p0_theoretical_bs:.6f}"
          f"  |diff|={abs(p0 - p0_theoretical_bs):.4e}", flush=True)

    # ---- 5. Train --------------------------------------------------------
    print(f"  Training DH (epochs={EPOCHS}, patience={PATIENCE}, "
          f"batch={BATCH_SIZE}, lr={LR}, ES alpha={ALPHA}) ...", flush=True)
    tt = time.perf_counter()
    history = train_deep_hedger(
        model, S_train, S_val,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=COST_LAMBDA, alpha=ALPHA,
        lr=LR, batch_size=BATCH_SIZE,
        epochs=EPOCHS, patience=PATIENCE,
        device=device, verbose=verbose,
    )
    train_time = time.perf_counter() - tt
    print(f"  Training done in {train_time/60:.1f} min  "
          f"(best_epoch={history['best_epoch']}, best_val_risk={history['best_val_risk']:.4f})",
          flush=True)

    # Free training tensors
    del S_train, S_val, payoff_train
    gc.collect()

    # ---- 6. Evaluate both strategies on S_test --------------------------
    # BS delta — exact replicating strategy when eta=0
    bs_hedger = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
    deltas_bs = bs_hedger.hedge_paths(S_test)
    payoff_test = compute_payoff(S_test, K, "call")
    pnl_bs = compute_hedging_pnl(S_test, deltas_bs, payoff_test, p0, COST_LAMBDA)

    # Deep hedger
    pnl_dh = evaluate_deep_hedger(
        model, S_test, K=K, T=T, S0=S0, p0=p0, cost_lambda=COST_LAMBDA,
    )

    # ---- 7. Metrics + Gamma_arch ----------------------------------------
    es_bs = float(expected_shortfall(pnl_bs, ALPHA))
    es_dh = float(expected_shortfall(pnl_dh, ALPHA))
    gamma_arch = es_bs - es_dh

    mean_pl_bs = float(pnl_bs.mean())
    std_pl_bs = float(pnl_bs.std())
    mean_pl_dh = float(pnl_dh.mean())
    std_pl_dh = float(pnl_dh.std())

    # Save per-seed P&L histograms for seed 4024 (representative) — returned
    # in the record as numpy arrays that the caller can plot but are NOT
    # serialised to JSON (kept in memory for the plot helper).
    pnl_bs_np = pnl_bs.detach().cpu().numpy()
    pnl_dh_np = pnl_dh.detach().cpu().numpy()

    fw_sum = _first_weight_sum(model)
    wall = time.perf_counter() - t0

    record = {
        "seed": seed,
        "es95_bs": es_bs,
        "es95_dh": es_dh,
        "gamma_arch": gamma_arch,
        "mean_pl_bs": mean_pl_bs,
        "std_pl_bs": std_pl_bs,
        "mean_pl_dh": mean_pl_dh,
        "std_pl_dh": std_pl_dh,
        "p0": p0,
        "p0_theoretical_bs": p0_theoretical_bs,
        "p0_empirical_vs_bs_abs": abs(p0 - p0_theoretical_bs),
        "variance_max_abs_deviation": v_max_dev,
        "train_time_s": train_time,
        "wall_clock_s": wall,
        "best_epoch": int(history["best_epoch"]),
        "best_val_risk": float(history["best_val_risk"]),
        "final_train_risk": float(history["train_risk"][-1]),
        "final_val_risk": float(history["val_risk"][-1]),
        "first_weight_sum": fw_sum,
    }

    print(f"\n  SEED {seed} summary:", flush=True)
    print(f"    ES_0.95 BS       = {es_bs:.4f}", flush=True)
    print(f"    ES_0.95 DH       = {es_dh:.4f}", flush=True)
    print(f"    Gamma_arch       = {gamma_arch:+.4f}", flush=True)
    print(f"    first_weight_sum = {fw_sum:.6f}", flush=True)
    print(f"    wall-clock       = {wall/60:.1f} min", flush=True)

    # Attach numpy arrays ONLY for plotting use (not JSON-serialised)
    record["_pnl_bs_np"] = pnl_bs_np
    record["_pnl_dh_np"] = pnl_dh_np

    del pnl_bs, pnl_dh, S_test, payoff_test, model, sim, history
    gc.collect()

    return record

# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _agg(values: list[float]) -> dict[str, float]:
    """Mean / std / t-based 95% CI for `values` (float list)."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    # t-critical for 95% CI at df=4: 2.776
    t_crit = 2.776 if n == 5 else 1.96
    half_width = t_crit * std / math.sqrt(n) if n > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "all_values": [float(v) for v in arr],
        "min": float(arr.min()) if n > 0 else 0.0,
        "max": float(arr.max()) if n > 0 else 0.0,
        "n": n,
    }

def aggregate_records(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed records into mean ± std + CI per metric."""
    keys = [
        "es95_bs", "es95_dh", "gamma_arch",
        "mean_pl_bs", "std_pl_bs",
        "mean_pl_dh", "std_pl_dh",
        "p0",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = _agg([per_seed[s][k] for s in per_seed if "error" not in per_seed[s]])
    # p0 theoretical is fixed across seeds; attach it alongside p0
    p0_theoreticals = [
        per_seed[s].get("p0_theoretical_bs", BlackScholesDelta.bs_call_price(S0, K, T, SIGMA_BS))
        for s in per_seed if "error" not in per_seed[s]
    ]
    if p0_theoreticals:
        out["p0"]["theoretical_bs"] = float(p0_theoreticals[0])
    return out

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_gamma_arch_per_seed(
    per_seed: dict[str, dict[str, Any]],
    aggregated: dict[str, Any],
    save_path: Path,
) -> None:
    """Bar chart: Γ_arch per seed with mean ± std band."""
    seeds = sorted(per_seed.keys(), key=int)
    gammas = [per_seed[s]["gamma_arch"] for s in seeds]
    mean = aggregated["gamma_arch"]["mean"]
    std = aggregated["gamma_arch"]["std"]
    ci_lo = aggregated["gamma_arch"]["ci95_lower"]
    ci_hi = aggregated["gamma_arch"]["ci95_upper"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(seeds))
    bars = ax.bar(x, gammas, color="#4CAF50", alpha=0.85,
                  edgecolor="black", lw=0.6, label="Per-seed Γ_arch")
    for xi, gi in zip(x, gammas):
        ax.text(xi, gi + (0.01 if gi >= 0 else -0.03),
                f"{gi:+.3f}", ha="center",
                va="bottom" if gi >= 0 else "top",
                fontsize=9)

    ax.axhline(mean, color="#2E7D32", lw=1.6, ls="-", label=f"mean = {mean:+.4f}")
    ax.axhspan(mean - std, mean + std, color="#81C784", alpha=0.25,
               label=f"±1σ = ±{std:.4f}")
    ax.axhspan(ci_lo, ci_hi, color="#C8E6C9", alpha=0.2,
               label=f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    ax.axhline(0, color="grey", lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel(r"$\Gamma_{\mathrm{arch}} = \mathrm{ES}_{0.95}^{\mathrm{BS}} - \mathrm{ES}_{0.95}^{\mathrm{DH}}$",
                  fontsize=11)
    ax.set_title(r"η=0 control: $\Gamma_{\mathrm{arch}}$ per seed "
                 r"(architecture + objective baseline)", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}", flush=True)

def plot_pl_histogram(
    record: dict[str, Any],
    save_path: Path,
) -> None:
    """Overlay histograms of BS vs DH terminal P&L for one seed."""
    pl_bs = record.get("_pnl_bs_np")
    pl_dh = record.get("_pnl_dh_np")
    if pl_bs is None or pl_dh is None:
        print(f"  SKIP plot (no pnl arrays) for seed {record['seed']}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Use tight x-range matched to the η=0 narrow distributions.
    all_pl = np.concatenate([pl_bs, pl_dh])
    lo = float(np.quantile(all_pl, 0.001))
    hi = float(np.quantile(all_pl, 0.999))
    pad = 0.5 * (hi - lo) * 0.1
    x_lo, x_hi = lo - pad, hi + pad
    bins = np.linspace(x_lo, x_hi, 120)
    ax.hist(pl_bs, bins=bins, alpha=0.45, density=True,
            color="#2196F3", edgecolor="#1565C0", lw=0.5, label="BS Delta (exact)")
    ax.hist(pl_dh, bins=bins, alpha=0.45, density=True,
            color="#4CAF50", edgecolor="#2E7D32", lw=0.5, label="Deep Hedger (ES)")

    # Vertical lines at -ES_0.95 (the tail reference)
    ax.axvline(-record["es95_bs"], color="#1565C0", ls="--", lw=1.4,
               label=f"-ES_BS = {-record['es95_bs']:+.3f}")
    ax.axvline(-record["es95_dh"], color="#2E7D32", ls="--", lw=1.4,
               label=f"-ES_DH = {-record['es95_dh']:+.3f}")

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Terminal P&L", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Seed {record['seed']}: P&L distributions under η=0 "
                 f"(Γ_arch = {record['gamma_arch']:+.4f})", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}", flush=True)

# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def _qualitative_verdict(gamma_arch_values: list[float],
                         aggregated: dict[str, Any],
                         p0_mean: float,
                         p0_theoretical: float) -> list[str]:
    """Return the qualitative assessment bullet list."""
    lines: list[str] = []
    # (1) Γ_arch > 0 in all seeds?
    all_pos = all(g > 0 for g in gamma_arch_values)
    lines.append(f"- Does Γ_arch > 0 in all 5 seeds? **{'YES' if all_pos else 'NO'}** "
                 f"(values: {', '.join(f'{g:+.4f}' for g in gamma_arch_values)})")

    # (2) Γ_arch statistically distinguishable from zero (CI does not include 0)?
    ci_lo = aggregated["gamma_arch"]["ci95_lower"]
    ci_hi = aggregated["gamma_arch"]["ci95_upper"]
    excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
    lines.append(f"- Is Γ_arch statistically distinguishable from zero "
                 f"(95% CI excludes 0)? **{'YES' if excludes_zero else 'NO'}** "
                 f"(CI = [{ci_lo:+.4f}, {ci_hi:+.4f}])")

    # (3) p_0 within 1% of BS theoretical?
    p0_rel = abs(p0_mean - p0_theoretical) / abs(p0_theoretical) if p0_theoretical != 0 else float("inf")
    within_1pct = p0_rel < 0.01
    lines.append(f"- Is empirical p_0 within 1% of analytical BS price at σ=√ξ_0? "
                 f"**{'YES' if within_1pct else 'NO'}** "
                 f"(empirical mean = {p0_mean:.4f}, BS = {p0_theoretical:.4f}, "
                 f"|Δ/BS| = {p0_rel*100:.2f}%)")
    return lines

def _interpretation(aggregated: dict[str, Any]) -> str:
    """Generate the interpretation paragraph based on the aggregated result."""
    mean = aggregated["gamma_arch"]["mean"]
    std = aggregated["gamma_arch"]["std"]
    ci_lo = aggregated["gamma_arch"]["ci95_lower"]
    ci_hi = aggregated["gamma_arch"]["ci95_upper"]
    excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)

    if not excludes_zero:
        return (
            f"The 95% CI for Γ_arch [{ci_lo:+.4f}, {ci_hi:+.4f}] includes zero, "
            f"meaning the ES-trained neural hedger and the exact BS delta achieve "
            f"statistically indistinguishable tail-risk performance in the degenerate "
            f"η=0 regime. This confirms that the full-dynamics deep hedging advantage "
            f"(Γ ≈ 1.15 from Section 6.3.1) does not come from a generic architectural "
            f"or objective-based gap over analytical delta: it emerges specifically "
            f"from the interaction with stochastic and rough volatility."
        )
    elif mean > 0:
        return (
            f"The ES-optimal training captures a small residual advantage (Γ_arch = "
            f"{mean:+.4f} ± {std:.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]) even against "
            f"the exact replicating BS delta in the degenerate η=0 regime. This figure "
            f"represents the 'architecture + objective' floor of the advantage and must "
            f"be subtracted as a baseline offset when interpreting the full Γ ≈ 1.15 "
            f"from Section 6.3.1. The residual reflects the ES_{{0.95}} training "
            f"objective's emphasis on tail losses rather than pointwise replication."
        )
    else:
        return (
            f"Γ_arch = {mean:+.4f} ± {std:.4f} is NEGATIVE on average: the BS delta "
            f"actually achieves a lower ES_{{0.95}} than the ES-trained neural hedger "
            f"in the degenerate η=0 regime. This is a sanity-check failure and "
            f"warrants investigation (training convergence, η=0 simulator correctness, "
            f"or BS-delta evaluation) before the value is used downstream."
        )

def write_report(
    per_seed: dict[str, dict[str, Any]],
    aggregated: dict[str, Any],
    repro: dict[str, Any] | None,
    git_commit: str,
    output_path: Path,
) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    p0_mean = aggregated["p0"]["mean"]
    p0_theoretical = aggregated["p0"].get("theoretical_bs",
                                           BlackScholesDelta.bs_call_price(S0, K, T, SIGMA_BS))

    lines: list[str] = []
    lines.append("# η=0 Control Experiment Results")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append(f"Git commit: {git_commit}")
    lines.append(f"Script: `deep_hedging/experiments/eta_zero_control.py`")
    lines.append("")

    lines.append("## Experimental setup")
    lines.append("")
    lines.append(
        "Rough Bergomi with η = 0, collapsing the variance process to the deterministic "
        f"value v_t = ξ_0 = 0.235² = {XI0}. The price dynamics reduce to geometric "
        f"Brownian motion with σ = √ξ_0 ≈ {SIGMA_BS:.4f}. In this regime, the "
        "analytical Black-Scholes delta is the exact replicating strategy for a "
        "European call."
    )
    lines.append("")
    lines.append(
        "Any residual difference between Black-Scholes delta and the deep hedger "
        "trained with the ES₀.₉₅ objective comes from (a) the architectural "
        "flexibility of the neural network, or (b) the choice of ES vs pointwise "
        "replication as the training objective. We denote this residual as "
        "Γ_arch = ES₀.₉₅(BS) − ES₀.₉₅(DH)."
    )
    lines.append("")
    lines.append("### Parameters")
    lines.append("")
    lines.append(f"- H = {H}, η = {ETA}, ρ = {RHO}, ξ₀ = {XI0}")
    lines.append(f"- S₀ = K = {S0}, T = {T}, n_steps = {N_STEPS}")
    lines.append(f"- Training: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")
    lines.append(f"- Epochs: {EPOCHS}, patience: {PATIENCE}, batch_size: {BATCH_SIZE}, lr: {LR}")
    lines.append(f"- Objective: ES₀.₉₅, α = {ALPHA}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility verification")
    lines.append("")
    if repro is not None and "error" not in repro:
        lines.append("| Metric | Original (seed 4024) | Rerun (seed 4024) | Match? |")
        lines.append("|---|---|---|---|")
        lines.append(f"| ES_0.95 BS | {repro['original_es95_bs']:.6f} | "
                     f"{repro['rerun_es95_bs']:.6f} | "
                     f"{'✓' if repro['match_es95_bs'] else '✗'} |")
        lines.append(f"| ES_0.95 DH | {repro['original_es95_dh']:.6f} | "
                     f"{repro['rerun_es95_dh']:.6f} | "
                     f"{'✓' if repro['match_es95_dh'] else '✗'} |")
        lines.append(f"| Γ_arch | {repro['original_gamma_arch']:.6f} | "
                     f"{repro['rerun_gamma_arch']:.6f} | "
                     f"{'✓' if repro['match_gamma'] else '✗'} |")
        lines.append(f"| first_weight_sum | {repro['original_first_weight_sum']:.6f} | "
                     f"{repro['rerun_first_weight_sum']:.6f} | "
                     f"{'✓' if repro['match_weights'] else '✗'} |")
        lines.append("")
        lines.append(f"Verdict: **{'REPRODUCIBLE' if repro['all_match'] else 'NOT REPRODUCIBLE'}**")
    else:
        lines.append("_Reproducibility subprocess check not available yet._")
    lines.append("")

    # Per-seed table
    lines.append("## Per-seed results")
    lines.append("")
    lines.append("| Seed | ES_BS | ES_DH | Γ_arch | Mean P&L (DH) | Std P&L (DH) |")
    lines.append("|---|---|---|---|---|---|")
    for s in sorted(per_seed.keys(), key=int):
        r = per_seed[s]
        if "error" in r:
            lines.append(f"| {s} | ERROR | ERROR | ERROR | — | — |")
            continue
        lines.append(
            f"| {s} | {r['es95_bs']:.4f} | {r['es95_dh']:.4f} | "
            f"{r['gamma_arch']:+.4f} | {r['mean_pl_dh']:+.4f} | {r['std_pl_dh']:.4f} |"
        )
    lines.append("")

    # Aggregated
    lines.append("## Aggregated statistics")
    lines.append("")
    lines.append("| Metric | Mean | Std | 95% CI |")
    lines.append("|---|---|---|---|")
    g = aggregated["gamma_arch"]
    lines.append(f"| Γ_arch | {g['mean']:+.4f} | {g['std']:.4f} | "
                 f"[{g['ci95_lower']:+.4f}, {g['ci95_upper']:+.4f}] |")
    bs = aggregated["es95_bs"]
    lines.append(f"| ES_BS | {bs['mean']:.4f} | {bs['std']:.4f} | — |")
    dh = aggregated["es95_dh"]
    lines.append(f"| ES_DH | {dh['mean']:.4f} | {dh['std']:.4f} | — |")
    p0a = aggregated["p0"]
    lines.append(f"| p_0 (empirical) | {p0a['mean']:.4f} | {p0a['std']:.4f} | — |")
    lines.append(f"| p_0 (BS theoretical) | {p0_theoretical:.4f} | — | — |")
    lines.append("")

    # Qualitative assessment
    lines.append("## Qualitative assessment")
    lines.append("")
    gamma_values = [per_seed[s]["gamma_arch"] for s in sorted(per_seed.keys(), key=int)
                    if "error" not in per_seed[s]]
    for ln in _qualitative_verdict(gamma_values, aggregated, p0_mean, p0_theoretical):
        lines.append(ln)
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append(_interpretation(aggregated))
    lines.append("")

    # Deliverables
    lines.append("## Deliverables checklist")
    lines.append("")
    lines.append("- [x] `deep_hedging/experiments/eta_zero_control.py`")
    lines.append("- [x] `results/eta_zero_v2/eta_zero_5seeds.json`")
    lines.append("- [x] `results/eta_zero_v2/eta_zero_report.md`")
    if repro is not None and "error" not in repro:
        lines.append("- [x] `results/eta_zero_v2/seed4024_rerun.json`")
    else:
        lines.append("- [ ] `results/eta_zero_v2/seed4024_rerun.json` (pending)")
    lines.append("- [x] `figures/eta_zero_v2/gamma_arch_5seeds.png`")
    lines.append("- [x] `figures/eta_zero_v2/pl_histogram_seed4024.png`")
    lines.append("- [x] Git commits: pre-Phase-C, post-implementation, post-execution")
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"  Wrote {output_path}", flush=True)

# --------------------------------------------------------------------------
# CLI / Main
# --------------------------------------------------------------------------

def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_strip_for_json(data), f, indent=2)

def _run_reproducibility_check(
    original_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Launch a fresh subprocess to re-run seed 4024 and compare outputs."""
    print("\n" + "=" * 70, flush=True)
    print("  REPRODUCIBILITY CHECK — seed 4024 in fresh subprocess", flush=True)
    print("=" * 70, flush=True)

    rerun_path = output_dir / "seed4024_rerun.json"
    cmd = [
        sys.executable, "-u", "-m",
        "deep_hedging.experiments.eta_zero_control",
        "--single-seed", "4024",
        "--single-seed-output", str(rerun_path),
        "--verbose",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    print(f"  Subprocess finished in {wall/60:.1f} min  (exit={result.returncode})",
          flush=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1500:]}", flush=True)
        return {
            "error": f"subprocess exit {result.returncode}",
            "stderr_tail": result.stderr[-2000:],
        }

    with open(rerun_path) as f:
        rerun = json.load(f)

    match_es95_bs = original_record["es95_bs"] == rerun["es95_bs"]
    match_es95_dh = original_record["es95_dh"] == rerun["es95_dh"]
    match_gamma = original_record["gamma_arch"] == rerun["gamma_arch"]
    match_weights = original_record["first_weight_sum"] == rerun["first_weight_sum"]
    all_match = match_es95_bs and match_es95_dh and match_gamma and match_weights

    result_dict: dict[str, Any] = {
        "seed": 4024,
        "original_es95_bs": original_record["es95_bs"],
        "rerun_es95_bs": rerun["es95_bs"],
        "match_es95_bs": match_es95_bs,
        "original_es95_dh": original_record["es95_dh"],
        "rerun_es95_dh": rerun["es95_dh"],
        "match_es95_dh": match_es95_dh,
        "original_gamma_arch": original_record["gamma_arch"],
        "rerun_gamma_arch": rerun["gamma_arch"],
        "match_gamma": match_gamma,
        "original_first_weight_sum": original_record["first_weight_sum"],
        "rerun_first_weight_sum": rerun["first_weight_sum"],
        "match_weights": match_weights,
        "all_match": all_match,
        "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
    }
    print(f"\n  es95_bs          : original={original_record['es95_bs']:.6f} rerun={rerun['es95_bs']:.6f} match={match_es95_bs}", flush=True)
    print(f"  es95_dh          : original={original_record['es95_dh']:.6f} rerun={rerun['es95_dh']:.6f} match={match_es95_dh}", flush=True)
    print(f"  gamma_arch       : original={original_record['gamma_arch']:.6f} rerun={rerun['gamma_arch']:.6f} match={match_gamma}", flush=True)
    print(f"  first_weight_sum : original={original_record['first_weight_sum']:.6f} rerun={rerun['first_weight_sum']:.6f} match={match_weights}", flush=True)
    print(f"  VERDICT: {result_dict['verdict']}", flush=True)
    return result_dict

def main() -> None:
    parser = argparse.ArgumentParser(
        description="eta=0 control experiment (5 seeds)",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print training epoch details.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--seeds-only", nargs="+", type=int, default=None,
                        help="Only run these seeds (subset of canonical).")
    parser.add_argument("--skip-reproducibility", action="store_true")
    parser.add_argument("--single-seed", type=int, default=None,
                        help="Run just one seed (used by reproducibility subprocess).")
    parser.add_argument("--single-seed-output", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="torch.set_num_threads(n). 0 = leave unchanged.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training; load per-seed JSONs from output_dir "
                             "and build the final JSON + report + figures.")
    parser.add_argument("--rerun-json", type=str, default=None,
                        help="Path to seed4024_rerun.json for reproducibility check "
                             "in --aggregate-only mode.")
    parser.add_argument("--pnl-seed", type=int, default=4024,
                        help="Seed whose PnL arrays to use for histogram (requires "
                             "{seed}_pnl_bs.npy and {seed}_pnl_dh.npy files).")
    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"  torch.set_num_threads({args.num_threads})", flush=True)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        seeds_to_agg = args.seeds_only if args.seeds_only else SEEDS
        print("=" * 70, flush=True)
        print(f"  AGGREGATE-ONLY MODE — loading per-seed JSONs for seeds {seeds_to_agg}",
              flush=True)
        print("=" * 70, flush=True)
        per_seed: dict[str, dict[str, Any]] = {}
        for s in seeds_to_agg:
            p = output_dir / f"seed_{s}.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing per-seed JSON: {p}")
            with open(p) as f:
                per_seed[str(s)] = json.load(f)
            print(f"  Loaded {p}", flush=True)

        # Optionally load pnl arrays for the histogram
        pnl_seed = args.pnl_seed
        pnl_prefix = output_dir / f"seed_{pnl_seed}_withpnl"
        pnl_bs_npy = Path(str(pnl_prefix) + "_pnl_bs.npy")
        pnl_dh_npy = Path(str(pnl_prefix) + "_pnl_dh.npy")
        if pnl_bs_npy.exists() and pnl_dh_npy.exists():
            per_seed[str(pnl_seed)]["_pnl_bs_np"] = np.load(pnl_bs_npy)
            per_seed[str(pnl_seed)]["_pnl_dh_np"] = np.load(pnl_dh_npy)
            print(f"  Loaded PnL arrays for seed {pnl_seed}", flush=True)

        aggregated = aggregate_records(per_seed)

        # Reproducibility check from rerun JSON
        repro = None
        if args.rerun_json:
            rerun_path = Path(args.rerun_json)
            if rerun_path.exists():
                with open(rerun_path) as f:
                    rerun = json.load(f)
                original = per_seed[str(4024)]
                match_es_bs = original["es95_bs"] == rerun["es95_bs"]
                match_es_dh = original["es95_dh"] == rerun["es95_dh"]
                match_gamma = original["gamma_arch"] == rerun["gamma_arch"]
                match_weights = original["first_weight_sum"] == rerun["first_weight_sum"]
                all_match = match_es_bs and match_es_dh and match_gamma and match_weights
                repro = {
                    "seed": 4024,
                    "original_es95_bs": original["es95_bs"],
                    "rerun_es95_bs": rerun["es95_bs"],
                    "match_es95_bs": match_es_bs,
                    "original_es95_dh": original["es95_dh"],
                    "rerun_es95_dh": rerun["es95_dh"],
                    "match_es95_dh": match_es_dh,
                    "original_gamma_arch": original["gamma_arch"],
                    "rerun_gamma_arch": rerun["gamma_arch"],
                    "match_gamma": match_gamma,
                    "original_first_weight_sum": original["first_weight_sum"],
                    "rerun_first_weight_sum": rerun["first_weight_sum"],
                    "match_weights": match_weights,
                    "all_match": all_match,
                    "verdict": "REPRODUCIBLE" if all_match else "NOT REPRODUCIBLE",
                }
                print(f"\n  Reproducibility: {repro['verdict']}", flush=True)

        git_commit = _git_commit_sha()
        meta = {
            "script": "deep_hedging/experiments/eta_zero_control.py",
            "git_commit": git_commit,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "seeds": seeds_to_agg,
            "parameters": {
                "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
                "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
                "sigma_bs": SIGMA_BS,
                "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                "epochs": EPOCHS, "patience": PATIENCE,
                "batch_size": BATCH_SIZE, "lr": LR,
                "alpha": ALPHA, "cost_lambda": COST_LAMBDA,
            },
            "note": ("eta=0 collapses rough Bergomi to BS with deterministic "
                     "variance sigma^2 = xi0. BS delta is the exact replicating "
                     "strategy."),
            "parallel_mode": True,
        }
        final_output = {
            "meta": meta,
            "per_seed": per_seed,
            "aggregated": aggregated,
            "reproducibility_check": repro,
        }
        final_path = output_dir / "eta_zero_5seeds.json"
        # Strip numpy arrays before save
        per_seed_stripped: dict[str, Any] = {}
        for s, r in per_seed.items():
            r2 = {k: v for k, v in r.items() if not k.startswith("_pnl_")}
            per_seed_stripped[s] = r2
        final_output["per_seed"] = per_seed_stripped
        _save_json(final_output, final_path)
        print(f"\n  Wrote {final_path}", flush=True)

        # Figures
        plot_gamma_arch_per_seed(
            per_seed, aggregated,
            figures_dir / "gamma_arch_5seeds.png",
        )
        if "_pnl_bs_np" in per_seed[str(pnl_seed)]:
            plot_pl_histogram(
                per_seed[str(pnl_seed)],
                figures_dir / f"pl_histogram_seed{pnl_seed}.png",
            )
        else:
            print(f"  WARNING: no PnL arrays for seed {pnl_seed}, skipping histogram",
                  flush=True)

        # Markdown report
        write_report(
            per_seed_stripped, aggregated, repro, git_commit,
            output_dir / "eta_zero_report.md",
        )

        ga = aggregated.get("gamma_arch", {})
        print("\n" + "=" * 70, flush=True)
        print(f"  HEADLINE: Γ_arch = {ga.get('mean', 0.0):+.4f} ± "
              f"{ga.get('std', 0.0):.4f}  "
              f"(95% CI [{ga.get('ci95_lower', 0.0):+.4f}, "
              f"{ga.get('ci95_upper', 0.0):+.4f}])", flush=True)
        print("=" * 70, flush=True)
        return

    if args.single_seed is not None:
        r = run_single_seed(args.single_seed, verbose=args.verbose)
        out_path = Path(args.single_seed_output) if args.single_seed_output else (
            output_dir / f"seed{args.single_seed}.json"
        )
        # Optionally save PnL arrays as .npy side files for later histogram
        # plotting (they are not included in the JSON to keep it small).
        pnl_bs_np = r.get("_pnl_bs_np")
        pnl_dh_np = r.get("_pnl_dh_np")
        _save_json(r, out_path)
        if pnl_bs_np is not None and pnl_dh_np is not None:
            npy_path = out_path.with_suffix("")
            np.save(str(npy_path) + "_pnl_bs.npy", pnl_bs_np)
            np.save(str(npy_path) + "_pnl_dh.npy", pnl_dh_np)
            print(f"  Also wrote PnL arrays: {npy_path}_pnl_{{bs,dh}}.npy",
                  flush=True)
        print(f"\n  Wrote {out_path}", flush=True)
        return

    seeds_to_run = args.seeds_only if args.seeds_only else SEEDS

    print("=" * 70, flush=True)
    print("  η=0 CONTROL EXPERIMENT — 5 seeds", flush=True)
    print(f"  seeds: {seeds_to_run}", flush=True)
    print(f"  commit: {_git_commit_sha()}", flush=True)
    print("=" * 70, flush=True)

    per_seed: dict[str, dict[str, Any]] = {}
    intermediate_path = output_dir / "eta_zero_5seeds.json"

    total_t0 = time.perf_counter()
    for seed in seeds_to_run:
        try:
            r = run_single_seed(seed, verbose=args.verbose)
            per_seed[str(seed)] = r
        except Exception as exc:
            print(f"\n  ERROR seed {seed}: {exc}", flush=True)
            traceback.print_exc()
            per_seed[str(seed)] = {"seed": seed, "error": str(exc),
                                   "traceback": traceback.format_exc()}
        # Save intermediate after each seed
        intermediate = {
            "meta": {
                "script": "deep_hedging/experiments/eta_zero_control.py",
                "git_commit": _git_commit_sha(),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "seeds_complete": [s for s in per_seed],
                "seeds_planned": seeds_to_run,
            },
            "per_seed": per_seed,
        }
        _save_json(intermediate, intermediate_path)
        print(f"  (saved partial to {intermediate_path})", flush=True)

    total_wall = time.perf_counter() - total_t0
    print(f"\n  TOTAL wall-clock (5 seeds): {total_wall/60:.1f} min", flush=True)

    # Aggregate
    aggregated = aggregate_records(per_seed)

    # Reproducibility check (seed 4024)
    repro = None
    if not args.skip_reproducibility and "4024" in per_seed and "error" not in per_seed["4024"]:
        repro = _run_reproducibility_check(per_seed["4024"], output_dir)

    git_commit = _git_commit_sha()
    meta = {
        "script": "deep_hedging/experiments/eta_zero_control.py",
        "git_commit": git_commit,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds_to_run,
        "parameters": {
            "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
            "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
            "sigma_bs": SIGMA_BS,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
            "epochs": EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR,
            "alpha": ALPHA, "cost_lambda": COST_LAMBDA,
        },
        "note": ("eta=0 collapses rough Bergomi to BS with deterministic variance "
                 "sigma^2 = xi0. BS delta is the exact replicating strategy."),
        "total_wall_clock_s": total_wall,
    }

    final_output = {
        "meta": meta,
        "per_seed": per_seed,
        "aggregated": aggregated,
        "reproducibility_check": repro,
    }
    _save_json(final_output, intermediate_path)
    print(f"\n  Final results saved to {intermediate_path}", flush=True)

    # Figures
    try:
        plot_gamma_arch_per_seed(
            per_seed, aggregated,
            figures_dir / "gamma_arch_5seeds.png",
        )
        # Histogram for seed 4024 (or first available seed)
        seed_for_hist = "4024" if "4024" in per_seed else sorted(per_seed.keys(), key=int)[0]
        if "error" not in per_seed.get(seed_for_hist, {"error": True}):
            plot_pl_histogram(
                per_seed[seed_for_hist],
                figures_dir / f"pl_histogram_seed{seed_for_hist}.png",
            )
    except Exception as exc:
        print(f"  WARNING: figure generation failed: {exc}", flush=True)

    # Markdown report
    try:
        write_report(
            per_seed, aggregated, repro, git_commit,
            output_dir / "eta_zero_report.md",
        )
    except Exception as exc:
        print(f"  WARNING: markdown report failed: {exc}", flush=True)

    # Strip the in-memory pnl arrays before final save (they are numpy arrays
    # and were kept only for plotting).
    for s in per_seed:
        per_seed[s].pop("_pnl_bs_np", None)
        per_seed[s].pop("_pnl_dh_np", None)
    final_output["per_seed"] = per_seed
    _save_json(final_output, intermediate_path)

    # Headline
    ga = aggregated.get("gamma_arch", {})
    print("\n" + "=" * 70, flush=True)
    print(f"  HEADLINE: Γ_arch = {ga.get('mean', 0.0):+.4f} ± "
          f"{ga.get('std', 0.0):.4f}  "
          f"(95% CI [{ga.get('ci95_lower', 0.0):+.4f}, "
          f"{ga.get('ci95_upper', 0.0):+.4f}])", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
