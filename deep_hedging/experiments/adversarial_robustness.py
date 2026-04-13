#!/usr/bin/env python
"""
H3: Adversarial robustness to parameter perturbations.

Two complementary approaches:
  A. Infinitesimal gradient sensitivity ∇_Θ ES_95 at the calibration Θ_0.
  B. Finite-perturbation sweep on H, η, ρ separately.

The deep hedger is trained ONCE at Θ_0 (or loaded from cache) and then
stress-tested on perturbed test sets without any retraining.

Run:
    python -u -m deep_hedging.experiments.adversarial_robustness
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN, train_deep_hedger, evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics, expected_shortfall
from deep_hedging.experiments.gradient_sensitivity import (
    compute_es_gradient_bs,
    compute_es_gradient_deep,
    gradient_sensitivity_bootstrap,
    _compute_p0_at_theta,
)


# ─── Constants ─────────────────────────────────────────────────
THETA_0: dict[str, float] = dict(H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2)
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
SIGMA = float(math.sqrt(THETA_0["xi0"]))
MASTER_SEED = 2024

# Perturbation grids
EPS_H: list[float] = [-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06]
EPS_ETA: list[float] = [-1.4, -0.9, -0.4, 0.0, 0.4, 0.9, 1.4]
EPS_RHO: list[float] = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]

# Training config (fast cached baseline hedger)
HEDGER_CFG = dict(
    hidden_dim=128,
    n_res_blocks=2,
    lr=1e-3,
    batch_size=2048,
    epochs=150,
    patience=30,
)

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
HEDGER_PATH = FIGURE_DIR / "adversarial_baseline_hedger.pt"
RESULTS_PATH = FIGURE_DIR / "adversarial_robustness.json"

# Colours
C_BS = "#2196F3"
C_DEEP = "#4CAF50"
C_BAND = "#FFB74D"


# =======================================================================
# Helpers
# =======================================================================

def _strip_for_json(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return float(obj.detach().cpu()) if obj.numel() == 1 else None
    if isinstance(obj, nn.Module):
        return None
    if isinstance(obj, dict):
        return {str(k): _strip_for_json(v) for k, v in obj.items()
                if _strip_for_json(v) is not None}
    if isinstance(obj, (list, tuple)):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    try:
        return float(obj)
    except Exception:
        return None


def _clamp_param(name: str, value: float) -> float:
    """Clamp parameter to physically valid range."""
    if name == "H":
        return float(np.clip(value, 0.005, 0.495))
    if name == "eta":
        return float(np.clip(value, 0.05, 5.0))
    if name == "rho":
        return float(np.clip(value, -0.99, -0.01))
    return value


# =======================================================================
# AdversarialRobustnessExperiment
# =======================================================================

class AdversarialRobustnessExperiment:
    """H3 test via gradient sensitivity + perturbation sweep."""

    def __init__(
        self,
        figures_dir: str | Path = FIGURE_DIR,
        n_train: int = 60_000,
        n_val: int = 10_000,
        n_test_per_perturbation: int = 30_000,
        n_grad_paths: int = 50_000,
        n_grad_seeds: int = 5,
        device: torch.device | None = None,
    ) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test_per_perturbation
        self.n_grad_paths = n_grad_paths
        self.n_grad_seeds = n_grad_seeds
        self.device = device or torch.device("cpu")

        self.hedger: DeepHedgerFNN | None = None
        self.p0_calibration: float | None = None

    # ──────────────────────────────────────────────────────────
    # Baseline hedger
    # ──────────────────────────────────────────────────────────

    def setup_baseline_hedger(self) -> DeepHedgerFNN:
        """Load cached hedger if available, else train fresh and save."""
        hedger = DeepHedgerFNN(
            input_dim=4,
            hidden_dim=HEDGER_CFG["hidden_dim"],
            n_res_blocks=HEDGER_CFG["n_res_blocks"],
        )

        if HEDGER_PATH.exists():
            print(f"  Loading cached baseline hedger from {HEDGER_PATH.name}", flush=True)
            state = torch.load(HEDGER_PATH, map_location=self.device, weights_only=True)
            hedger.load_state_dict(state)
            hedger.eval()
            self.hedger = hedger.to(self.device)
            return self.hedger

        print(f"  Training baseline hedger at Theta_0 ({self.n_train} paths, "
              f"{HEDGER_CFG['epochs']} epochs) ...", flush=True)
        t0 = time.time()

        sim = DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T,
            H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
            xi0=THETA_0["xi0"],
        )
        total = self.n_train + self.n_val
        S, _, _ = sim.simulate(n_paths=total, S0=S0, seed=MASTER_SEED)
        S_train = S[: self.n_train]
        S_val = S[self.n_train :]
        del S
        gc.collect()

        payoff_train = compute_payoff(S_train, K, "call")
        p0 = float(payoff_train.mean())
        self.p0_calibration = p0

        torch.manual_seed(MASTER_SEED + 1000)
        np.random.seed(MASTER_SEED + 1000)
        train_deep_hedger(
            hedger, S_train, S_val,
            K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
            alpha=0.95,
            lr=HEDGER_CFG["lr"],
            batch_size=HEDGER_CFG["batch_size"],
            epochs=HEDGER_CFG["epochs"],
            patience=HEDGER_CFG["patience"],
            verbose=False,
        )
        elapsed = time.time() - t0
        print(f"    trained in {elapsed/60:.1f} min", flush=True)

        # Save
        torch.save(hedger.state_dict(), HEDGER_PATH)
        print(f"    saved to {HEDGER_PATH.name}", flush=True)

        del S_train, S_val
        gc.collect()
        hedger.eval()
        self.hedger = hedger.to(self.device)
        return self.hedger

    def _ensure_p0(self) -> float:
        """Compute (and cache) the calibration p_0."""
        if self.p0_calibration is None:
            print("  Computing calibration p_0 (100k paths) ...", flush=True)
            self.p0_calibration = _compute_p0_at_theta(
                H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
                xi0=THETA_0["xi0"], n_steps=N_STEPS, T=T, S0=S0, K=K,
                n_paths=100_000, seed=MASTER_SEED + 999,
                device=self.device,
            )
            print(f"    p_0 = {self.p0_calibration:.4f}", flush=True)
        return self.p0_calibration

    # ──────────────────────────────────────────────────────────
    # Approach A: gradient sensitivity
    # ──────────────────────────────────────────────────────────

    def run_gradient_sensitivity(self) -> dict[str, Any]:
        print("\n" + "=" * 65, flush=True)
        print("  APPROACH A — Gradient Sensitivity at Theta_0", flush=True)
        print("=" * 65, flush=True)
        if self.hedger is None:
            raise RuntimeError("call setup_baseline_hedger() first")

        p0 = self._ensure_p0()

        # BS gradient
        print(f"\n  BS delta: {self.n_grad_seeds} seeds x "
              f"{self.n_grad_paths} paths ...", flush=True)
        t0 = time.time()
        bs_boot = gradient_sensitivity_bootstrap(
            compute_es_gradient_bs,
            n_seeds=self.n_grad_seeds,
            seed_base=MASTER_SEED,
            H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
            xi0=THETA_0["xi0"],
            S0=S0, K=K, T=T, n_steps=N_STEPS,
            n_paths=self.n_grad_paths,
            sigma_assumed=SIGMA,
            alpha=0.95,
            cost_lambda=0.0,
            p0=p0,
        )
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
        print(f"    grad_H     = {bs_boot['mean_grad_H']:+.4f} "
              f"± {bs_boot['std_grad_H']:.4f}", flush=True)
        print(f"    grad_eta   = {bs_boot['mean_grad_eta']:+.4f} "
              f"± {bs_boot['std_grad_eta']:.4f}", flush=True)
        print(f"    grad_rho   = {bs_boot['mean_grad_rho']:+.4f} "
              f"± {bs_boot['std_grad_rho']:.4f}", flush=True)
        print(f"    L2 norm    = {bs_boot['mean_grad_l2_norm']:.4f} "
              f"± {bs_boot['std_grad_l2_norm']:.4f}", flush=True)
        print(f"    ES_95      = {bs_boot['mean_es_value']:.4f}", flush=True)

        # Deep hedger gradient
        print(f"\n  Deep hedger: {self.n_grad_seeds} seeds x "
              f"{self.n_grad_paths} paths ...", flush=True)
        t0 = time.time()
        deep_boot = gradient_sensitivity_bootstrap(
            compute_es_gradient_deep,
            n_seeds=self.n_grad_seeds,
            seed_base=MASTER_SEED,
            hedger=self.hedger,
            H=THETA_0["H"], eta=THETA_0["eta"], rho=THETA_0["rho"],
            xi0=THETA_0["xi0"],
            S0=S0, K=K, T=T, n_steps=N_STEPS,
            n_paths=self.n_grad_paths,
            alpha=0.95,
            cost_lambda=0.0,
            p0=p0,
        )
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
        print(f"    grad_H     = {deep_boot['mean_grad_H']:+.4f} "
              f"± {deep_boot['std_grad_H']:.4f}", flush=True)
        print(f"    grad_eta   = {deep_boot['mean_grad_eta']:+.4f} "
              f"± {deep_boot['std_grad_eta']:.4f}", flush=True)
        print(f"    grad_rho   = {deep_boot['mean_grad_rho']:+.4f} "
              f"± {deep_boot['std_grad_rho']:.4f}", flush=True)
        print(f"    L2 norm    = {deep_boot['mean_grad_l2_norm']:.4f} "
              f"± {deep_boot['std_grad_l2_norm']:.4f}", flush=True)
        print(f"    ES_95      = {deep_boot['mean_es_value']:.4f}", flush=True)

        # Ratios (BS / DH on |gradient| per axis)
        ratios: dict[str, float] = {}
        for k in ("H", "eta", "rho"):
            num = abs(bs_boot[f"mean_grad_{k}"])
            den = abs(deep_boot[f"mean_grad_{k}"])
            ratios[k] = num / den if den > 1e-12 else float("inf")
        ratios["l2"] = (bs_boot["mean_grad_l2_norm"]
                        / max(deep_boot["mean_grad_l2_norm"], 1e-12))

        print(f"\n  Sensitivity ratios |BS| / |DH|:", flush=True)
        for k, v in ratios.items():
            print(f"    {k:>4s}: {v:.3f}", flush=True)

        return {
            "bs": bs_boot,
            "deep": deep_boot,
            "ratio_bs_over_deep": ratios,
        }

    # ──────────────────────────────────────────────────────────
    # Approach B: perturbation sweep
    # ──────────────────────────────────────────────────────────

    def _simulate_perturbed_test_set(
        self,
        H: float, eta: float, rho: float,
        n_paths: int, seed: int,
    ) -> dict[str, Any]:
        sim = DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T, H=H, eta=eta, rho=rho, xi0=THETA_0["xi0"],
        )
        with torch.no_grad():
            S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed,
                                   device=self.device)
        return {"S": S}

    def run_perturbation_sweep_single_axis(
        self,
        axis: str,
        eps_values: list[float],
        p0_option: str = "fixed",
    ) -> dict[str, Any]:
        if axis not in ("H", "eta", "rho"):
            raise ValueError(f"unknown axis {axis!r}")
        if self.hedger is None:
            raise RuntimeError("call setup_baseline_hedger() first")

        p0_calib = self._ensure_p0()
        axis_idx = {"H": 0, "eta": 1, "rho": 2}[axis]

        out: dict[str, Any] = {
            "axis": axis,
            "eps_values": list(eps_values),
            "p0_option": p0_option,
            "p0_calibration": p0_calib,
            "p0_perturbed": {},
            "bs": {},
            "deep": {},
            "degradation_bs": {},
            "degradation_deep": {},
            "relative_degradation_bs": {},
            "relative_degradation_deep": {},
        }

        # First pass: collect ES_95 at ε=0 for relative degradation
        es_baseline_bs: float | None = None
        es_baseline_deep: float | None = None

        bs_hedger = BlackScholesDelta(sigma=SIGMA, K=K, T=T)

        for j, eps in enumerate(eps_values):
            # Build perturbed Θ
            params = dict(THETA_0)
            params[axis] = _clamp_param(axis, params[axis] + eps)

            # Diagnostic p0 at perturbed Θ
            seed = MASTER_SEED + axis_idx * 1000 + j * 10
            p0_pert = _compute_p0_at_theta(
                H=params["H"], eta=params["eta"], rho=params["rho"],
                xi0=THETA_0["xi0"],
                n_steps=N_STEPS, T=T, S0=S0, K=K,
                n_paths=20_000, seed=seed + 7,
                device=self.device,
            )
            out["p0_perturbed"][float(eps)] = p0_pert

            # Choose p0 for the actual evaluation
            p0_eval = p0_calib if p0_option == "fixed" else p0_pert

            # Generate test paths
            data = self._simulate_perturbed_test_set(
                H=params["H"], eta=params["eta"], rho=params["rho"],
                n_paths=self.n_test, seed=seed,
            )
            S = data["S"]

            # BS delta
            with torch.no_grad():
                deltas_bs = bs_hedger.hedge_paths(S)
                payoff = compute_payoff(S, K, "call")
                pnl_bs = compute_hedging_pnl(S, deltas_bs, payoff, p0_eval, 0.0)
                metrics_bs = compute_all_metrics(pnl_bs)

            # Deep hedger
            self.hedger.eval()
            with torch.no_grad():
                deltas_deep = self.hedger.hedge_paths(S, T=T, S0=S0)
                deltas_deep = deltas_deep.to(S.dtype)
                pnl_deep = compute_hedging_pnl(S, deltas_deep, payoff, p0_eval, 0.0)
                metrics_deep = compute_all_metrics(pnl_deep)

            out["bs"][float(eps)] = {
                "metrics": metrics_bs,
                "params": params,
            }
            out["deep"][float(eps)] = {
                "metrics": metrics_deep,
                "params": params,
            }

            if abs(eps) < 1e-12:
                es_baseline_bs = metrics_bs["es_95"]
                es_baseline_deep = metrics_deep["es_95"]

            print(
                f"    eps={eps:+.4f}  Theta={axis}={params[axis]:+.3f}  "
                f"p0_eval={p0_eval:.3f}  "
                f"ES_BS={metrics_bs['es_95']:.3f}  "
                f"ES_DH={metrics_deep['es_95']:.3f}",
                flush=True,
            )

            del S, data, deltas_bs, deltas_deep, pnl_bs, pnl_deep, payoff
            gc.collect()

        # Compute degradations relative to ε=0 baseline
        if es_baseline_bs is None or es_baseline_deep is None:
            raise RuntimeError("eps=0 cell missing — cannot compute degradation")

        for eps in eps_values:
            eps_f = float(eps)
            es_bs = out["bs"][eps_f]["metrics"]["es_95"]
            es_dh = out["deep"][eps_f]["metrics"]["es_95"]
            out["degradation_bs"][eps_f] = es_bs - es_baseline_bs
            out["degradation_deep"][eps_f] = es_dh - es_baseline_deep
            out["relative_degradation_bs"][eps_f] = (
                (es_bs - es_baseline_bs) / max(es_baseline_bs, 1e-12)
            )
            out["relative_degradation_deep"][eps_f] = (
                (es_dh - es_baseline_deep) / max(es_baseline_deep, 1e-12)
            )

        return out

    def run_perturbation_sweep_all_axes(
        self,
        p0_option: str = "fixed",
    ) -> dict[str, Any]:
        print("\n" + "=" * 65, flush=True)
        print("  APPROACH B — Perturbation Sweep (3 axes)", flush=True)
        print("=" * 65, flush=True)

        results: dict[str, Any] = {}

        print("\n  --- H axis ---", flush=True)
        results["H"] = self.run_perturbation_sweep_single_axis(
            "H", EPS_H, p0_option=p0_option,
        )
        print("\n  --- eta axis ---", flush=True)
        results["eta"] = self.run_perturbation_sweep_single_axis(
            "eta", EPS_ETA, p0_option=p0_option,
        )
        print("\n  --- rho axis ---", flush=True)
        results["rho"] = self.run_perturbation_sweep_single_axis(
            "rho", EPS_RHO, p0_option=p0_option,
        )
        return results

    # ──────────────────────────────────────────────────────────
    # H3 verdict
    # ──────────────────────────────────────────────────────────

    def compute_h3_verdict(
        self, gradient_results: dict, sweep_results: dict,
    ) -> dict[str, Any]:
        l2_ratio = gradient_results["ratio_bs_over_deep"]["l2"]
        per_axis_ratio = {
            k: gradient_results["ratio_bs_over_deep"][k]
            for k in ("H", "eta", "rho")
        }

        # Per-axis robustness comparison
        axis_robustness: dict[str, Any] = {}
        for axis in ("H", "eta", "rho"):
            ax_data = sweep_results[axis]
            eps_values = ax_data["eps_values"]
            dh_better_at = []
            for eps in eps_values:
                if abs(eps) < 1e-12:
                    continue
                rel_bs = abs(ax_data["relative_degradation_bs"][float(eps)])
                rel_dh = abs(ax_data["relative_degradation_deep"][float(eps)])
                dh_better_at.append(rel_dh < rel_bs)
            axis_robustness[axis] = {
                "dh_better_count": sum(dh_better_at),
                "total_perturbations": len(dh_better_at),
                "fraction_dh_wins": (sum(dh_better_at) / len(dh_better_at)
                                     if dh_better_at else 0.0),
            }

        # Largest |ε| comparison
        largest_eps_dh_wins: dict[str, bool] = {}
        for axis in ("H", "eta", "rho"):
            ax = sweep_results[axis]
            eps_max = max(abs(e) for e in ax["eps_values"])
            cands = [e for e in ax["eps_values"] if abs(e) >= eps_max - 1e-9]
            wins = []
            for e in cands:
                rel_bs = abs(ax["relative_degradation_bs"][float(e)])
                rel_dh = abs(ax["relative_degradation_deep"][float(e)])
                wins.append(rel_dh < rel_bs - 0.05)  # >5% margin
            largest_eps_dh_wins[axis] = all(wins) if wins else False

        # Verdict
        n_axes_dh_robust = sum(1 for v in axis_robustness.values()
                               if v["fraction_dh_wins"] >= 0.5)
        n_axes_largest_eps_wins = sum(1 for v in largest_eps_dh_wins.values() if v)

        if (l2_ratio > 1.3 and n_axes_dh_robust >= 2
                and n_axes_largest_eps_wins >= 2):
            verdict = "strong H3"
            summary = (
                "STRONG H3 confirmed: deep hedger less sensitive on the "
                "L2 norm and on at least 2 of 3 axes at the largest "
                "perturbations."
            )
        elif l2_ratio > 1.0 and n_axes_dh_robust >= 1:
            verdict = "moderate H3"
            summary = (
                f"MODERATE H3 confirmed: deep hedger weakly less sensitive "
                f"(L2 ratio {l2_ratio:.2f}) and more robust on at least one axis."
            )
        elif l2_ratio < 1.0 or n_axes_dh_robust == 0:
            verdict = "h3 refuted"
            summary = (
                f"H3 REFUTED: deep hedger is at least as sensitive as BS "
                f"on the L2 norm (ratio {l2_ratio:.2f}). The hedger may "
                f"be overfitting to the calibration point."
            )
        else:
            verdict = "mixed"
            summary = "Mixed evidence — see per-axis details."

        return {
            "l2_ratio": l2_ratio,
            "per_axis_ratio": per_axis_ratio,
            "axis_robustness": axis_robustness,
            "largest_eps_dh_wins": largest_eps_dh_wins,
            "verdict": verdict,
            "summary": summary,
        }

    # ──────────────────────────────────────────────────────────
    # Orchestration
    # ──────────────────────────────────────────────────────────

    def run_all(self) -> dict[str, Any]:
        print("=" * 65, flush=True)
        print("  ADVERSARIAL ROBUSTNESS — H3 TEST", flush=True)
        print("=" * 65, flush=True)
        print(f"  Theta_0: {THETA_0}", flush=True)
        print(f"  n_train={self.n_train}, n_val={self.n_val}, "
              f"n_test/eps={self.n_test}", flush=True)
        print(f"  Gradient: {self.n_grad_paths} paths x "
              f"{self.n_grad_seeds} seeds", flush=True)

        # Setup
        print("\n  --- Baseline hedger setup ---", flush=True)
        self.setup_baseline_hedger()

        # Approach A
        grad_results = self.run_gradient_sensitivity()

        # Approach B
        sweep_results = self.run_perturbation_sweep_all_axes(p0_option="fixed")

        # Verdict
        verdict = self.compute_h3_verdict(grad_results, sweep_results)

        results = {
            "theta_0": THETA_0,
            "calibration_p0": self.p0_calibration,
            "gradient_sensitivity": grad_results,
            "perturbation_sweep": sweep_results,
            "verdict": verdict,
        }

        self.save_results(results)
        self.generate_figures(results)
        self.print_report(results)

        return results

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save_results(self, results: dict) -> None:
        path = self.figures_dir / "adversarial_robustness.json"
        out = _strip_for_json(results)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved {path.name}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Figures
    # ──────────────────────────────────────────────────────────

    def generate_figures(self, results: dict) -> None:
        print("\n  Generating figures ...", flush=True)
        self._fig_gradient_bars(results)
        for axis in ("H", "eta", "rho"):
            self._fig_sweep_axis(axis, results)
        self._fig_degradation_curves(results)
        self._fig_summary(results)

    def _fig_gradient_bars(self, results: dict) -> None:
        bs = results["gradient_sensitivity"]["bs"]
        dh = results["gradient_sensitivity"]["deep"]
        ratios = results["gradient_sensitivity"]["ratio_bs_over_deep"]

        axes = ("H", "eta", "rho")
        bs_vals = [abs(bs[f"mean_grad_{a}"]) for a in axes]
        bs_err = [bs[f"std_grad_{a}"] for a in axes]
        dh_vals = [abs(dh[f"mean_grad_{a}"]) for a in axes]
        dh_err = [dh[f"std_grad_{a}"] for a in axes]

        x = np.arange(len(axes))
        w = 0.36
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.bar(x - w / 2, bs_vals, w, yerr=bs_err, color=C_BS,
               label="BS delta", capsize=4, edgecolor="black", lw=0.5)
        ax.bar(x + w / 2, dh_vals, w, yerr=dh_err, color=C_DEEP,
               label="Deep hedger", capsize=4, edgecolor="black", lw=0.5)

        # Annotate ratios above each axis group
        for i, a in enumerate(axes):
            r = ratios[a]
            top = max(bs_vals[i] + bs_err[i], dh_vals[i] + dh_err[i])
            ax.annotate(
                f"BS/DH = {r:.2f}",
                xy=(x[i], top * 1.06),
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85),
            )

        # Axis labels
        ax.set_xticks(x)
        ax.set_xticklabels([r"$H$", r"$\eta$", r"$\rho$"], fontsize=12)
        ax.set_ylabel(r"$|\partial \mathrm{ES}_{95} / \partial \theta|$", fontsize=11)
        ax.set_title(r"Local Sensitivity of ES$_{95}$ at $\Theta_0$", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.20)
        fig.tight_layout()
        path = self.figures_dir / "fig_h3_gradient_bars.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_sweep_axis(self, axis: str, results: dict) -> None:
        sweep = results["perturbation_sweep"][axis]
        eps_values = sweep["eps_values"]
        # Build perturbed parameter values
        param_values = []
        for eps in eps_values:
            params = dict(THETA_0)
            params[axis] = _clamp_param(axis, params[axis] + eps)
            param_values.append(params[axis])
        es_bs = [sweep["bs"][float(e)]["metrics"]["es_95"] for e in eps_values]
        es_dh = [sweep["deep"][float(e)]["metrics"]["es_95"] for e in eps_values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_values, es_bs, "o-", color=C_BS, lw=2, ms=8,
                label="BS delta", markeredgecolor="black", markeredgewidth=0.5)
        ax.plot(param_values, es_dh, "D-", color=C_DEEP, lw=2, ms=8,
                label="Deep hedger", markeredgecolor="black", markeredgewidth=0.5)

        # Calibration point
        ax.axvline(THETA_0[axis], color="grey", ls="--", lw=1,
                   label=f"Calibration $\\Theta_0[{axis}]={THETA_0[axis]}$")

        ax.set_xlabel({"H": r"$H$", "eta": r"$\eta$", "rho": r"$\rho$"}[axis],
                      fontsize=12)
        ax.set_ylabel("ES$_{95}$", fontsize=11)
        ax.set_title(f"Robustness to {axis} Perturbation", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.figures_dir / f"fig_h3_sweep_{axis}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_degradation_curves(self, results: dict) -> None:
        sweep = results["perturbation_sweep"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=False)
        for ax, axis_name in zip(axes, ("H", "eta", "rho")):
            data = sweep[axis_name]
            eps = data["eps_values"]
            rel_bs = [data["relative_degradation_bs"][float(e)] * 100 for e in eps]
            rel_dh = [data["relative_degradation_deep"][float(e)] * 100 for e in eps]
            ax.axhline(0, color="grey", ls="--", lw=0.7)
            ax.plot(eps, rel_bs, "o-", color=C_BS, lw=2, ms=7,
                    label="BS delta")
            ax.plot(eps, rel_dh, "D-", color=C_DEEP, lw=2, ms=7,
                    label="Deep hedger")
            ax.set_xlabel(f"$\\epsilon$ ({axis_name})", fontsize=11)
            ax.set_ylabel("Relative ES$_{95}$ degradation (%)", fontsize=10)
            ax.set_title(f"{axis_name} axis", fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle("H3 Relative Degradation under Parameter Perturbation",
                     fontsize=12, y=0.998)
        fig.tight_layout()
        path = self.figures_dir / "fig_h3_degradation_curves.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_summary(self, results: dict) -> None:
        sweep = results["perturbation_sweep"]
        bs = results["gradient_sensitivity"]["bs"]
        dh = results["gradient_sensitivity"]["deep"]
        ratios = results["gradient_sensitivity"]["ratio_bs_over_deep"]
        verdict = results["verdict"]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        # (A) Gradient bars
        ax = axes[0, 0]
        axes_names = ("H", "eta", "rho")
        bs_vals = [abs(bs[f"mean_grad_{a}"]) for a in axes_names]
        dh_vals = [abs(dh[f"mean_grad_{a}"]) for a in axes_names]
        x = np.arange(3)
        w = 0.36
        ax.bar(x - w / 2, bs_vals, w, color=C_BS, label="BS delta",
               edgecolor="black", lw=0.5)
        ax.bar(x + w / 2, dh_vals, w, color=C_DEEP, label="Deep hedger",
               edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([r"$H$", r"$\eta$", r"$\rho$"])
        ax.set_ylabel(r"$|\partial \mathrm{ES}/\partial \theta|$")
        ax.set_title(f"(A) Local sensitivity at $\\Theta_0$\n"
                     f"L2 ratio BS/DH = {ratios['l2']:.2f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # (B), (C), (D) sweep panels
        for ax, axis_name, label in [
            (axes[0, 1], "H", "(B) $H$ sweep"),
            (axes[1, 0], "eta", "(C) $\\eta$ sweep"),
            (axes[1, 1], "rho", "(D) $\\rho$ sweep"),
        ]:
            data = sweep[axis_name]
            eps = data["eps_values"]
            param_vals = []
            for e in eps:
                p = dict(THETA_0)
                p[axis_name] = _clamp_param(axis_name, p[axis_name] + e)
                param_vals.append(p[axis_name])
            es_bs = [data["bs"][float(e)]["metrics"]["es_95"] for e in eps]
            es_dh = [data["deep"][float(e)]["metrics"]["es_95"] for e in eps]
            ax.plot(param_vals, es_bs, "o-", color=C_BS, lw=1.7, ms=5, label="BS")
            ax.plot(param_vals, es_dh, "D-", color=C_DEEP, lw=1.7, ms=5, label="DH")
            ax.axvline(THETA_0[axis_name], color="grey", ls="--", lw=0.7)
            ax.set_xlabel({"H": r"$H$", "eta": r"$\eta$", "rho": r"$\rho$"}[axis_name])
            ax.set_ylabel("ES$_{95}$")
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        verdict_str = verdict["verdict"].upper()
        fig.suptitle(f"H3 Adversarial Robustness — {verdict_str}",
                     fontsize=13, y=1.00)
        fig.tight_layout()
        path = self.figures_dir / "fig_h3_summary.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────

    def print_report(self, results: dict) -> None:
        v = results["verdict"]
        gs = results["gradient_sensitivity"]
        sweep = results["perturbation_sweep"]

        print("\n" + "=" * 65, flush=True)
        print("  H3 ADVERSARIAL ROBUSTNESS REPORT", flush=True)
        print("=" * 65, flush=True)

        print("\n1. GRADIENT SENSITIVITY (Approach A)", flush=True)
        print("-" * 65, flush=True)
        print(f"  {'axis':>5s}  {'|grad_BS|':>14s}  {'|grad_DH|':>14s}  "
              f"{'BS/DH ratio':>12s}", flush=True)
        for k in ("H", "eta", "rho"):
            bs_val = abs(gs["bs"][f"mean_grad_{k}"])
            bs_std = gs["bs"][f"std_grad_{k}"]
            dh_val = abs(gs["deep"][f"mean_grad_{k}"])
            dh_std = gs["deep"][f"std_grad_{k}"]
            ratio = gs["ratio_bs_over_deep"][k]
            print(f"  {k:>5s}  {bs_val:>9.4f}±{bs_std:.3f}  "
                  f"{dh_val:>9.4f}±{dh_std:.3f}  {ratio:>11.3f}",
                  flush=True)
        l2_bs = gs["bs"]["mean_grad_l2_norm"]
        l2_dh = gs["deep"]["mean_grad_l2_norm"]
        l2_ratio = gs["ratio_bs_over_deep"]["l2"]
        print(f"\n  L2 norm:  BS = {l2_bs:.4f},  DH = {l2_dh:.4f},  "
              f"ratio = {l2_ratio:.3f}", flush=True)

        print("\n2. PERTURBATION SWEEP (Approach B)", flush=True)
        print("-" * 65, flush=True)
        for axis in ("H", "eta", "rho"):
            data = sweep[axis]
            print(f"\n  Axis {axis}:", flush=True)
            print(f"    {'eps':>8s}  {'ES_BS':>8s}  {'rel_BS':>8s}  "
                  f"{'ES_DH':>8s}  {'rel_DH':>8s}  winner",
                  flush=True)
            for eps in data["eps_values"]:
                ef = float(eps)
                es_bs = data["bs"][ef]["metrics"]["es_95"]
                es_dh = data["deep"][ef]["metrics"]["es_95"]
                rel_bs = data["relative_degradation_bs"][ef] * 100
                rel_dh = data["relative_degradation_deep"][ef] * 100
                winner = "DH" if abs(rel_dh) < abs(rel_bs) else "BS"
                if abs(eps) < 1e-12:
                    winner = "—"
                print(f"    {eps:>+8.4f}  {es_bs:>8.3f}  {rel_bs:>+7.2f}%  "
                      f"{es_dh:>8.3f}  {rel_dh:>+7.2f}%  {winner}",
                      flush=True)

        print("\n3. H3 VERDICT", flush=True)
        print("-" * 65, flush=True)
        print(f"  Verdict: {v['verdict'].upper()}", flush=True)
        print(f"  {v['summary']}", flush=True)
        print(f"\n  L2 ratio BS/DH: {v['l2_ratio']:.3f}", flush=True)
        print("  Per-axis ratios:", flush=True)
        for k, val in v["per_axis_ratio"].items():
            print(f"    {k}: {val:.3f}", flush=True)
        print("\n  Per-axis robustness (DH wins fraction):", flush=True)
        for axis, info in v["axis_robustness"].items():
            print(f"    {axis}: {info['dh_better_count']}/{info['total_perturbations']}"
                  f" = {info['fraction_dh_wins']:.0%}", flush=True)
        print("\n  At largest |eps| on each axis (DH wins by >5% margin):", flush=True)
        for axis, w in v["largest_eps_dh_wins"].items():
            print(f"    {axis}: {w}", flush=True)

        print("\n4. DISSERTATION TEXT DRAFT", flush=True)
        print("-" * 65, flush=True)
        print(self._draft_h3(results), flush=True)

    def _draft_h3(self, results: dict) -> str:
        gs = results["gradient_sensitivity"]
        v = results["verdict"]
        sweep = results["perturbation_sweep"]

        l2_bs = gs["bs"]["mean_grad_l2_norm"]
        l2_dh = gs["deep"]["mean_grad_l2_norm"]
        l2_ratio = v["l2_ratio"]

        # Find the largest |ε| degradations
        deg_text_parts = []
        for axis in ("H", "eta", "rho"):
            data = sweep[axis]
            eps = data["eps_values"]
            eps_max = max(abs(e) for e in eps)
            cands = [e for e in eps if abs(e) >= eps_max - 1e-9]
            for e in cands:
                rel_bs = data["relative_degradation_bs"][float(e)] * 100
                rel_dh = data["relative_degradation_deep"][float(e)] * 100
                deg_text_parts.append(
                    f"At |eps|={abs(e):.3f} on the {axis} axis, BS degrades by "
                    f"{rel_bs:+.1f}% and DH by {rel_dh:+.1f}%."
                )
        deg_text = " ".join(deg_text_parts[:3])  # one per axis

        if v["verdict"] == "strong H3":
            return (
                f"We test Hypothesis H3 via two complementary approaches. "
                f"First, we exploit the differentiability of our rBergomi "
                f"simulator to compute nabla_Theta ES_95 at the baseline "
                f"calibration Theta_0 = ({THETA_0['H']}, {THETA_0['eta']}, "
                f"{THETA_0['rho']}). The L2 norm of the gradient is "
                f"{l2_bs:.3f} for BS delta vs {l2_dh:.3f} for the deep "
                f"hedger — a factor of {l2_ratio:.2f} reduction. "
                f"Second, finite-perturbation sweeps on each axis (using "
                f"the calibration p_0) confirm graceful degradation. "
                f"{deg_text} "
                f"Hypothesis H3 is confirmed in its strong form."
            )
        elif v["verdict"] == "moderate H3":
            return (
                f"The L2 norm of nabla_Theta ES_95 is {l2_bs:.3f} for BS "
                f"delta vs {l2_dh:.3f} for the deep hedger (ratio "
                f"{l2_ratio:.2f}). The deep hedger is weakly more robust, "
                f"but the effect is not uniform across axes. {deg_text} "
                f"Hypothesis H3 is confirmed in moderate form."
            )
        else:
            return (
                f"Contrary to expectation, the deep hedger shows L2 gradient "
                f"norm {l2_dh:.3f} versus {l2_bs:.3f} for BS delta (ratio "
                f"{l2_ratio:.2f}). The deep hedger is at least as sensitive "
                f"as BS along the tested axes. {deg_text} "
                f"Hypothesis H3 is REFUTED at this calibration. This is "
                f"consistent with the decomposition from §6.3.2: a strategy "
                f"whose advantage is objective-driven has no particular "
                f"reason to be robust to dynamic misspecification."
            )


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    exp = AdversarialRobustnessExperiment(
        figures_dir=FIGURE_DIR,
        n_train=60_000,
        n_val=10_000,
        n_test_per_perturbation=30_000,
        n_grad_paths=50_000,
        n_grad_seeds=5,
    )
    exp.run_all()


if __name__ == "__main__":
    main()
