#!/usr/bin/env python
"""
Part B of Prompt 12: transfer learning GBM → rBergomi.

Pretrain a deep hedger on cheap GBM paths, then fine-tune on varying
amounts of rBergomi data. Compare with from-scratch training on the
same rBergomi data. Measures sample efficiency as a function of
n_ft ∈ {0, 500, 2000, 5000, 20000, 80000}.

Run:
    python -u -m deep_hedging.experiments.transfer_learning
"""
from __future__ import annotations

import copy
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

from deep_hedging.core.gbm import GBM
from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN, train_deep_hedger, evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics


# ─── Constants ─────────────────────────────────────────────────
THETA_RBG: dict[str, float] = dict(H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2)
SIGMA_GBM = 0.235
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
MASTER_SEED = 2024

N_FT_VALUES: list[int] = [0, 500, 2000, 5000, 20_000, 80_000]

N_TEST = 30_000
N_VAL_RBG = 10_000  # small fixed validation for fine-tune/scratch

EPOCHS_PRETRAIN = 200
EPOCHS_FINETUNE = 100
EPOCHS_SCRATCH = 200
LR_PRETRAIN = 1e-3
LR_FINETUNE = 3e-4
LR_SCRATCH = 1e-3
PATIENCE_FINETUNE = 15
PATIENCE_SCRATCH = 30
BATCH_SIZE = 2048

HIDDEN_DIM = 128
N_RES_BLOCKS = 2

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
GBM_PATH = FIGURE_DIR / "gbm_pretrained_hedger.pt"
BASELINE_PATH = FIGURE_DIR / "adversarial_baseline_hedger.pt"
RESULTS_PATH = FIGURE_DIR / "transfer_learning.json"

# Colours
C_BS = "#2196F3"
C_DEEP_BASE = "#4CAF50"
C_FT = "#FFA000"
C_SCRATCH = "#9C27B0"


# =======================================================================
# Helpers
# =======================================================================

def _make_fresh_hedger(seed: int = 2024) -> DeepHedgerFNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return DeepHedgerFNN(
        input_dim=4, hidden_dim=HIDDEN_DIM, n_res_blocks=N_RES_BLOCKS,
    )


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


# =======================================================================
# TransferLearningExperiment
# =======================================================================

class TransferLearningExperiment:
    """Pretrain on GBM, fine-tune on limited rBergomi data."""

    def __init__(
        self,
        n_ft_values: list[int] = N_FT_VALUES,
        n_test: int = N_TEST,
        epochs_pretrain: int = EPOCHS_PRETRAIN,
        epochs_finetune: int = EPOCHS_FINETUNE,
        epochs_scratch: int = EPOCHS_SCRATCH,
        figures_dir: str | Path = FIGURE_DIR,
        device: torch.device | None = None,
    ) -> None:
        self.n_ft_values = sorted(set(n_ft_values))
        self.n_test = int(n_test)
        self.epochs_pretrain = int(epochs_pretrain)
        self.epochs_finetune = int(epochs_finetune)
        self.epochs_scratch = int(epochs_scratch)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cpu")

        self.pretrained_state: dict | None = None
        self.gbm_p0: float | None = None
        self.rbg_p0: float | None = None

    # ──────────────────────────────────────────────────────────
    # Step 1: GBM pretraining
    # ──────────────────────────────────────────────────────────

    def pretrain_on_gbm(
        self,
        n_train: int = 80_000,
        n_val: int = 20_000,
        seed: int = MASTER_SEED,
    ) -> dict[str, Any]:
        if GBM_PATH.exists():
            print(f"  Loading cached GBM-pretrained hedger from "
                  f"{GBM_PATH.name}", flush=True)
            state = torch.load(GBM_PATH, map_location=self.device, weights_only=True)
            hedger = _make_fresh_hedger(seed=seed + 1000)
            hedger.load_state_dict(state)
            self.pretrained_state = copy.deepcopy(state)
            return {"hedger": hedger, "history": None, "training_time_s": 0.0,
                    "from_cache": True}

        print(f"  Pretraining hedger on GBM (sigma={SIGMA_GBM}, "
              f"{n_train} paths, {self.epochs_pretrain} epochs) ...", flush=True)
        t0 = time.time()

        gbm = GBM(n_steps=N_STEPS, T=T, sigma=SIGMA_GBM)
        total = n_train + n_val
        S, _, _ = gbm.simulate(n_paths=total, S0=S0, seed=seed)
        S_train = S[:n_train]
        S_val = S[n_train:]
        del S
        gc.collect()

        payoff = compute_payoff(S_train, K, "call")
        self.gbm_p0 = float(payoff.mean())

        hedger = _make_fresh_hedger(seed=seed + 1000)
        torch.manual_seed(seed + 2000)
        np.random.seed(seed + 2000)
        history = train_deep_hedger(
            hedger, S_train, S_val,
            K=K, T=T, S0=S0, p0=self.gbm_p0,
            cost_lambda=0.0, alpha=0.95,
            lr=LR_PRETRAIN,
            batch_size=BATCH_SIZE,
            epochs=self.epochs_pretrain,
            patience=30,
            verbose=False,
        )
        elapsed = time.time() - t0
        print(f"    pretrained in {elapsed/60:.1f} min  "
              f"(best_val={history['best_val_risk']:.3f})", flush=True)

        # Save state
        self.pretrained_state = copy.deepcopy(hedger.state_dict())
        torch.save(hedger.state_dict(), GBM_PATH)
        print(f"    saved {GBM_PATH.name}", flush=True)

        del S_train, S_val
        gc.collect()

        return {"hedger": hedger, "history": history,
                "training_time_s": elapsed, "from_cache": False}

    # ──────────────────────────────────────────────────────────
    # Step 2: rBergomi data preparation
    # ──────────────────────────────────────────────────────────

    def prepare_rbergomi_data(self, seed: int = MASTER_SEED) -> dict[str, Any]:
        max_ft = max(self.n_ft_values)
        total = max_ft + N_VAL_RBG + self.n_test
        print(f"  Generating rBergomi data: {total} paths "
              f"(ft_max={max_ft}, val={N_VAL_RBG}, test={self.n_test}) ...",
              flush=True)
        t0 = time.time()

        sim = DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T,
            H=THETA_RBG["H"], eta=THETA_RBG["eta"], rho=THETA_RBG["rho"],
            xi0=THETA_RBG["xi0"],
        )
        S, _, _ = sim.simulate(n_paths=total, S0=S0, seed=seed)
        n1 = max_ft
        n2 = n1 + N_VAL_RBG
        S_ft_full = S[:n1]
        S_val = S[n1:n2]
        S_test = S[n2:]
        del S
        gc.collect()

        # p0 computed from the FULL fine-tune set (80k)
        payoff_ft = compute_payoff(S_ft_full, K, "call")
        p0 = float(payoff_ft.mean())
        self.rbg_p0 = p0
        print(f"    done in {time.time()-t0:.1f}s, p0={p0:.4f}", flush=True)

        return {"S_ft_full": S_ft_full, "S_val": S_val, "S_test": S_test, "p0": p0}

    # ──────────────────────────────────────────────────────────
    # Step 3: Fine-tune
    # ──────────────────────────────────────────────────────────

    def fine_tune(
        self, pretrained_state: dict, n_ft: int, data: dict,
    ) -> dict[str, Any]:
        hedger = _make_fresh_hedger(seed=MASTER_SEED + 3000 + n_ft)
        hedger.load_state_dict(copy.deepcopy(pretrained_state))

        if n_ft == 0:
            # Pure pretrained evaluation — no training
            return {"hedger": hedger, "history": None, "training_time_s": 0.0,
                    "converged": True, "skipped_training": True}

        S_ft = data["S_ft_full"][:n_ft]
        S_val = data["S_val"]
        p0 = data["p0"]

        t0 = time.time()
        torch.manual_seed(MASTER_SEED + 4000 + n_ft)
        np.random.seed(MASTER_SEED + 4000 + n_ft)

        # For very small n_ft, shrink batch size to allow at least a few steps
        batch_size = min(BATCH_SIZE, max(32, n_ft // 4))

        history = train_deep_hedger(
            hedger, S_ft, S_val,
            K=K, T=T, S0=S0, p0=p0,
            cost_lambda=0.0, alpha=0.95,
            lr=LR_FINETUNE,
            batch_size=batch_size,
            epochs=self.epochs_finetune,
            patience=PATIENCE_FINETUNE,
            verbose=False,
        )
        elapsed = time.time() - t0
        return {
            "hedger": hedger, "history": history,
            "training_time_s": elapsed,
            "converged": True,
            "skipped_training": False,
            "batch_size": batch_size,
        }

    # ──────────────────────────────────────────────────────────
    # Step 3b: From-scratch training
    # ──────────────────────────────────────────────────────────

    def train_from_scratch(self, n_ft: int, data: dict) -> dict[str, Any] | None:
        if n_ft == 0:
            return None

        hedger = _make_fresh_hedger(seed=MASTER_SEED + 5000 + n_ft)

        S_ft = data["S_ft_full"][:n_ft]
        S_val = data["S_val"]
        p0 = data["p0"]

        t0 = time.time()
        torch.manual_seed(MASTER_SEED + 6000 + n_ft)
        np.random.seed(MASTER_SEED + 6000 + n_ft)

        batch_size = min(BATCH_SIZE, max(32, n_ft // 4))

        history = train_deep_hedger(
            hedger, S_ft, S_val,
            K=K, T=T, S0=S0, p0=p0,
            cost_lambda=0.0, alpha=0.95,
            lr=LR_SCRATCH,
            batch_size=batch_size,
            epochs=self.epochs_scratch,
            patience=PATIENCE_SCRATCH,
            verbose=False,
        )
        elapsed = time.time() - t0

        # Mark as not-converged if training risk didn't decrease meaningfully
        tr = history["train_risk"]
        converged = len(tr) >= 2 and tr[-1] < tr[0] - 0.01

        return {
            "hedger": hedger, "history": history,
            "training_time_s": elapsed,
            "converged": converged,
            "skipped_training": False,
            "batch_size": batch_size,
        }

    # ──────────────────────────────────────────────────────────
    # Step 4: Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self, hedger: DeepHedgerFNN, data: dict,
    ) -> dict[str, Any]:
        hedger.eval()
        with torch.no_grad():
            pnl = evaluate_deep_hedger(
                hedger, data["S_test"],
                K=K, T=T, S0=S0, p0=data["p0"], cost_lambda=0.0,
            )
            metrics = compute_all_metrics(pnl)
            deltas = hedger.hedge_paths(data["S_test"], T=T, S0=S0)
            deltas = deltas.to(data["S_test"].dtype)
            batch = deltas.shape[0]
            delta_prev = torch.cat(
                [torch.zeros(batch, 1, dtype=deltas.dtype), deltas[:, :-1]], dim=1,
            )
            turnover = float((deltas - delta_prev).abs().sum(dim=1).mean())
        return {"metrics": metrics, "turnover": turnover}

    def evaluate_bs_reference(self, data: dict) -> dict[str, Any]:
        bs = BlackScholesDelta(sigma=SIGMA_GBM, K=K, T=T)
        with torch.no_grad():
            deltas = bs.hedge_paths(data["S_test"])
            payoff = compute_payoff(data["S_test"], K, "call")
            pnl = compute_hedging_pnl(
                data["S_test"], deltas, payoff, data["p0"], 0.0,
            )
            metrics = compute_all_metrics(pnl)
        return {"metrics": metrics}

    # ──────────────────────────────────────────────────────────
    # Orchestration
    # ──────────────────────────────────────────────────────────

    def run_all(self) -> dict[str, Any]:
        print("=" * 70, flush=True)
        print("  TRANSFER LEARNING — PART B OF PROMPT 12", flush=True)
        print("=" * 70, flush=True)
        print(f"  n_ft values: {self.n_ft_values}", flush=True)
        print(f"  n_test: {self.n_test}", flush=True)
        print(f"  epochs: pretrain={self.epochs_pretrain}, "
              f"finetune={self.epochs_finetune}, scratch={self.epochs_scratch}",
              flush=True)

        # Step 1
        print("\n  --- Step 1: GBM pretraining ---", flush=True)
        pretrain_out = self.pretrain_on_gbm()

        # Step 2
        print("\n  --- Step 2: rBergomi data preparation ---", flush=True)
        data = self.prepare_rbergomi_data()

        # BS reference
        print("\n  --- Step 3: BS delta reference on rBergomi test set ---",
              flush=True)
        bs_ref = self.evaluate_bs_reference(data)
        print(f"    BS ES_95 = {bs_ref['metrics']['es_95']:.4f}", flush=True)

        # Full-budget baseline reference
        full_budget_ref: dict[str, Any] = {"available": False}
        if BASELINE_PATH.exists():
            print("\n  --- Step 4: Full-budget baseline reference ---", flush=True)
            ref_hedger = _make_fresh_hedger()
            ref_hedger.load_state_dict(
                torch.load(BASELINE_PATH, map_location=self.device, weights_only=True),
            )
            ref_eval = self.evaluate(ref_hedger, data)
            full_budget_ref = {"available": True, **ref_eval}
            print(f"    Full-budget ES_95 = "
                  f"{full_budget_ref['metrics']['es_95']:.4f}", flush=True)
            del ref_hedger
            gc.collect()

        # Step 5: transfer curve
        print("\n  --- Step 5: transfer curve (fine-tune + from-scratch) ---",
              flush=True)

        transfer_curve: dict[int, dict[str, Any]] = {}
        results_so_far: dict[str, Any] = {
            "config": {
                "n_ft_values": self.n_ft_values,
                "n_test": self.n_test,
                "epochs_pretrain": self.epochs_pretrain,
                "epochs_finetune": self.epochs_finetune,
                "epochs_scratch": self.epochs_scratch,
                "lr_pretrain": LR_PRETRAIN,
                "lr_finetune": LR_FINETUNE,
                "lr_scratch": LR_SCRATCH,
                "theta_rbg": THETA_RBG,
            },
            "bs_reference": _strip_for_json(bs_ref),
            "full_budget_reference": _strip_for_json(full_budget_ref),
            "transfer_curve": {},
        }

        for i, n_ft in enumerate(self.n_ft_values):
            print(f"\n  [{i+1}/{len(self.n_ft_values)}] n_ft = {n_ft}",
                  flush=True)

            # Fine-tune
            t0 = time.time()
            ft_out = self.fine_tune(self.pretrained_state, n_ft, data)
            ft_eval = self.evaluate(ft_out["hedger"], data)
            ft_time = time.time() - t0
            ft_record = {
                "metrics": ft_eval["metrics"],
                "turnover": ft_eval["turnover"],
                "training_time_s": ft_out["training_time_s"],
                "eval_time_s": ft_time,
                "history": ft_out["history"],
                "skipped_training": ft_out.get("skipped_training", False),
                "converged": ft_out.get("converged", True),
            }
            print(f"    fine-tuned:   ES_95 = {ft_eval['metrics']['es_95']:.4f}  "
                  f"(trained {ft_out['training_time_s']:.0f}s)", flush=True)

            del ft_out
            gc.collect()

            # From-scratch
            scratch_record = None
            if n_ft > 0:
                t0 = time.time()
                scratch_out = self.train_from_scratch(n_ft, data)
                if scratch_out is not None:
                    scratch_eval = self.evaluate(scratch_out["hedger"], data)
                    scratch_record = {
                        "metrics": scratch_eval["metrics"],
                        "turnover": scratch_eval["turnover"],
                        "training_time_s": scratch_out["training_time_s"],
                        "history": scratch_out["history"],
                        "converged": scratch_out["converged"],
                    }
                    print(f"    from-scratch: ES_95 = "
                          f"{scratch_eval['metrics']['es_95']:.4f}  "
                          f"(trained {scratch_out['training_time_s']:.0f}s, "
                          f"converged={scratch_out['converged']})", flush=True)
                    del scratch_out
                    gc.collect()

            transfer_curve[n_ft] = {
                "fine_tuned": ft_record,
                "from_scratch": scratch_record,
            }

            # Incremental save
            results_so_far["transfer_curve"] = _strip_for_json(transfer_curve)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results_so_far, f, indent=2)

        results = {
            "bs_reference": bs_ref,
            "full_budget_reference": full_budget_ref,
            "transfer_curve": transfer_curve,
            "gbm_p0": self.gbm_p0,
            "rbg_p0": self.rbg_p0,
        }

        # Final save with consistent schema
        results_so_far["transfer_curve"] = _strip_for_json(transfer_curve)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results_so_far, f, indent=2)
        print(f"\n  Saved {RESULTS_PATH.name}", flush=True)

        # Figures + report
        self.generate_figures(results)
        self.print_report(results)

        return results

    # ──────────────────────────────────────────────────────────
    # Figures
    # ──────────────────────────────────────────────────────────

    def generate_figures(self, results: dict) -> None:
        print("\n  Generating figures ...", flush=True)
        self._fig_sample_efficiency(results)
        self._fig_training_curves(results)
        self._fig_pnl_distributions(results)

    def _fig_sample_efficiency(self, results: dict) -> None:
        curve = results["transfer_curve"]
        bs_ref = results["bs_reference"]["metrics"]["es_95"]
        full_budget = None
        if results["full_budget_reference"].get("available"):
            full_budget = results["full_budget_reference"]["metrics"]["es_95"]

        n_ft_sorted = sorted(curve.keys())
        ft_es = [curve[n]["fine_tuned"]["metrics"]["es_95"] for n in n_ft_sorted]
        scratch_es = [
            curve[n]["from_scratch"]["metrics"]["es_95"]
            if curve[n]["from_scratch"] is not None else None
            for n in n_ft_sorted
        ]

        # x-axis: n_ft with 0 plotted at a symbolic "leftmost" position
        n_plot = [max(n, 1) for n in n_ft_sorted]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(n_plot, ft_es, "D-", color=C_FT, lw=2, ms=9,
                label="Fine-tuned (GBM → rBergomi)",
                markeredgecolor="black", markeredgewidth=0.5)
        scratch_x = [n for n, v in zip(n_plot, scratch_es) if v is not None]
        scratch_y = [v for v in scratch_es if v is not None]
        ax.plot(scratch_x, scratch_y, "s-", color=C_SCRATCH, lw=2, ms=9,
                label="From scratch",
                markeredgecolor="black", markeredgewidth=0.5)

        ax.axhline(bs_ref, color=C_BS, ls="--", lw=1.5,
                   label=f"BS delta = {bs_ref:.2f}")
        if full_budget is not None:
            ax.axhline(full_budget, color=C_DEEP_BASE, ls="--", lw=1.5,
                       label=f"Full-budget baseline = {full_budget:.2f}")

        ax.set_xscale("log")
        ax.set_xlabel("Number of rBergomi fine-tuning paths $n_{ft}$", fontsize=11)
        ax.set_ylabel("ES$_{95}$ on rBergomi test set", fontsize=11)
        ax.set_title("Sample Efficiency: Fine-tune vs From-Scratch", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")

        # Annotate the n_ft=0 label
        ax.annotate("$n_{ft}=0$", xy=(1, ft_es[0]),
                    xytext=(3, ft_es[0] + 0.3), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="grey"))

        fig.tight_layout()
        path = self.figures_dir / "fig_transfer_sample_efficiency.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_training_curves(self, results: dict) -> None:
        curve = results["transfer_curve"]
        n_ft_sorted = sorted(curve.keys())
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, n_ft in zip(axes.flat, n_ft_sorted):
            ft = curve[n_ft]["fine_tuned"]
            sc = curve[n_ft]["from_scratch"]

            if ft.get("history") is not None:
                hist = ft["history"]
                ax.plot(hist["val_risk"], "-", color=C_FT, lw=1.5,
                        label="Fine-tuned val")
                ax.plot(hist["train_risk"], "--", color=C_FT, lw=1, alpha=0.6)
            if sc is not None and sc.get("history") is not None:
                hist = sc["history"]
                ax.plot(hist["val_risk"], "-", color=C_SCRATCH, lw=1.5,
                        label="From-scratch val")
                ax.plot(hist["train_risk"], "--", color=C_SCRATCH, lw=1, alpha=0.6)
            ax.set_title(f"$n_{{ft}}={n_ft}$", fontsize=10)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Risk", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide any unused axes
        for k in range(len(n_ft_sorted), len(axes.flat)):
            axes.flat[k].axis("off")

        fig.suptitle("Training Curves: Fine-tune vs From-Scratch",
                     fontsize=12, y=1.00)
        fig.tight_layout()
        path = self.figures_dir / "fig_transfer_training_curves.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_pnl_distributions(self, results: dict) -> None:
        """Schematic distributions based on per-cell ES/std summaries."""
        curve = results["transfer_curve"]
        picks = [0, 2000, 20000]
        picks = [p for p in picks if p in curve]
        if not picks:
            return

        fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4))
        if len(picks) == 1:
            axes = [axes]

        labels = ["Fine-tuned", "From-scratch", "BS ref", "Full ref"]
        colors = [C_FT, C_SCRATCH, C_BS, C_DEEP_BASE]

        for ax, n_ft in zip(axes, picks):
            ft_metric = curve[n_ft]["fine_tuned"]["metrics"]
            sc = curve[n_ft]["from_scratch"]
            sc_metric = sc["metrics"] if sc is not None else None

            # Build a bar chart of ES_95 / std_pnl
            names = ["Fine-tuned"]
            es_vals = [ft_metric["es_95"]]
            std_vals = [ft_metric["std_pnl"]]
            color_list = [C_FT]
            if sc_metric:
                names.append("From-scratch")
                es_vals.append(sc_metric["es_95"])
                std_vals.append(sc_metric["std_pnl"])
                color_list.append(C_SCRATCH)

            bs_es = results["bs_reference"]["metrics"]["es_95"]
            names.append("BS")
            es_vals.append(bs_es)
            std_vals.append(results["bs_reference"]["metrics"]["std_pnl"])
            color_list.append(C_BS)

            if results["full_budget_reference"].get("available"):
                names.append("Baseline")
                es_vals.append(
                    results["full_budget_reference"]["metrics"]["es_95"]
                )
                std_vals.append(
                    results["full_budget_reference"]["metrics"]["std_pnl"]
                )
                color_list.append(C_DEEP_BASE)

            x = np.arange(len(names))
            ax.bar(x, es_vals, color=color_list, edgecolor="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, fontsize=8)
            ax.set_ylabel("ES$_{95}$")
            ax.set_title(f"$n_{{ft}}={n_ft}$", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

            for xi, v in zip(x, es_vals):
                ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=7)

        fig.suptitle("ES$_{95}$ Comparison at Representative $n_{ft}$ Values",
                     fontsize=11, y=1.03)
        fig.tight_layout()
        path = self.figures_dir / "fig_transfer_pnl_distributions.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────

    def print_report(self, results: dict) -> None:
        curve = results["transfer_curve"]
        bs_ref_es = results["bs_reference"]["metrics"]["es_95"]
        full_budget = None
        if results["full_budget_reference"].get("available"):
            full_budget = results["full_budget_reference"]["metrics"]["es_95"]

        print("\n" + "=" * 70, flush=True)
        print("  TRANSFER LEARNING REPORT", flush=True)
        print("=" * 70, flush=True)
        print(f"\n  References on rBergomi test set:", flush=True)
        print(f"    BS delta:              {bs_ref_es:.4f}", flush=True)
        if full_budget is not None:
            print(f"    Full-budget baseline:  {full_budget:.4f}", flush=True)

        print(f"\n  Transfer curve:", flush=True)
        print(f"    {'n_ft':>8s}  {'Fine-tuned':>12s}  {'From-scratch':>14s}  "
              f"{'Δ(FT vs SC)':>12s}", flush=True)
        print("    " + "-" * 55, flush=True)

        for n_ft in sorted(curve.keys()):
            ft_es = curve[n_ft]["fine_tuned"]["metrics"]["es_95"]
            sc = curve[n_ft]["from_scratch"]
            if sc is None:
                sc_str = "—"
                diff_str = "—"
            else:
                sc_es = sc["metrics"]["es_95"]
                sc_str = f"{sc_es:.4f}"
                diff = sc_es - ft_es
                diff_str = f"{diff:+.4f}"
            print(f"    {n_ft:>8d}  {ft_es:>12.4f}  {sc_str:>14s}  {diff_str:>12s}",
                  flush=True)

        # Verdict
        print(f"\n  Verdict:", flush=True)
        practical_threshold = None
        if full_budget is not None:
            for n_ft in sorted(curve.keys()):
                ft_es = curve[n_ft]["fine_tuned"]["metrics"]["es_95"]
                if ft_es <= full_budget + 0.5:
                    practical_threshold = n_ft
                    break
        if practical_threshold is not None:
            print(f"    Practical threshold (fine-tuned within 0.5 of baseline): "
                  f"n_ft = {practical_threshold}", flush=True)
        else:
            print(f"    Fine-tuned did not reach within 0.5 of baseline at "
                  f"any tested n_ft.", flush=True)

        max_transfer_benefit = None
        max_benefit_n_ft = None
        for n_ft in sorted(curve.keys()):
            ft_es = curve[n_ft]["fine_tuned"]["metrics"]["es_95"]
            sc = curve[n_ft]["from_scratch"]
            if sc is None:
                continue
            benefit = sc["metrics"]["es_95"] - ft_es
            if max_transfer_benefit is None or benefit > max_transfer_benefit:
                max_transfer_benefit = benefit
                max_benefit_n_ft = n_ft
        if max_transfer_benefit is not None:
            print(f"    Max transfer benefit (from-scratch − fine-tuned): "
                  f"{max_transfer_benefit:+.4f} at n_ft={max_benefit_n_ft}",
                  flush=True)

        # BS comparison
        beats_bs = [n_ft for n_ft in sorted(curve.keys())
                    if curve[n_ft]["fine_tuned"]["metrics"]["es_95"] < bs_ref_es]
        if beats_bs:
            print(f"    Fine-tuned beats BS delta at n_ft >= {min(beats_bs)}",
                  flush=True)

        print("\n  Dissertation text draft:", flush=True)
        print("-" * 70, flush=True)
        print(self._draft_text(results, practical_threshold), flush=True)

    def _draft_text(self, results: dict, practical_threshold: int | None) -> str:
        curve = results["transfer_curve"]
        bs_es = results["bs_reference"]["metrics"]["es_95"]
        full_budget = None
        if results["full_budget_reference"].get("available"):
            full_budget = results["full_budget_reference"]["metrics"]["es_95"]

        ft0_es = curve[0]["fine_tuned"]["metrics"]["es_95"]
        n_values = sorted(curve.keys())
        ft_series_str = ", ".join(
            f"({n}, {curve[n]['fine_tuned']['metrics']['es_95']:.2f})"
            for n in n_values
        )

        full_budget_str = f"{full_budget:.2f}" if full_budget is not None else "N/A"

        if practical_threshold is not None and practical_threshold <= 5000:
            narrative = (
                f"a practical threshold of n_ft={practical_threshold} "
                f"rBergomi paths is enough to bring the fine-tuned hedger "
                f"within 0.5 of the full-budget baseline (ES_95 "
                f"{full_budget_str}). In production terms, this means a "
                f"quant can substitute an inexpensive GBM pretraining stage "
                f"for a large rBergomi sample, then fine-tune on a few "
                f"thousand rBergomi paths and recover near-baseline "
                f"performance. Transfer learning is effective."
            )
        elif practical_threshold is not None:
            narrative = (
                f"the fine-tuned hedger reaches within 0.5 of the full-budget "
                f"baseline only at n_ft={practical_threshold} — a regime in "
                f"which a from-scratch hedger trained on the same paths also "
                f"converges. Transfer learning provides a mild head-start "
                f"but does not dramatically reduce the required rBergomi "
                f"sample size."
            )
        else:
            narrative = (
                f"across the tested range, the fine-tuned hedger does not "
                f"reach within 0.5 of the full-budget baseline at any tested "
                f"n_ft. The GBM → rBergomi transfer is weak: the hedger must "
                f"see enough rBergomi data to re-learn the stochastic-vol "
                f"structure, and the GBM pretraining provides little head start."
            )

        return (
            f"We test whether rBergomi training can be replaced by GBM "
            f"pretraining followed by a small rBergomi fine-tune. A flat "
            f"deep hedger is pretrained on 80 000 GBM(sigma={SIGMA_GBM}) paths "
            f"and fine-tuned on n_ft rBergomi(H=0.07) paths for "
            f"n_ft in {{0, 500, 2000, 5000, 20 000, 80 000}}. Without "
            f"fine-tuning (n_ft=0), the pretrained hedger achieves ES_95 = "
            f"{ft0_es:.2f} on the rBergomi test set, compared to {bs_es:.2f} "
            f"for BS delta and {full_budget_str} for the full-budget baseline. "
            f"The fine-tuning curve is {ft_series_str}. "
            f"We observe {narrative}"
        )


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    exp = TransferLearningExperiment()
    exp.run_all()


if __name__ == "__main__":
    main()
