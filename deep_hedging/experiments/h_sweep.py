#!/usr/bin/env python
"""
H-sweep experiment: continuous deformation from GBM (H=0.5) to ultra-rough (H=0.01).

Quantifies how the deep hedging advantage gap
    Gamma(H) := ES_95^{BS}(H) - ES_95^{DH}(H)
scales with the Hurst parameter.  Tests Proposition 6.9.

Run:
    python -u -m deep_hedging.experiments.h_sweep
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    hedge_paths_deep,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics
from deep_hedging.utils.config import H_SWEEP_VALUES

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"


class HurstSweepExperiment:
    """Systematic H-deformation experiment (Definition 6.8).

    For each H in a grid, generates rBergomi paths, trains a deep hedger,
    evaluates BS delta and the deep hedger, and records all metrics.
    """

    def __init__(
        self,
        H_values: list[float] | None = None,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.235 ** 2,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        n_steps: int = 100,
        n_train: int = 80_000,
        n_val: int = 20_000,
        n_test: int = 50_000,
        cost_lambda: float = 0.0,
        epochs: int = 150,
        patience: int = 25,
        device: torch.device | None = None,
    ) -> None:
        self.H_values = H_values or list(H_SWEEP_VALUES)
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.S0 = S0
        self.K = K
        self.T = T
        self.n_steps = n_steps
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.cost_lambda = cost_lambda
        self.epochs = epochs
        self.patience = patience
        self.device = device or torch.device("cpu")
        self.sigma = math.sqrt(xi0)

    # ------------------------------------------------------------------

    def run_single_H(self, H: float, seed: int = 2024) -> dict[str, Any]:
        """Full pipeline for one value of H."""
        total = self.n_train + self.n_val + self.n_test

        # 1. Generate paths
        sim = DifferentiableRoughBergomi(
            n_steps=self.n_steps, T=self.T, H=H,
            eta=self.eta, rho=self.rho, xi0=self.xi0,
        )
        S_all, V_all, _ = sim.simulate(n_paths=total, S0=self.S0, seed=seed)
        n1 = self.n_train
        n2 = n1 + self.n_val
        S_tr, S_va, S_te = S_all[:n1], S_all[n1:n2], S_all[n2:]
        del S_all, V_all
        gc.collect()

        # 2. MC option price
        payoff_tr = compute_payoff(S_tr, self.K, "call")
        p0 = float(payoff_tr.mean())

        # 3. BS delta on test
        bs = BlackScholesDelta(sigma=self.sigma, K=self.K, T=self.T)
        deltas_bs = bs.hedge_paths(S_te)
        payoff_te = compute_payoff(S_te, self.K, "call")
        pnl_bs = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, self.cost_lambda)
        bs_metrics = compute_all_metrics(pnl_bs)

        # 4. Train deep hedger
        t0 = time.perf_counter()
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        history = train_deep_hedger(
            model, S_tr, S_va,
            K=self.K, T=self.T, S0=self.S0, p0=p0,
            cost_lambda=self.cost_lambda, alpha=0.95,
            lr=1e-3, batch_size=2048,
            epochs=self.epochs, patience=self.patience,
            device=self.device, verbose=False,
        )
        train_time = time.perf_counter() - t0

        # 5. Evaluate deep hedger
        pnl_dh = evaluate_deep_hedger(
            model, S_te, K=self.K, T=self.T,
            S0=self.S0, p0=p0, cost_lambda=self.cost_lambda,
        )
        dh_metrics = compute_all_metrics(pnl_dh)

        gamma = bs_metrics["es_95"] - dh_metrics["es_95"]

        # Cleanup training data
        del S_tr, S_va, model
        gc.collect()

        return {
            "H": H,
            "p0": p0,
            "bs_metrics": bs_metrics,
            "bs_pnl": pnl_bs,
            "dh_metrics": dh_metrics,
            "dh_pnl": pnl_dh,
            "gamma": gamma,
            "train_history": history,
            "training_time_s": train_time,
        }

    # ------------------------------------------------------------------

    def run_full_sweep(self) -> list[dict[str, Any]]:
        """Run run_single_H for every H value."""
        results: list[dict[str, Any]] = []
        n_total = len(self.H_values)
        elapsed_total = 0.0

        for i, H in enumerate(self.H_values):
            seed = 2024 + i
            t0 = time.perf_counter()
            print(f"\n  H = {H:.2f}  ({i+1}/{n_total})  training ...", end="", flush=True)

            r = self.run_single_H(H, seed=seed)
            dt = time.perf_counter() - t0
            elapsed_total += dt
            avg_per_H = elapsed_total / (i + 1)
            remaining = avg_per_H * (n_total - i - 1)

            print(
                f"  done in {dt:.0f}s  |  "
                f"Gamma={r['gamma']:+.3f}  "
                f"ES95_BS={r['bs_metrics']['es_95']:.2f}  "
                f"ES95_DH={r['dh_metrics']['es_95']:.2f}  "
                f"|  ETA {remaining/60:.0f}min",
                flush=True,
            )
            results.append(r)

        return results

    # ------------------------------------------------------------------

    def save_results(
        self, results: list[dict], path: str | Path = "figures/h_sweep_results.json",
    ) -> None:
        """Serialize scalar metrics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for r in results:
            out.append({
                "H": r["H"],
                "p0": r["p0"],
                "gamma": r["gamma"],
                "training_time_s": r["training_time_s"],
                "bs_metrics": r["bs_metrics"],
                "dh_metrics": r["dh_metrics"],
                "best_epoch": r["train_history"]["best_epoch"],
                "best_val_risk": r["train_history"]["best_val_risk"],
            })
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {path}", flush=True)

    # ------------------------------------------------------------------

    def print_summary_table(self, results: list[dict]) -> None:
        hdr = (
            f"{'H':>5s} | {'p0':>7s} | {'ES95_BS':>8s} | {'ES95_DH':>8s} | "
            f"{'Gamma':>7s} | {'std_BS':>7s} | {'std_DH':>7s} | {'time':>5s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            bm, dm = r["bs_metrics"], r["dh_metrics"]
            print(
                f"{r['H']:5.2f} | {r['p0']:7.3f} | {bm['es_95']:8.3f} | "
                f"{dm['es_95']:8.3f} | {r['gamma']:+7.3f} | "
                f"{bm['std_pnl']:7.3f} | {dm['std_pnl']:7.3f} | "
                f"{r['training_time_s']:5.0f}s"
            )

    # ------------------------------------------------------------------

    def generate_figures(
        self, results: list[dict], save_dir: str | Path = "figures",
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        Hs = [r["H"] for r in results]
        es_bs = [r["bs_metrics"]["es_95"] for r in results]
        es_dh = [r["dh_metrics"]["es_95"] for r in results]
        gammas = [r["gamma"] for r in results]

        # --- Figure 1: ES_95 vs H ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Hs, es_bs, "o-", color="#2196F3", label="BS Delta", lw=2, ms=6)
        ax.plot(Hs, es_dh, "s-", color="#4CAF50", label="Deep Hedger", lw=2, ms=6)
        ax.set_xlabel("Hurst parameter $H$", fontsize=12)
        ax.set_ylabel("ES$_{0.95}$", fontsize=12)
        ax.set_title("Expected Shortfall vs Roughness", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_h_sweep_es95.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_h_sweep_es95.png'}", flush=True)

        # --- Figure 2: Gamma vs H ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Hs, gammas, "D-", color="#9C27B0", lw=2, ms=7)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Hurst parameter $H$", fontsize=12)
        ax.set_ylabel("$\\Gamma(H) = $ ES$_{95}^{BS}$ $-$ ES$_{95}^{DH}$", fontsize=12)
        ax.set_title("Deep Hedging Advantage Gap vs Roughness", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_h_sweep_gamma.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_h_sweep_gamma.png'}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65, flush=True)
    print("  H-Sweep: Deep Hedging Advantage vs Roughness", flush=True)
    print("=" * 65, flush=True)

    exp = HurstSweepExperiment()
    results = exp.run_full_sweep()

    print("\n" + "=" * 65, flush=True)
    print("  SUMMARY TABLE", flush=True)
    print("=" * 65, flush=True)
    exp.print_summary_table(results)

    exp.save_results(results, FIGURE_DIR / "h_sweep_results.json")

    print("\n  Generating figures ...", flush=True)
    exp.generate_figures(results, FIGURE_DIR)

    print("\n" + "=" * 65, flush=True)
    print("  DONE", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
