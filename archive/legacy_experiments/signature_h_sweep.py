#!/usr/bin/env python
"""
Stage 2: H-sweep with three feature sets (flat, sig-3, sig-full).

Produces the headline H4 result: how the deep hedging advantage
depends on roughness for each feature set.

Run:
    python -u -m deep_hedging.experiments.signature_h_sweep
    python -u -m deep_hedging.experiments.signature_h_sweep --H-values 0.05 0.2 0.5
    python -u -m deep_hedging.experiments.signature_h_sweep --n-train 40000 --epochs 100
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import train_deep_hedger, evaluate_deep_hedger
from deep_hedging.hedging.signature_hedger import SignatureDeepHedger
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics
from deep_hedging.utils.config import H_SWEEP_VALUES

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

# Constants
SIGMA = 0.235
XI0 = SIGMA ** 2
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100

TRAIN_CFG = dict(
    lr=1e-3,
    batch_size=2048,
    patience=30,
    alpha=0.95,
    verbose=False,
)


class SignatureHSweepExperiment:
    """H-sweep with three feature sets."""

    def __init__(
        self,
        H_values: Optional[list[float]] = None,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = XI0,
        S0_: float = S0,
        K_: float = K,
        T_: float = T,
        n_steps: int = N_STEPS,
        n_train: int = 80_000,
        n_val: int = 20_000,
        n_test: int = 50_000,
        epochs: int = 200,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "figures",
    ) -> None:
        self.H_values = list(H_values) if H_values is not None else list(H_SWEEP_VALUES)
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.S0 = S0_
        self.K = K_
        self.T = T_
        self.n_steps = n_steps
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.epochs = epochs
        self.device = device or torch.device("cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sigma = math.sqrt(xi0)

    # ──────────────────────────────────────────────────────────

    def run_single_H(self, H: float, seed: int = 2024) -> dict[str, Any]:
        """Train and evaluate flat / sig-3 / sig-full at one H value."""
        total = self.n_train + self.n_val + self.n_test

        # 1. Generate paths (shared across feature sets)
        sim = DifferentiableRoughBergomi(
            n_steps=self.n_steps, T=self.T, H=H,
            eta=self.eta, rho=self.rho, xi0=self.xi0,
        )
        S_all, _, _ = sim.simulate(n_paths=total, S0=self.S0, seed=seed)
        n1 = self.n_train
        n2 = n1 + self.n_val
        S_tr, S_va, S_te = S_all[:n1], S_all[n1:n2], S_all[n2:]
        del S_all
        gc.collect()

        payoff_tr = compute_payoff(S_tr, self.K, "call")
        p0 = float(payoff_tr.mean())

        # 2. BS delta
        bs = BlackScholesDelta(sigma=self.sigma, K=self.K, T=self.T)
        deltas_bs = bs.hedge_paths(S_te)
        payoff_te = compute_payoff(S_te, self.K, "call")
        bs_pnl = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, 0.0)
        bs_metrics = compute_all_metrics(bs_pnl)

        out: dict[str, Any] = {
            "H": H, "p0": p0,
            "bs_metrics": bs_metrics,
            "bs_pnl": bs_pnl,
        }

        # 3. Three deep hedgers (same NN init seed for fair comparison)
        init_seed = seed + 1000
        training_times: dict[str, float] = {}
        for fs in ["flat", "sig-3", "sig-full"]:
            label = fs.replace("-", "")
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            hedger = SignatureDeepHedger(
                feature_set=fs, hidden_dim=128, n_res_blocks=2,
                xi0=self.xi0, eta_ref=self.eta, T=self.T,
            )
            t0 = time.perf_counter()
            history = train_deep_hedger(
                hedger, S_tr, S_va,
                K=self.K, T=self.T, S0=self.S0, p0=p0, cost_lambda=0.0,
                epochs=self.epochs, **TRAIN_CFG,
            )
            dt = time.perf_counter() - t0
            training_times[label] = dt

            pnl = evaluate_deep_hedger(
                hedger, S_te, K=self.K, T=self.T, S0=self.S0, p0=p0,
            )
            metrics = compute_all_metrics(pnl)
            out[f"{label}_metrics"] = metrics
            out[f"{label}_pnl"] = pnl
            out[f"{label}_history"] = history
            del hedger
            gc.collect()

        # 4. Gamma values
        out["gamma_flat"] = bs_metrics["es_95"] - out["flat_metrics"]["es_95"]
        out["gamma_sig3"] = bs_metrics["es_95"] - out["sig3_metrics"]["es_95"]
        out["gamma_sigfull"] = bs_metrics["es_95"] - out["sigfull_metrics"]["es_95"]
        out["roughness_advantage"] = out["gamma_sigfull"] - out["gamma_flat"]
        out["training_times"] = training_times

        del S_tr, S_va, S_te
        gc.collect()
        return out

    # ──────────────────────────────────────────────────────────

    def run_full_sweep(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        n_total = len(self.H_values)
        elapsed_total = 0.0

        for i, H in enumerate(self.H_values):
            seed = 2024 + i
            print(f"\n  H = {H:.2f}  ({i+1}/{n_total}) ...", flush=True)
            t0 = time.perf_counter()
            try:
                r = self.run_single_H(H, seed=seed)
                dt = time.perf_counter() - t0
                elapsed_total += dt
                eta_min = (elapsed_total / (i + 1)) * (n_total - i - 1) / 60.0
                print(
                    f"    done in {dt:.0f}s  |  "
                    f"Gamma_flat={r['gamma_flat']:+.3f}  "
                    f"Gamma_sig3={r['gamma_sig3']:+.3f}  "
                    f"Gamma_sigfull={r['gamma_sigfull']:+.3f}  "
                    f"roughness_adv={r['roughness_advantage']:+.3f}  |  "
                    f"ETA {eta_min:.0f}min",
                    flush=True,
                )
                results.append(r)

                # Save incrementally
                self.save_results(results, self.save_dir / "signature_h_sweep.json")
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                continue

        return results

    # ──────────────────────────────────────────────────────────

    def save_results(
        self, results: list[dict],
        path: str | Path = "figures/signature_h_sweep.json",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for r in results:
            out.append({
                "H": r["H"], "p0": r["p0"],
                "bs_metrics": r["bs_metrics"],
                "flat_metrics": r["flat_metrics"],
                "sig3_metrics": r["sig3_metrics"],
                "sigfull_metrics": r["sigfull_metrics"],
                "gamma_flat": r["gamma_flat"],
                "gamma_sig3": r["gamma_sig3"],
                "gamma_sigfull": r["gamma_sigfull"],
                "roughness_advantage": r["roughness_advantage"],
                "training_times": r["training_times"],
            })
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def print_summary_table(self, results: list[dict]) -> None:
        print(f"\n{'H':>5s} | {'Γ_flat':>8s} | {'Γ_sig3':>8s} | {'Γ_sigf':>8s} | "
              f"{'rough_adv':>9s} | best", flush=True)
        print("-" * 60, flush=True)
        for r in results:
            best_label = "sig-full" if r["gamma_sigfull"] >= max(r["gamma_flat"], r["gamma_sig3"]) \
                         else ("sig-3" if r["gamma_sig3"] >= r["gamma_flat"] else "flat")
            print(
                f"{r['H']:5.2f} | {r['gamma_flat']:+8.3f} | {r['gamma_sig3']:+8.3f} | "
                f"{r['gamma_sigfull']:+8.3f} | {r['roughness_advantage']:+9.3f} | {best_label}",
                flush=True,
            )


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H-values", nargs="+", type=float, default=None,
                        help="H values to sweep (default: H_SWEEP_VALUES)")
    parser.add_argument("--n-train", type=int, default=80_000)
    parser.add_argument("--n-val", type=int, default=20_000)
    parser.add_argument("--n-test", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    print("=" * 65, flush=True)
    print("  STAGE 2: H-sweep with three feature sets", flush=True)
    print("=" * 65, flush=True)

    exp = SignatureHSweepExperiment(
        H_values=args.H_values,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        epochs=args.epochs, save_dir=FIGURE_DIR,
    )

    print(f"  H values: {exp.H_values}", flush=True)
    print(f"  n_train={exp.n_train}, epochs={exp.epochs}", flush=True)

    results = exp.run_full_sweep()

    print("\n" + "=" * 65, flush=True)
    print("  STAGE 2 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    exp.print_summary_table(results)

    print("\n  Done.", flush=True)


if __name__ == "__main__":
    main()
