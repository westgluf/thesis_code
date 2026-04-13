#!/usr/bin/env python
"""
Pareto front and H2 factorial experiment (Prompt 9).

Part A: frequency x cost factorial (tests H2).
Part B: Pareto front over risk objectives at fixed (n_steps, cost).

This script ONLY produces raw data (JSON + PnL tensors). Figures and
analysis are Prompt 10's job.

Run:
    python -u -m deep_hedging.experiments.pareto_front --part all
    python -u -m deep_hedging.experiments.pareto_front --part A --full
    python -u -m deep_hedging.experiments.pareto_front --part B
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import evaluate_deep_hedger
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics
from deep_hedging.experiments._training_helpers import (
    train_deep_hedger_with_objective,
    make_objective_tag,
)

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"


# =======================================================================
# Helpers
# =======================================================================

def _mean_turnover(deltas: torch.Tensor) -> float:
    """Mean over paths of sum_k |delta_k - delta_{k-1}|, with delta_{-1} = 0."""
    batch = deltas.shape[0]
    delta_prev = torch.cat([
        torch.zeros(batch, 1, dtype=deltas.dtype, device=deltas.device),
        deltas[:, :-1],
    ], dim=1)
    per_path = (deltas - delta_prev).abs().sum(dim=1)
    return float(per_path.mean())


def _strip_for_json(obj: Any) -> Any:
    """Recursively strip torch tensors, nn.Modules, and non-JSON types."""
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, nn.Module):
        return None
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items()
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


def _normalise_keys(obj: Any) -> Any:
    """Stringify all dict keys (JSON requires string keys)."""
    if isinstance(obj, dict):
        return {str(k): _normalise_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_keys(v) for v in obj]
    return obj


# =======================================================================
# ParetoExperiment
# =======================================================================

class ParetoExperiment:
    """H2 factorial + Pareto front over risk objectives."""

    def __init__(
        self,
        H: float = 0.07,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.235 ** 2,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        n_train: int = 60_000,
        n_val: int = 10_000,
        n_test: int = 30_000,
        save_dir: str | Path = "figures",
        device: Optional[torch.device] = None,
    ) -> None:
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.sigma = float(np.sqrt(xi0))
        self.S0 = S0
        self.K = K
        self.T = T
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Part A: frequency x cost factorial
    # ------------------------------------------------------------------

    def generate_paths(self, n_steps: int, seed: int) -> dict:
        """Simulate rBergomi at (n_steps, H) and split into train/val/test."""
        total = self.n_train + self.n_val + self.n_test
        sim = DifferentiableRoughBergomi(
            n_steps=n_steps, T=self.T, H=self.H,
            eta=self.eta, rho=self.rho, xi0=self.xi0,
        )
        S_all, _, _ = sim.simulate(n_paths=total, S0=self.S0, seed=seed)

        n1 = self.n_train
        n2 = n1 + self.n_val
        S_train = S_all[:n1]
        S_val = S_all[n1:n2]
        S_test = S_all[n2:]
        del S_all
        gc.collect()

        payoff_tr = compute_payoff(S_train, self.K, "call")
        p0 = float(payoff_tr.mean())
        return {"S_train": S_train, "S_val": S_val, "S_test": S_test, "p0": p0}

    def run_bs_delta(
        self, data: dict, n_steps: int, cost_lambda: float,
    ) -> dict:
        """Evaluate BS delta on one (n_steps, cost) cell."""
        bs = BlackScholesDelta(sigma=self.sigma, K=self.K, T=self.T)
        deltas = bs.hedge_paths(data["S_test"])
        payoff = compute_payoff(data["S_test"], self.K, "call")
        pnl = compute_hedging_pnl(
            data["S_test"], deltas, payoff, data["p0"], cost_lambda,
        )
        return {
            "metrics": compute_all_metrics(pnl),
            "pnl": pnl,
            "mean_turnover": _mean_turnover(deltas),
        }

    def run_deep_hedger_cell(
        self,
        data: dict,
        n_steps: int,
        cost_lambda: float,
        seed: int,
        epochs: int = 150,
    ) -> dict:
        """Train and evaluate a flat deep hedger on one (n_steps, cost) cell.

        The hedger is trained with ``cost_lambda`` inside the objective.
        Default risk: ES at alpha=0.95.
        """
        out = train_deep_hedger_with_objective(
            data["S_train"], data["S_val"],
            objective_name="es",
            objective_kwargs={"alpha": 0.95},
            cost_lambda=cost_lambda,
            p0=data["p0"],
            K=self.K, T=self.T, S0=self.S0,
            epochs=epochs, seed=seed, device=self.device,
        )
        model = out["model"]
        pnl = evaluate_deep_hedger(
            model, data["S_test"],
            K=self.K, T=self.T, S0=self.S0, p0=data["p0"],
            cost_lambda=cost_lambda,
        )
        model.eval()
        with torch.no_grad():
            deltas = model.hedge_paths(data["S_test"], self.T, self.S0)
        deltas = deltas.to(data["S_test"].dtype)
        turnover = _mean_turnover(deltas)
        metrics = compute_all_metrics(pnl)
        del model
        gc.collect()
        return {
            "metrics": metrics,
            "pnl": pnl,
            "mean_turnover": turnover,
            "history": out["history"],
            "train_time_s": out["train_time_s"],
        }

    def run_part_A(
        self,
        freq_values: list[int] = (25, 50, 100, 200),
        cost_values: list[float] = (0.0, 0.0005, 0.001, 0.002),
        epochs: int = 150,
    ) -> dict:
        """Run the full frequency x cost factorial with incremental saving."""
        results: dict[str, Any] = {
            "bs": {},
            "deep": {},
            "config": {
                "freq_values": list(freq_values),
                "cost_values": list(cost_values),
                "epochs": epochs,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "H": self.H, "eta": self.eta, "rho": self.rho, "xi0": self.xi0,
            },
        }

        n_cells = len(freq_values) * len(cost_values)
        cell_count = 0
        t_start = time.time()

        for i, n_steps in enumerate(freq_values):
            print(f"\n--- Frequency n_steps={n_steps} "
                  f"({i+1}/{len(freq_values)}) ---", flush=True)
            data_seed = 2024 + i * 100
            t_gen = time.time()
            data = self.generate_paths(n_steps=n_steps, seed=data_seed)
            print(f"  Paths generated in {time.time()-t_gen:.1f}s, "
                  f"p0={data['p0']:.4f}", flush=True)

            results["bs"][n_steps] = {}
            results["deep"][n_steps] = {}

            for j, cost_lambda in enumerate(cost_values):
                cell_count += 1
                print(f"\n  Cell ({n_steps}, {cost_lambda})  "
                      f"[{cell_count}/{n_cells}]", flush=True)

                # --- BS delta (cheap) ---
                bs_cell = self.run_bs_delta(data, n_steps, cost_lambda)
                results["bs"][n_steps][cost_lambda] = {
                    "metrics": bs_cell["metrics"],
                    "mean_turnover": bs_cell["mean_turnover"],
                }
                print(f"    BS:   ES_95={bs_cell['metrics']['es_95']:.3f}  "
                      f"turnover={bs_cell['mean_turnover']:.3f}", flush=True)
                torch.save(
                    bs_cell["pnl"].detach().float().cpu(),
                    self.save_dir / f"pareto_A_n{n_steps}_cost{cost_lambda}_bs_pnl.pt",
                )

                # --- Deep hedger (expensive) ---
                cell_seed = 2024 + i * 100 + j * 10 + 1
                t_deep = time.time()
                deep_cell = self.run_deep_hedger_cell(
                    data, n_steps, cost_lambda,
                    seed=cell_seed, epochs=epochs,
                )
                dt_deep = time.time() - t_deep
                results["deep"][n_steps][cost_lambda] = {
                    "metrics": deep_cell["metrics"],
                    "mean_turnover": deep_cell["mean_turnover"],
                    "train_time_s": deep_cell["train_time_s"],
                    "best_epoch": deep_cell["history"]["best_epoch"],
                }
                print(f"    Deep: ES_95={deep_cell['metrics']['es_95']:.3f}  "
                      f"turnover={deep_cell['mean_turnover']:.3f}  "
                      f"({dt_deep:.0f}s)", flush=True)
                torch.save(
                    deep_cell["pnl"].detach().float().cpu(),
                    self.save_dir / f"pareto_A_n{n_steps}_cost{cost_lambda}_deep_pnl.pt",
                )

                # Incremental save after every cell
                self.save_results(results, "part_A")

                # ETA
                elapsed = time.time() - t_start
                eta_min = elapsed / cell_count * (n_cells - cell_count) / 60.0
                print(f"    elapsed {elapsed/60:.1f} min  |  ETA {eta_min:.1f} min",
                      flush=True)

            # Print running table after each frequency row
            self._print_running_row(results, freq_values, cost_values, i + 1)

            del data
            gc.collect()

        return results

    def _print_running_row(
        self, results: dict, freq_values: list, cost_values: list, n_done: int,
    ) -> None:
        """Compact snapshot of ES_95 so far."""
        print("\n  Running Part A snapshot (ES_95):", flush=True)
        for strat in ("bs", "deep"):
            row = f"    {strat:>4s}  "
            for n in freq_values[:n_done]:
                row += f"n={n:<4d}["
                for c in cost_values:
                    v = results[strat][n][c]["metrics"]["es_95"]
                    row += f"{v:>6.2f} "
                row += "]  "
            print(row, flush=True)

    # ------------------------------------------------------------------
    # Part B: Pareto front over risk objectives
    # ------------------------------------------------------------------

    def run_part_B(
        self,
        objectives: list | None = None,
        n_steps: int = 100,
        cost_lambda: float = 0.001,
        epochs: int = 200,
    ) -> dict:
        """Train multiple deep hedgers with different objectives."""
        if objectives is None:
            objectives = [
                ("es", {"alpha": 0.50}),
                ("es", {"alpha": 0.90}),
                ("es", {"alpha": 0.95}),
                ("es", {"alpha": 0.99}),
                ("entropic", {"lam": 0.1}),
                ("entropic", {"lam": 1.0}),
                ("entropic", {"lam": 5.0}),
                ("mse", {}),
            ]

        print(f"\n--- Part B: Pareto sweep, n_steps={n_steps}, "
              f"cost={cost_lambda} ---", flush=True)

        data_seed = 2024
        t_gen = time.time()
        data = self.generate_paths(n_steps=n_steps, seed=data_seed)
        print(f"  Paths generated in {time.time()-t_gen:.1f}s, "
              f"p0={data['p0']:.4f}", flush=True)

        results: dict[str, Any] = {
            "config": {
                "n_steps": n_steps,
                "cost_lambda": cost_lambda,
                "epochs": epochs,
                "objectives": [[n, k] for n, k in objectives],
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "H": self.H, "eta": self.eta, "rho": self.rho, "xi0": self.xi0,
            },
        }

        # BS reference
        bs_cell = self.run_bs_delta(data, n_steps, cost_lambda)
        results["bs"] = {
            "metrics": bs_cell["metrics"],
            "mean_turnover": bs_cell["mean_turnover"],
        }
        torch.save(
            bs_cell["pnl"].detach().float().cpu(),
            self.save_dir / "pareto_B_bs_pnl.pt",
        )
        print(f"  BS:  ES_95={bs_cell['metrics']['es_95']:.3f}  "
              f"mean_pnl={bs_cell['metrics']['mean_pnl']:+.3f}", flush=True)

        for k, (name, kwargs) in enumerate(objectives):
            tag = make_objective_tag(name, kwargs)
            print(f"\n  Training {tag}  [{k+1}/{len(objectives)}]", flush=True)
            seed = 2024 + k + 1
            t0 = time.time()
            out = train_deep_hedger_with_objective(
                data["S_train"], data["S_val"],
                objective_name=name, objective_kwargs=kwargs,
                cost_lambda=cost_lambda, p0=data["p0"],
                K=self.K, T=self.T, S0=self.S0,
                epochs=epochs, seed=seed, device=self.device,
            )
            model = out["model"]
            pnl = evaluate_deep_hedger(
                model, data["S_test"],
                K=self.K, T=self.T, S0=self.S0,
                p0=data["p0"], cost_lambda=cost_lambda,
            )
            model.eval()
            with torch.no_grad():
                deltas = model.hedge_paths(data["S_test"], self.T, self.S0)
            deltas = deltas.to(data["S_test"].dtype)
            turnover = _mean_turnover(deltas)
            metrics = compute_all_metrics(pnl)

            results[tag] = {
                "objective": name,
                "objective_kwargs": kwargs,
                "metrics": metrics,
                "mean_turnover": turnover,
                "train_time_s": out["train_time_s"],
                "best_epoch": out["history"]["best_epoch"],
            }
            print(
                f"    ES_95={metrics['es_95']:.3f}  "
                f"mean_pnl={metrics['mean_pnl']:+.3f}  "
                f"std={metrics['std_pnl']:.3f}  "
                f"turnover={turnover:.3f}  "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

            torch.save(
                pnl.detach().float().cpu(),
                self.save_dir / f"pareto_B_{tag}_pnl.pt",
            )

            del model
            gc.collect()

            # Incremental save after every hedger
            self.save_results(results, "part_B")

        del data
        gc.collect()
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, all_results: dict, tag: str) -> None:
        """Write scalar metrics to JSON (drops tensors)."""
        path = self.save_dir / f"pareto_{tag}_results.json"
        out = _strip_for_json(all_results)
        out = _normalise_keys(out)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    def print_part_A_tables(self, part_A_results: dict) -> None:
        freq_values = part_A_results["config"]["freq_values"]
        cost_values = part_A_results["config"]["cost_values"]

        for strat in ("bs", "deep"):
            label = "BS Delta" if strat == "bs" else "Deep Hedger"
            print(f"\n{label} — ES_95 by (n_steps, lambda):", flush=True)
            header = "  n_steps  " + "  ".join(f"lam={c:<7.4f}" for c in cost_values)
            print(header, flush=True)
            for n in freq_values:
                row = f"  {n:>7d}  "
                for c in cost_values:
                    v = part_A_results[strat][n][c]["metrics"]["es_95"]
                    row += f"{v:>11.3f}  "
                print(row, flush=True)

            print(f"\n{label} — mean turnover:", flush=True)
            print(header, flush=True)
            for n in freq_values:
                row = f"  {n:>7d}  "
                for c in cost_values:
                    v = part_A_results[strat][n][c]["mean_turnover"]
                    row += f"{v:>11.3f}  "
                print(row, flush=True)

    def print_part_B_table(self, part_B_results: dict) -> None:
        print("\nPart B Pareto results:", flush=True)
        header = (
            f"  {'Objective':<18s}  "
            f"{'mean_pnl':>10s}  {'std':>8s}  "
            f"{'ES_95':>8s}  {'ES_99':>8s}  "
            f"{'ent(1)':>8s}  {'turn':>8s}"
        )
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        order = ["bs"] + [k for k in part_B_results if k not in ("bs", "config")]
        for key in order:
            if key == "config":
                continue
            r = part_B_results[key]
            if "metrics" not in r:
                continue
            m = r["metrics"]
            t = r.get("mean_turnover", float("nan"))
            label = "BS delta" if key == "bs" else key
            print(
                f"  {label:<18s}  "
                f"{m['mean_pnl']:>+10.3f}  "
                f"{m['std_pnl']:>8.3f}  "
                f"{m['es_95']:>8.3f}  "
                f"{m['es_99']:>8.3f}  "
                f"{m['entropic_1']:>8.3f}  "
                f"{t:>8.3f}",
                flush=True,
            )


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto / H2 experiment")
    parser.add_argument("--part", choices=["A", "B", "all"], default="all")
    parser.add_argument(
        "--full", action="store_true",
        help="Use full 4x4 grid and 8 objectives "
             "(default: reduced 3x3 grid and 5 objectives)",
    )
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=30_000)
    parser.add_argument("--part-a-epochs", type=int, default=100)
    parser.add_argument("--part-b-epochs", type=int, default=150)
    args = parser.parse_args()

    exp = ParetoExperiment(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        save_dir=FIGURE_DIR,
    )

    print("=" * 65, flush=True)
    print("  Pareto Front / H2 Factorial Experiment", flush=True)
    print("=" * 65, flush=True)

    if args.full:
        freq_values = [25, 50, 100, 200]
        cost_values = [0.0, 0.0005, 0.001, 0.002]
        objectives = None  # default 8
    else:
        freq_values = [50, 100, 200]
        cost_values = [0.0, 0.001, 0.002]
        objectives = [
            ("es", {"alpha": 0.50}),
            ("es", {"alpha": 0.90}),
            ("es", {"alpha": 0.95}),
            ("es", {"alpha": 0.99}),
            ("mse", {}),
        ]

    print(f"  grid: {'FULL' if args.full else 'REDUCED'}", flush=True)
    print(f"  n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}", flush=True)
    print(f"  Part A freq: {freq_values}", flush=True)
    print(f"  Part A cost: {cost_values}", flush=True)
    print(f"  Part A epochs: {args.part_a_epochs}", flush=True)
    print(f"  Part B epochs: {args.part_b_epochs}", flush=True)

    if args.part in ("A", "all"):
        t0 = time.time()
        rA = exp.run_part_A(
            freq_values=freq_values,
            cost_values=cost_values,
            epochs=args.part_a_epochs,
        )
        print(f"\n  Part A total time: {(time.time()-t0)/60:.1f} min", flush=True)
        exp.print_part_A_tables(rA)

    if args.part in ("B", "all"):
        t0 = time.time()
        rB = exp.run_part_B(
            objectives=objectives,
            epochs=args.part_b_epochs,
        )
        print(f"\n  Part B total time: {(time.time()-t0)/60:.1f} min", flush=True)
        exp.print_part_B_table(rB)

    print("\n" + "=" * 65, flush=True)
    print("  PARETO EXPERIMENT COMPLETE", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
