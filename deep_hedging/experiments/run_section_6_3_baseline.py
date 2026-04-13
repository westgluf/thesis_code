#!/usr/bin/env python
"""
Section 6.3 baseline experiment: Deep Hedger vs Delta Hedgers on rBergomi.

Tests Hypothesis H1: under rough volatility dynamics, BS and Heston
delta-hedges have heavier left P&L tails than a deep hedger trained
on the same dynamics.

Run:
    python -m deep_hedging.experiments.run_section_6_3_baseline
"""
from __future__ import annotations

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
from scipy import stats as sp_stats

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta, HestonDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    hedge_paths_deep,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import (
    expected_shortfall,
    compute_all_metrics,
)
from deep_hedging.utils.config import RoughBergomiParams, DatasetConfig

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

# Consistent colour scheme across all figures
COLORS = {
    "BS Delta": "#2196F3",
    "Heston Delta": "#FF9800",
    "Deep Hedger": "#4CAF50",
}


class Section63Experiment:
    """Master experiment for Section 6.3 baseline results."""

    def __init__(
        self,
        params: RoughBergomiParams | None = None,
        dataset_config: DatasetConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.params = params or RoughBergomiParams()
        self.ds = dataset_config or DatasetConfig()
        self.device = device or torch.device("cpu")

        self.K = 100.0
        self.T = self.params.T
        self.S0 = self.params.S0
        self.sigma = math.sqrt(self.params.xi0)  # for BS assumed vol

        # Populated by generate_data
        self.S_train = self.S_val = self.S_test = None
        self.V_train = self.V_val = self.V_test = None
        self.p0: float | None = None

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_data(self, seed: int = 2024) -> None:
        """Generate rBergomi paths and split into train/val/test."""
        p = self.params
        total = self.ds.n_train + self.ds.n_val + self.ds.n_test
        print(f"  Generating {total:,} rBergomi paths "
              f"(H={p.H}, eta={p.eta}, rho={p.rho}) ...")
        t0 = time.perf_counter()

        model = DifferentiableRoughBergomi(
            n_steps=p.n_steps, T=p.T, H=p.H,
            eta=p.eta, rho=p.rho, xi0=p.xi0,
        )
        S_all, V_all, _ = model.simulate(n_paths=total, S0=p.S0, seed=seed)

        n1 = self.ds.n_train
        n2 = n1 + self.ds.n_val
        self.S_train, self.S_val, self.S_test = S_all[:n1], S_all[n1:n2], S_all[n2:]
        self.V_train, self.V_val, self.V_test = V_all[:n1], V_all[n1:n2], V_all[n2:]

        # MC price for p0 (no closed-form under rBergomi)
        payoff_train = compute_payoff(self.S_train, self.K, "call")
        self.p0 = float(payoff_train.mean())
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s.  MC option price p0 = {self.p0:.4f}")

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------

    def run_bs_delta(self, cost_lambda: float = 0.0) -> dict[str, Any]:
        """Run BS delta hedger on test paths."""
        hedger = BlackScholesDelta(sigma=self.sigma, K=self.K, T=self.T)
        deltas = hedger.hedge_paths(self.S_test)
        payoff = compute_payoff(self.S_test, self.K, "call")
        pnl = compute_hedging_pnl(self.S_test, deltas, payoff, self.p0, cost_lambda)
        return {"deltas": deltas, "pnl": pnl, "metrics": compute_all_metrics(pnl)}

    def run_heston_delta(self, cost_lambda: float = 0.0) -> dict[str, Any]:
        """Run Heston plug-in delta hedger on test paths (observes V)."""
        hedger = HestonDelta(K=self.K, T=self.T)
        deltas = hedger.hedge_paths(self.S_test, self.V_test)
        payoff = compute_payoff(self.S_test, self.K, "call")
        pnl = compute_hedging_pnl(self.S_test, deltas, payoff, self.p0, cost_lambda)
        return {"deltas": deltas, "pnl": pnl, "metrics": compute_all_metrics(pnl)}

    def run_deep_hedger(
        self,
        cost_lambda: float = 0.0,
        epochs: int = 200,
        **train_kwargs,
    ) -> dict[str, Any]:
        """Train deep hedger on rBergomi paths, evaluate on test set."""
        model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        defaults = dict(
            K=self.K, T=self.T, S0=self.S0, p0=self.p0,
            cost_lambda=cost_lambda, alpha=0.95,
            lr=1e-3, batch_size=2048, epochs=epochs, patience=30,
            device=self.device, verbose=True,
        )
        defaults.update(train_kwargs)

        history = train_deep_hedger(model, self.S_train, self.S_val, **defaults)

        pnl = evaluate_deep_hedger(
            model, self.S_test, K=self.K, T=self.T,
            S0=self.S0, p0=self.p0, cost_lambda=cost_lambda,
        )
        model.eval()
        with torch.no_grad():
            deltas = hedge_paths_deep(model, self.S_test, self.T, self.S0)
            deltas = deltas.to(self.S_test.dtype)

        return {
            "deltas": deltas, "pnl": pnl,
            "metrics": compute_all_metrics(pnl),
            "history": history, "model": model,
        }

    # ------------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------------

    def run_full_comparison(
        self, cost_lambdas: list[float] | None = None,
    ) -> dict[float, dict[str, Any]]:
        """Run all strategies for each cost level."""
        if cost_lambdas is None:
            cost_lambdas = [0.0, 0.001]

        results: dict[float, dict[str, Any]] = {}
        for lam in cost_lambdas:
            print(f"\n{'='*60}")
            print(f"  Cost level lambda = {lam}")
            print(f"{'='*60}")

            r: dict[str, Any] = {}
            print("\n  --- BS Delta ---")
            r["BS Delta"] = self.run_bs_delta(lam)

            print("\n  --- Heston Delta ---")
            r["Heston Delta"] = self.run_heston_delta(lam)

            print(f"\n  --- Deep Hedger (training, lambda={lam}) ---")
            r["Deep Hedger"] = self.run_deep_hedger(cost_lambda=lam)

            results[lam] = r
        return results

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_results_table(self, results: dict) -> None:
        """Print LaTeX-ready comparison table."""
        header = (
            r"\begin{table}[h]" "\n"
            r"\centering" "\n"
            r"\caption{Section 6.3: Hedging performance under "
            r"rBergomi($H=" + f"{self.params.H}" + r"$)}" "\n"
            r"\begin{tabular}{l|rrrrrr}" "\n"
            r"\hline" "\n"
            r"Strategy & Mean & Std & VaR$_{95}$ & ES$_{95}$ "
            r"& ES$_{99}$ & Entropic \\" "\n"
            r"\hline"
        )
        print(header)

        for lam, strats in results.items():
            label = "Frictionless" if lam == 0.0 else f"With costs ($\\lambda = {lam}$)"
            print(rf"\multicolumn{{7}}{{c}}{{\textit{{{label}}}}} \\")
            print(r"\hline")
            for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
                m = strats[name]["metrics"]
                print(
                    f"{name:15s} & {m['mean_pnl']:7.3f} & {m['std_pnl']:6.3f} "
                    f"& {m['var_95']:6.3f} & {m['es_95']:6.3f} "
                    f"& {m['es_99']:6.3f} & {m['entropic_1']:6.3f} \\\\"
                )
            print(r"\hline")

        print(r"\end{tabular}")
        print(r"\end{table}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def generate_all_figures(
        self, results: dict, save_dir: str | Path = "figures",
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._fig_pnl_histograms(results, save_dir)
        self._fig_tail_comparison(results, save_dir)
        self._fig_qq_plots(results, save_dir)
        self._fig_metrics_bar(results, save_dir)
        self._fig_delta_comparison(results, save_dir)
        self._fig_pnl_over_time(results, save_dir)
        self._fig_training_curve(results, save_dir)

    # --- individual figure methods ---

    def _fig_pnl_histograms(self, results, save_dir):
        """Figure 1: P&L histograms for all strategies."""
        lams = sorted(results.keys())
        fig, axes = plt.subplots(len(lams), 1, figsize=(9, 5 * len(lams)), squeeze=False)
        for ax_row, lam in zip(axes, lams):
            ax = ax_row[0]
            label = "Frictionless" if lam == 0.0 else f"With costs ($\\lambda={lam}$)"
            for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
                pnl = results[lam][name]["pnl"].detach().float().numpy()
                m = results[lam][name]["metrics"]
                ax.hist(pnl, bins=80, alpha=0.45, density=True,
                        color=COLORS[name], label=name)
                ax.axvline(m["mean_pnl"], color=COLORS[name], ls="--", lw=1.2)
                ax.axvline(-m["es_95"], color=COLORS[name], ls=":", lw=1.2)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Terminal P&L")
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
            ax.set_xlim(-30, 20)
        fig.suptitle(f"rBergomi (H={self.params.H}): P&L Distributions", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_pnl_histograms.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_pnl_histograms.png'}")

    def _fig_tail_comparison(self, results, save_dir):
        """Figure 2: Left tail zoom."""
        lam = 0.0
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
            pnl = results[lam][name]["pnl"].detach().float().numpy()
            cutoff = np.percentile(pnl, 10)
            tail = pnl[pnl < cutoff]
            ax.hist(tail, bins=60, alpha=0.5, density=True,
                    color=COLORS[name], label=name)
        ax.set_xlabel("Terminal P&L (left tail)")
        ax.set_ylabel("Density")
        ax.set_title("Left Tail Comparison (frictionless)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_tail_comparison.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_tail_comparison.png'}")

    def _fig_qq_plots(self, results, save_dir):
        """Figure 3: Q-Q plots vs normal."""
        lam = 0.0
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, name in zip(axes, ["BS Delta", "Heston Delta", "Deep Hedger"]):
            pnl = results[lam][name]["pnl"].detach().float().numpy()
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(pnl, dist="norm")
            ax.scatter(osm, osr, s=2, alpha=0.3, color=COLORS[name])
            xlim = ax.get_xlim()
            xs = np.linspace(xlim[0], xlim[1], 50)
            ax.plot(xs, slope * xs + intercept, "k--", lw=1)
            ax.set_title(name)
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
        fig.suptitle("Q-Q Plots vs Normal (frictionless)", y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_qq_plots.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_qq_plots.png'}")

    def _fig_metrics_bar(self, results, save_dir):
        """Figure 4: Grouped bar chart of risk metrics."""
        metric_keys = ["es_95", "es_99", "var_95", "std_pnl"]
        labels = ["ES$_{95}$", "ES$_{99}$", "VaR$_{95}$", "Std"]
        strat_names = ["BS Delta", "Heston Delta", "Deep Hedger"]
        lams = sorted(results.keys())

        n_metrics = len(metric_keys)
        n_groups = len(lams)
        n_strats = len(strat_names)
        width = 0.25
        x = np.arange(n_metrics)

        fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5), squeeze=False)
        for ax_col, lam in zip(axes[0], lams):
            for i, name in enumerate(strat_names):
                m = results[lam][name]["metrics"]
                vals = [m[k] for k in metric_keys]
                ax_col.bar(x + i * width, vals, width,
                           color=COLORS[name], label=name, alpha=0.85)
            ax_col.set_xticks(x + width)
            ax_col.set_xticklabels(labels)
            ax_col.set_ylabel("Risk metric value")
            lbl = "Frictionless" if lam == 0 else f"$\\lambda={lam}$"
            ax_col.set_title(lbl)
            ax_col.legend(fontsize=8)
        fig.suptitle(f"rBergomi (H={self.params.H}): Risk Metrics", y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_metrics_bar.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_metrics_bar.png'}")

    def _fig_delta_comparison(self, results, save_dir):
        """Figure 5: Delta vs spot at three time slices."""
        lam = 0.0
        n = self.S_test.shape[1] - 1
        steps = [n // 4, n // 2, 3 * n // 4]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        idx = slice(None, 2000)  # subsample

        for ax, k in zip(axes, steps):
            S_k = self.S_test[idx, k].detach().float().numpy()
            for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
                d_k = results[lam][name]["deltas"][idx, k].detach().float().numpy()
                ax.scatter(S_k, d_k, s=2, alpha=0.2, color=COLORS[name], label=name)
            ax.set_xlabel("Spot S")
            ax.set_ylabel("Delta")
            t_k = k * self.T / n
            ax.set_title(f"t = {t_k:.2f}")
            ax.legend(fontsize=7, markerscale=4)
        fig.suptitle("Delta vs Spot at Three Time Slices (frictionless)", y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_delta_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_delta_comparison.png'}")

    def _fig_pnl_over_time(self, results, save_dir):
        """Figure 6: Cumulative P&L for 3 sample paths."""
        lam = 0.0
        sample_idx = [0, 1, 2]
        n = self.S_test.shape[1] - 1
        t_grid = np.linspace(0, self.T, n)
        linestyles = ["-", "--", ":"]

        fig, ax = plt.subplots(figsize=(9, 5))
        for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
            deltas = results[lam][name]["deltas"]  # (n_test, n)
            S = self.S_test
            dS = (S[:, 1:] - S[:, :-1])            # (n_test, n)
            cum_gains = torch.cumsum(deltas * dS, dim=1)  # (n_test, n)
            for j, si in enumerate(sample_idx):
                vals = cum_gains[si].detach().float().numpy()
                ax.plot(t_grid, vals, color=COLORS[name],
                        ls=linestyles[j], lw=1.2,
                        label=f"{name} (path {si})" if j == 0 else None)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative hedge gains")
        ax.set_title("Cumulative P&L Evolution (3 sample paths, frictionless)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_pnl_over_time.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_pnl_over_time.png'}")

    def _fig_training_curve(self, results, save_dir):
        """Figure 7: Training curve for the deep hedger on rBergomi."""
        # Use the frictionless deep hedger history if available
        for lam in sorted(results.keys()):
            if "Deep Hedger" in results[lam] and "history" in results[lam]["Deep Hedger"]:
                h = results[lam]["Deep Hedger"]["history"]
                break
        else:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ep = range(1, len(h["train_risk"]) + 1)
        ax.plot(ep, h["train_risk"], label="Train risk", alpha=0.8)
        ax.plot(ep, h["val_risk"], label="Val risk", alpha=0.8)
        ax.axvline(h["best_epoch"], color="grey", ls="--", lw=1,
                   label=f"Best epoch ({h['best_epoch']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth CVaR (risk)")
        ax.set_title("Deep Hedger Training on rBergomi Paths")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "fig_63_training_rbergomi.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {save_dir / 'fig_63_training_rbergomi.png'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Section 6.3 Baseline: Deep Hedger vs Delta Hedgers on rBergomi")
    print("=" * 65)

    exp = Section63Experiment()
    exp.generate_data(seed=2024)

    results = exp.run_full_comparison(cost_lambdas=[0.0, 0.001])

    print("\n" + "=" * 65)
    print("  RESULTS TABLE (LaTeX)")
    print("=" * 65 + "\n")
    exp.print_results_table(results)

    print("\n" + "=" * 65)
    print("  PLAIN-TEXT SUMMARY")
    print("=" * 65)
    for lam, strats in results.items():
        tag = "frictionless" if lam == 0 else f"lambda={lam}"
        print(f"\n  [{tag}]")
        for name in ["BS Delta", "Heston Delta", "Deep Hedger"]:
            m = strats[name]["metrics"]
            print(f"    {name:15s}  mean={m['mean_pnl']:+.3f}  std={m['std_pnl']:.3f}"
                  f"  ES95={m['es_95']:.3f}  ES99={m['es_99']:.3f}"
                  f"  skew={m['skewness']:.3f}")

    print("\n  Generating figures...")
    exp.generate_all_figures(results, save_dir=FIGURE_DIR)

    # Save metrics to JSON
    metrics_out = {}
    for lam, strats in results.items():
        metrics_out[str(lam)] = {name: strats[name]["metrics"] for name in strats}
    json_path = FIGURE_DIR / "section_63_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved to {json_path}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
