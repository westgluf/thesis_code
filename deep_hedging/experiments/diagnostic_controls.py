#!/usr/bin/env python
"""
Diagnostic controls: decompose the deep hedging advantage into
objective, stochastic-volatility, and roughness components.

Four experiments:
  A — eta=0 sanity check (deterministic variance)
  B — eta-sweep at fixed H=0.07
  C — objective ablation (MSE vs Mean vs ES)
  D — joint (H, eta) grid

Run:
    python -u -m deep_hedging.experiments.diagnostic_controls
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import (
    DeepHedgerFNN,
    train_deep_hedger,
    evaluate_deep_hedger,
)
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics, expected_shortfall

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

# Colours
C_BS = "#2196F3"
C_DH = "#4CAF50"
C_DH2 = "#81C784"
C_DH3 = "#A5D6A7"
C_GAM = "#9C27B0"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _run_single_point(
    H: float, eta: float, rho: float = -0.7, xi0: float = 0.235 ** 2,
    S0: float = 100.0, K: float = 100.0, T: float = 1.0,
    n_steps: int = 100,
    n_train: int = 60_000, n_val: int = 10_000, n_test: int = 30_000,
    epochs: int = 150, patience: int = 25,
    risk_fn=None, risk_label: str = "es",
    seed: int = 2024, verbose: bool = False,
) -> dict[str, Any]:
    """Simulate, train, evaluate — one (H, eta) point."""
    sigma = math.sqrt(xi0)
    total = n_train + n_val + n_test

    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    S_all, V_all, _ = sim.simulate(n_paths=total, S0=S0, seed=seed)
    S_tr, S_va, S_te = S_all[:n_train], S_all[n_train:n_train + n_val], S_all[n_train + n_val:]
    del S_all, V_all
    gc.collect()

    payoff_tr = compute_payoff(S_tr, K, "call")
    p0 = float(payoff_tr.mean())

    # BS delta
    bs = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas_bs = bs.hedge_paths(S_te)
    payoff_te = compute_payoff(S_te, K, "call")
    pnl_bs = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, 0.0)
    es95_bs = float(expected_shortfall(pnl_bs, 0.95))

    # Deep hedger
    t0 = time.perf_counter()
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    history = train_deep_hedger(
        model, S_tr, S_va,
        K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0,
        risk_fn=risk_fn, alpha=0.95,
        lr=1e-3, batch_size=2048, epochs=epochs, patience=patience,
        verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    pnl_dh = evaluate_deep_hedger(model, S_te, K=K, T=T, S0=S0, p0=p0, cost_lambda=0.0)
    es95_dh = float(expected_shortfall(pnl_dh, 0.95))
    gamma = es95_bs - es95_dh

    del S_tr, S_va, model
    gc.collect()

    return {
        "H": H, "eta": eta, "risk_label": risk_label,
        "p0": p0, "es95_bs": es95_bs, "es95_dh": es95_dh,
        "gamma": gamma, "training_time_s": train_time,
        "bs_metrics": compute_all_metrics(pnl_bs),
        "dh_metrics": compute_all_metrics(pnl_dh),
        "pnl_bs": pnl_bs.detach(), "pnl_dh": pnl_dh.detach(),
        "history": history,
    }


class DiagnosticControlsExperiment:
    """Decomposition of the deep hedging advantage."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.results: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Experiment A: eta=0 control
    # -------------------------------------------------------------------

    def run_experiment_A(self, n_train=60_000, n_val=10_000, n_test=30_000,
                         epochs=150, seed=2024) -> dict:
        """eta=0: rBergomi collapses to GBM. BS should be near-optimal."""
        print("\n  [A] eta=0 control ...", end="", flush=True)
        r = _run_single_point(
            H=0.07, eta=0.0, n_train=n_train, n_val=n_val, n_test=n_test,
            epochs=epochs, seed=seed,
        )
        print(f"  Gamma_A = {r['gamma']:+.3f}  (ES_BS={r['es95_bs']:.3f}, ES_DH={r['es95_dh']:.3f})", flush=True)
        self.results["A"] = r
        return r

    # -------------------------------------------------------------------
    # Experiment A': eta=0, MSE objective (fourth 2x2 factorial cell)
    # -------------------------------------------------------------------

    def run_experiment_A_prime(self, n_train=60_000, n_val=10_000, n_test=30_000,
                               epochs=150, seed=2024) -> dict:
        """Experiment A': DH-MSE at eta=0, H=0.07.

        This is the fourth cell of the 2x2 (eta, objective) factorial needed
        to close the decomposition. Matches the seed / n_test / epochs of
        Experiment A so the two are directly comparable. The ONLY difference
        from A is the risk function: MSE instead of smooth CVaR.
        """
        print("\n  [A'] eta=0, MSE objective ...", end="", flush=True)
        r = _run_single_point(
            H=0.07, eta=0.0, n_train=n_train, n_val=n_val, n_test=n_test,
            epochs=epochs, seed=seed,
            risk_fn=lambda pnl: (pnl ** 2).mean(),
            risk_label="mse",
        )
        print(f"  Gamma_A' = {r['gamma']:+.3f}  (ES_BS={r['es95_bs']:.3f}, ES_DH_MSE={r['es95_dh']:.3f})", flush=True)
        self.results["A_prime"] = r
        return r

    # -------------------------------------------------------------------
    # Experiment B: eta-sweep
    # -------------------------------------------------------------------

    def run_experiment_B(self, eta_values=None, H=0.07,
                         n_train=60_000, n_val=10_000, n_test=30_000,
                         epochs=150, seed=2024) -> list[dict]:
        """eta-sweep at fixed H."""
        if eta_values is None:
            eta_values = [0.1, 0.5, 1.0, 1.9, 3.0]
        results = []
        for i, eta in enumerate(eta_values):
            print(f"\n  [B] eta={eta:.1f}  ({i+1}/{len(eta_values)}) ...", end="", flush=True)
            r = _run_single_point(
                H=H, eta=eta, n_train=n_train, n_val=n_val, n_test=n_test,
                epochs=epochs, seed=seed + i,
            )
            print(f"  Gamma = {r['gamma']:+.3f}", flush=True)
            results.append(r)
        self.results["B"] = results
        return results

    # -------------------------------------------------------------------
    # Experiment C: objective ablation
    # -------------------------------------------------------------------

    def run_experiment_C(self, H=0.07, eta=1.9,
                         n_train=60_000, n_val=10_000, n_test=30_000,
                         epochs=150, seed=2024) -> dict:
        """Train DH-MSE, DH-Mean, DH-ES on same data, compare ES_95."""
        sigma = math.sqrt(0.235 ** 2)
        total = n_train + n_val + n_test

        print("\n  [C] Generating paths ...", end="", flush=True)
        sim = DifferentiableRoughBergomi(n_steps=100, T=1.0, H=H, eta=eta, rho=-0.7, xi0=0.235**2)
        S_all, _, _ = sim.simulate(n_paths=total, S0=100.0, seed=seed)
        S_tr, S_va, S_te = S_all[:n_train], S_all[n_train:n_train+n_val], S_all[n_train+n_val:]
        del S_all; gc.collect()

        payoff_tr = compute_payoff(S_tr, 100.0, "call")
        p0 = float(payoff_tr.mean())

        # BS delta
        bs = BlackScholesDelta(sigma=sigma, K=100.0, T=1.0)
        deltas_bs = bs.hedge_paths(S_te)
        payoff_te = compute_payoff(S_te, 100.0, "call")
        pnl_bs = compute_hedging_pnl(S_te, deltas_bs, payoff_te, p0, 0.0)
        es95_bs = float(expected_shortfall(pnl_bs, 0.95))

        loss_fns = {
            "dh_mse":  lambda pnl: (pnl ** 2).mean(),
            "dh_mean": lambda pnl: (-pnl).mean(),
            "dh_es":   None,  # default smooth CVaR
        }

        out: dict[str, Any] = {"bs": {"es95": es95_bs, "pnl": pnl_bs.detach()}}

        for label, risk_fn in loss_fns.items():
            print(f"\n  [C] Training {label} ...", end="", flush=True)
            model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
            history = train_deep_hedger(
                model, S_tr, S_va,
                K=100.0, T=1.0, S0=100.0, p0=p0, cost_lambda=0.0,
                risk_fn=risk_fn, alpha=0.95,
                lr=1e-3, batch_size=2048, epochs=epochs, patience=25,
                verbose=False,
            )
            pnl = evaluate_deep_hedger(model, S_te, K=100.0, T=1.0, S0=100.0, p0=p0)
            es = float(expected_shortfall(pnl, 0.95))
            print(f"  ES_95 = {es:.3f}", flush=True)
            out[label] = {"es95": es, "pnl": pnl.detach(), "history": history}
            del model; gc.collect()

        del S_tr, S_va; gc.collect()

        # Decomposition
        es_mse = out["dh_mse"]["es95"]
        es_es = out["dh_es"]["es95"]
        out["gamma_decomposition"] = {
            "total": es95_bs - es_es,
            "objective": es_mse - es_es,
            "residual": es95_bs - es_mse,
        }
        self.results["C"] = out
        return out

    # -------------------------------------------------------------------
    # Experiment D: joint (H, eta) grid
    # -------------------------------------------------------------------

    def run_experiment_D(self, H_values=None, eta_values=None,
                         n_train=50_000, n_val=10_000, n_test=20_000,
                         epochs=100, seed=2024) -> list[dict]:
        """Joint (H, eta) grid."""
        if H_values is None:
            H_values = [0.05, 0.2, 0.5]
        if eta_values is None:
            eta_values = [0.5, 1.9, 3.0]

        results = []
        total = len(H_values) * len(eta_values)
        k = 0
        for i, H in enumerate(H_values):
            for j, eta in enumerate(eta_values):
                k += 1
                print(f"\n  [D] H={H:.2f}, eta={eta:.1f}  ({k}/{total}) ...", end="", flush=True)
                r = _run_single_point(
                    H=H, eta=eta, n_train=n_train, n_val=n_val, n_test=n_test,
                    epochs=epochs, seed=seed + k,
                )
                print(f"  Gamma = {r['gamma']:+.3f}", flush=True)
                results.append(r)
        self.results["D"] = results
        return results

    # -------------------------------------------------------------------
    # Run all
    # -------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.run_experiment_A()
        self.run_experiment_A_prime()
        self.run_experiment_B()
        self.run_experiment_C()
        self.run_experiment_D()
        total = time.perf_counter() - t0
        print(f"\n  Total time: {total/60:.1f} min", flush=True)
        return self.results

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save_results(self, path: str | Path = "figures/diagnostic_controls_results.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _strip(obj):
            if isinstance(obj, torch.Tensor):
                return None  # skip large tensors
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if _strip(v) is not None}
            if isinstance(obj, list):
                return [_strip(v) for v in obj]
            if isinstance(obj, (float, int, str, bool)):
                return obj
            if isinstance(obj, np.floating):
                return float(obj)
            return None

        with open(path, "w") as f:
            json.dump(_strip(self.results), f, indent=2)
        print(f"  Saved {path}", flush=True)

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------

    def generate_report(self) -> str:
        lines = ["", "=" * 55, "  DIAGNOSTIC CONTROLS REPORT", "=" * 55]

        # A
        if "A" in self.results:
            a = self.results["A"]
            lines += [
                "", "EXPERIMENT A (eta=0 sanity check)",
                f"  ES_95 BS:   {a['es95_bs']:.3f}",
                f"  ES_95 DH:   {a['es95_dh']:.3f}",
                f"  Gamma_A:    {a['gamma']:+.3f}",
            ]
            if abs(a["gamma"]) < 0.3:
                lines.append("  --> Near-zero: BS is near-optimal at eta=0.")
            else:
                lines.append(f"  --> Non-zero ({a['gamma']:+.3f}): objective/architecture effect exists.")

        # A'
        if "A_prime" in self.results:
            ap = self.results["A_prime"]
            lines += [
                "", "EXPERIMENT A' (eta=0 with MSE objective)",
                f"  ES_95 BS:       {ap['es95_bs']:.3f}",
                f"  ES_95 DH-MSE:   {ap['es95_dh']:.3f}",
                f"  Gamma_A':       {ap['gamma']:+.3f}",
            ]

        # B
        if "B" in self.results:
            lines += ["", "EXPERIMENT B (eta-sweep at H=0.07)"]
            etas = [r["eta"] for r in self.results["B"]]
            gammas = [r["gamma"] for r in self.results["B"]]
            for eta, g in zip(etas, gammas):
                lines.append(f"  eta={eta:.1f}:  Gamma = {g:+.3f}")
            if len(etas) >= 2:
                slope = (gammas[-1] - gammas[0]) / (etas[-1] - etas[0])
                lines.append(f"  Approx slope dGamma/deta = {slope:.3f}")

        # C
        if "C" in self.results:
            c = self.results["C"]
            lines += [
                "", "EXPERIMENT C (objective ablation, H=0.07, eta=1.9)",
                f"  ES_95 BS:       {c['bs']['es95']:.3f}",
                f"  ES_95 DH-MSE:   {c['dh_mse']['es95']:.3f}",
                f"  ES_95 DH-Mean:  {c['dh_mean']['es95']:.3f}",
                f"  ES_95 DH-ES:    {c['dh_es']['es95']:.3f}",
            ]
            d = c["gamma_decomposition"]
            lines += [
                f"  Gamma_total:     {d['total']:+.3f}",
                f"  Gamma_objective: {d['objective']:+.3f} (ES vs MSE training)",
                f"  Gamma_residual:  {d['residual']:+.3f} (architecture + data)",
            ]

        # D
        if "D" in self.results:
            lines += ["", "EXPERIMENT D (joint H x eta grid)"]
            D = self.results["D"]
            Hs = sorted(set(r["H"] for r in D))
            etas_d = sorted(set(r["eta"] for r in D))
            grid = {}
            for r in D:
                grid[(r["H"], r["eta"])] = r["gamma"]

            header = "        " + "".join(f"eta={e:.1f}  " for e in etas_d)
            lines.append(header)
            for h in Hs:
                row = f"  H={h:.2f}  "
                for e in etas_d:
                    row += f"{grid.get((h, e), float('nan')):+7.3f}  "
                lines.append(row)

            # Row means (eta effect) and col means (H effect)
            row_means = [np.mean([grid.get((h, e), 0) for e in etas_d]) for h in Hs]
            col_means = [np.mean([grid.get((h, e), 0) for h in Hs]) for e in etas_d]
            lines.append(f"  Row-mean var (H effect):   {np.var(row_means):.4f}")
            lines.append(f"  Col-mean var (eta effect): {np.var(col_means):.4f}")

        # Summary decomposition
        lines += ["", "=" * 55, "  DECOMPOSITION SUMMARY", "=" * 55]
        if "C" in self.results and "A" in self.results:
            d = self.results["C"]["gamma_decomposition"]
            ga = self.results["A"]["gamma"]
            total = d["total"]
            obj = d["objective"]
            arch = ga  # advantage even at eta=0
            stoch = max(0, total - obj - arch)
            lines += [
                f"  Gamma_total     = {total:+.3f}",
                f"  Gamma_arch      ~ {arch:+.3f} (from eta=0 control)",
                f"  Gamma_objective ~ {obj:+.3f} (ES vs MSE training)",
                f"  Gamma_stochvol  ~ {stoch:+.3f} (residual)",
            ]

        report = "\n".join(lines)
        print(report, flush=True)
        return report

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------

    def generate_figures(self, save_dir: str | Path = "figures") -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._fig_A(save_dir)
        self._fig_B(save_dir)
        self._fig_C(save_dir)
        self._fig_D(save_dir)
        self._fig_decomposition(save_dir)

    def _fig_A(self, d: Path) -> None:
        if "A" not in self.results:
            return
        a = self.results["A"]
        fig, ax = plt.subplots(figsize=(8, 5))
        pbs = a["pnl_bs"].float().numpy()
        pdh = a["pnl_dh"].float().numpy()
        ax.hist(pbs, bins=80, alpha=0.5, density=True, color=C_BS, label="BS Delta")
        ax.hist(pdh, bins=80, alpha=0.5, density=True, color=C_DH, label="Deep Hedger")
        ax.axvline(-a["es95_bs"], color=C_BS, ls=":", lw=1.5)
        ax.axvline(-a["es95_dh"], color=C_DH, ls=":", lw=1.5)
        ax.set_xlabel("P&L")
        ax.set_ylabel("Density")
        ax.set_title(f"$\\eta=0$ Control (deterministic variance)   $\\Gamma_A = {a['gamma']:+.3f}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "fig_diagnostic_A_eta_zero.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_diagnostic_A_eta_zero.png'}", flush=True)

    def _fig_B(self, d: Path) -> None:
        if "B" not in self.results:
            return
        B = self.results["B"]
        etas = [r["eta"] for r in B]
        gammas = [r["gamma"] for r in B]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(etas, gammas, "D-", color=C_GAM, lw=2, ms=8)
        ax.axhline(0, color="grey", ls="--", lw=0.7)
        ax.set_xlabel("$\\eta$ (vol-of-vol)", fontsize=12)
        ax.set_ylabel("$\\Gamma(\\eta)$", fontsize=12)
        ax.set_title("Advantage Gap vs Vol-of-Vol ($H=0.07$)", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / "fig_diagnostic_B_eta_sweep.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_diagnostic_B_eta_sweep.png'}", flush=True)

    def _fig_C(self, d: Path) -> None:
        if "C" not in self.results:
            return
        c = self.results["C"]
        names = ["BS Delta", "DH-MSE", "DH-Mean", "DH-ES"]
        vals = [c["bs"]["es95"], c["dh_mse"]["es95"], c["dh_mean"]["es95"], c["dh_es"]["es95"]]
        colours = [C_BS, C_DH3, C_DH2, C_DH]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, vals, color=colours, alpha=0.85, edgecolor="k", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.2f}",
                    ha="center", fontsize=10)
        ax.set_ylabel("ES$_{0.95}$", fontsize=12)
        ax.set_title("Objective Ablation ($H=0.07$, $\\eta=1.9$)", fontsize=13)
        decomp = c["gamma_decomposition"]
        ax.annotate(
            f"$\\Gamma_{{obj}}$ = {decomp['objective']:.2f}\n"
            f"$\\Gamma_{{resid}}$ = {decomp['residual']:.2f}",
            xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
        )
        fig.tight_layout()
        fig.savefig(d / "fig_diagnostic_C_objective_ablation.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_diagnostic_C_objective_ablation.png'}", flush=True)

    def _fig_D(self, d: Path) -> None:
        if "D" not in self.results:
            return
        D = self.results["D"]
        Hs = sorted(set(r["H"] for r in D))
        etas = sorted(set(r["eta"] for r in D))
        grid = np.full((len(Hs), len(etas)), np.nan)
        for r in D:
            i = Hs.index(r["H"])
            j = etas.index(r["eta"])
            grid[i, j] = r["gamma"]

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, cmap="plasma", aspect="auto", origin="lower")
        ax.set_xticks(range(len(etas)))
        ax.set_xticklabels([f"{e:.1f}" for e in etas])
        ax.set_yticks(range(len(Hs)))
        ax.set_yticklabels([f"{h:.2f}" for h in Hs])
        ax.set_xlabel("$\\eta$", fontsize=12)
        ax.set_ylabel("$H$", fontsize=12)
        ax.set_title("$\\Gamma(H, \\eta)$ = ES$_{95}$(BS) $-$ ES$_{95}$(DH)", fontsize=13)
        for i in range(len(Hs)):
            for j in range(len(etas)):
                ax.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < np.nanmedian(grid) else "black",
                        fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="$\\Gamma$")
        fig.tight_layout()
        fig.savefig(d / "fig_diagnostic_D_grid_heatmap.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_diagnostic_D_grid_heatmap.png'}", flush=True)

    def _fig_decomposition(self, d: Path) -> None:
        if "C" not in self.results or "A" not in self.results:
            return
        dc = self.results["C"]["gamma_decomposition"]
        ga = self.results["A"]["gamma"]
        total = dc["total"]
        obj = dc["objective"]
        arch = ga
        stoch = max(0, total - obj - arch)

        components = [
            ("Architecture\n(eta=0 control)", arch, "#FFB74D"),
            ("Objective\n(ES vs MSE)", obj, "#81C784"),
            ("Stoch vol\n(residual)", stoch, "#64B5F6"),
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        left = 0.0
        for label, val, colour in components:
            ax.barh(0, val, left=left, height=0.5, color=colour,
                    edgecolor="k", lw=0.5, label=f"{label}: {val:.2f}")
            if val > 0.05:
                ax.text(left + val / 2, 0, f"{val:.2f}", ha="center", va="center", fontsize=10)
            left += val
        ax.barh(1, total, height=0.5, color=C_GAM, alpha=0.6,
                edgecolor="k", lw=0.5, label=f"Total: {total:.2f}")
        ax.text(total / 2, 1, f"Total = {total:.2f}", ha="center", va="center", fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Components", "Total $\\Gamma$"])
        ax.set_xlabel("$\\Gamma$ (ES$_{95}$ advantage)", fontsize=12)
        ax.set_title("Decomposition of Deep Hedging Advantage", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0, max(total, left) * 1.15)
        fig.tight_layout()
        fig.savefig(d / "fig_diagnostic_decomposition.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_diagnostic_decomposition.png'}", flush=True)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic controls: decompose the deep hedging advantage.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only this experiment (e.g. 'A_prime'). "
             "Merges result into existing JSON without overwriting other keys.",
    )
    args = parser.parse_args()

    json_path = FIGURE_DIR / "diagnostic_controls_results.json"

    print("=" * 60, flush=True)
    print("  Diagnostic Controls: Decomposing the Advantage", flush=True)
    print("=" * 60, flush=True)

    exp = DiagnosticControlsExperiment()

    if args.only is not None:
        # Run a single experiment and merge into existing JSON
        method_name = f"run_experiment_{args.only}"
        fn = getattr(exp, method_name, None)
        if fn is None:
            raise ValueError(f"No experiment method '{method_name}' found.")
        fn()

        # Merge into existing JSON
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            existing[args.only] = exp.results[args.only]
            exp.results = existing
        exp.save_results(json_path)
        exp.generate_report()
    else:
        exp.run_all()
        exp.generate_report()
        print("\n  Generating figures ...", flush=True)
        exp.generate_figures(FIGURE_DIR)
        exp.save_results(json_path)

    print("\n" + "=" * 60, flush=True)
    print("  DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
