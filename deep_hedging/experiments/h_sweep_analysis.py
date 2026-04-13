#!/usr/bin/env python
"""
H-sweep analysis: power-law fit, bootstrap CIs, publication figures.

Loads the JSON output from :class:`HurstSweepExperiment` and produces
statistical analysis and publication-quality figures for Section 6.3.

Run:
    python -u -m deep_hedging.experiments.h_sweep_analysis
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

# Colour scheme (consistent with Prompt 4)
C_BS = "#2196F3"
C_DH = "#4CAF50"
C_FIT = "#E53935"
C_GAM = "#9C27B0"


class HSweepAnalyser:
    """Statistical analysis and visualisation of H-sweep results."""

    def __init__(self, results_path: str | Path = "figures/h_sweep_results.json") -> None:
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.H_values = np.array([r["H"] for r in self.results])
        self.gamma = np.array([r["gamma"] for r in self.results])
        self.es95_bs = np.array([r["bs_metrics"]["es_95"] for r in self.results])
        self.es95_dh = np.array([r["dh_metrics"]["es_95"] for r in self.results])
        self.es99_bs = np.array([r["bs_metrics"]["es_99"] for r in self.results])
        self.es99_dh = np.array([r["dh_metrics"]["es_99"] for r in self.results])
        self.std_bs = np.array([r["bs_metrics"]["std_pnl"] for r in self.results])
        self.std_dh = np.array([r["dh_metrics"]["std_pnl"] for r in self.results])

    def _load_results(self) -> list[dict]:
        with open(self.results_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Power-law fit
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Fit y = c * x^beta via log-log OLS.  Returns (beta, log_c, R2)."""
        lx, ly = np.log(x), np.log(y)
        beta, log_c = np.polyfit(lx, ly, 1)
        y_hat = beta * lx + log_c
        ss_res = np.sum((ly - y_hat) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        return float(beta), float(log_c), float(r2)

    def fit_power_law(self, exclude_h05: bool = True) -> dict[str, Any]:
        """Fit Gamma(H) = c * (1/2 - H)^beta via log-log regression."""
        mask = self.gamma > 0
        if exclude_h05:
            mask &= self.H_values < 0.499
        x = 0.5 - self.H_values[mask]
        y = self.gamma[mask]

        if len(x) < 3:
            return {"beta": float("nan"), "c": float("nan"), "r_squared": float("nan"),
                    "n_points": int(np.sum(mask)), "fitted_gamma": self.gamma * np.nan}

        beta, log_c, r2 = self._fit_loglog(x, y)
        c = math.exp(log_c)

        # Fitted curve over all H (including excluded)
        x_all = np.clip(0.5 - self.H_values, 1e-12, None)
        fitted = c * x_all ** beta

        return {
            "beta": beta, "c": c, "log_c": log_c, "r_squared": r2,
            "n_points": int(np.sum(mask)),
            "fitted_gamma": fitted,
        }

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_confidence(self, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, Any]:
        """Bootstrap CIs for beta and c."""
        mask = (self.gamma > 0) & (self.H_values < 0.499)
        x = 0.5 - self.H_values[mask]
        y = self.gamma[mask]
        n = len(x)
        if n < 3:
            return {"beta_samples": np.array([]), "c_samples": np.array([]),
                    "beta_ci": (float("nan"), float("nan")),
                    "c_ci": (float("nan"), float("nan"))}

        rng = np.random.default_rng(seed)
        betas, cs = [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            try:
                b, lc, _ = self._fit_loglog(x[idx], y[idx])
                betas.append(b)
                cs.append(math.exp(lc))
            except Exception:
                continue

        betas, cs = np.array(betas), np.array(cs)
        return {
            "beta_samples": betas,
            "c_samples": cs,
            "beta_ci": (float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))),
            "c_ci": (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))),
        }

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def test_phase_transition(self) -> dict[str, Any]:
        """Test for piecewise-linear breakpoint in log-log space."""
        mask = (self.gamma > 0) & (self.H_values < 0.499)
        x = np.log(0.5 - self.H_values[mask])
        y = np.log(self.gamma[mask])
        n = len(x)

        # Single fit RSS
        beta_s, lc_s = np.polyfit(x, y, 1)
        rss_single = float(np.sum((y - (beta_s * x + lc_s)) ** 2))

        best_rss, best_k = rss_single, -1
        betas_low, betas_high = beta_s, beta_s
        for k in range(2, n - 2):
            try:
                b1, c1 = np.polyfit(x[:k], y[:k], 1)
                b2, c2 = np.polyfit(x[k:], y[k:], 1)
                rss = float(np.sum((y[:k] - (b1 * x[:k] + c1)) ** 2) +
                            np.sum((y[k:] - (b2 * x[k:] + c2)) ** 2))
                if rss < best_rss:
                    best_rss, best_k = rss, k
                    betas_low, betas_high = float(b1), float(b2)
            except Exception:
                continue

        improvement = (rss_single - best_rss) / max(rss_single, 1e-30)
        H_break = float(self.H_values[mask][best_k]) if best_k >= 0 else float("nan")

        return {
            "break_point": H_break,
            "rss_single": rss_single,
            "rss_piecewise": best_rss,
            "improvement": float(improvement),
            "beta_low": betas_low,
            "beta_high": betas_high,
            "significant": improvement > 0.20 and best_k >= 2 and best_k <= n - 3,
        }

    # ------------------------------------------------------------------
    # Relative gap
    # ------------------------------------------------------------------

    def compute_relative_gap(self) -> np.ndarray:
        """Γ(H) / ES_95^BS(H) as a fraction."""
        return self.gamma / np.clip(self.es95_bs, 1e-12, None)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def generate_all_figures(self, save_dir: str | Path = "figures") -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fit = self.fit_power_law()
        boot = self.bootstrap_confidence()
        phase = self.test_phase_transition()

        self._fig_es95_es99(save_dir)
        self._fig_gamma_loglog(save_dir, fit, boot)
        self._fig_relative_gap(save_dir)
        self._fig_phase_transition(save_dir, phase)
        self._fig_summary(save_dir, fit)

    def _fig_es95_es99(self, d: Path) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
        for ax, bs, dh, label in [
            (ax1, self.es95_bs, self.es95_dh, "ES$_{0.95}$"),
            (ax2, self.es99_bs, self.es99_dh, "ES$_{0.99}$"),
        ]:
            ax.plot(self.H_values, bs, "o-", color=C_BS, label="BS Delta", lw=2, ms=6)
            ax.plot(self.H_values, dh, "s-", color=C_DH, label="Deep Hedger", lw=2, ms=6)
            ax.fill_between(self.H_values, dh, bs, alpha=0.15, color=C_GAM)
            ax.axvline(0.07, color="grey", ls=":", lw=0.8, label="$H=0.07$")
            ax.set_ylabel(label, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        ax2.set_xlabel("Hurst parameter $H$", fontsize=12)
        fig.suptitle("Risk Measures vs Roughness", fontsize=13, y=0.98)
        fig.tight_layout()
        fig.savefig(d / "fig_h_sweep_es95_es99.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {d / 'fig_h_sweep_es95_es99.png'}", flush=True)

    def _fig_gamma_loglog(self, d: Path, fit: dict, boot: dict) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

        # Top: linear scale
        ax1.plot(self.H_values, self.gamma, "D-", color=C_GAM, lw=2, ms=7)
        if np.isfinite(fit.get("beta", float("nan"))):
            ax1.plot(self.H_values, fit["fitted_gamma"], "--", color=C_FIT, lw=1.5,
                     label=f"Fit: $\\beta$={fit['beta']:.2f}, $R^2$={fit['r_squared']:.2f}")
        ax1.axhline(0, color="grey", ls="--", lw=0.7)
        ax1.axvline(0.07, color="grey", ls=":", lw=0.8)
        ax1.set_xlabel("$H$", fontsize=12)
        ax1.set_ylabel("$\\Gamma(H)$", fontsize=12)
        ax1.set_title("Advantage Gap (linear scale)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom: log-log
        mask = (self.gamma > 0) & (self.H_values < 0.499)
        x_plot = 0.5 - self.H_values[mask]
        y_plot = self.gamma[mask]
        ax2.scatter(x_plot, y_plot, color=C_GAM, s=50, zorder=5, edgecolors="k", lw=0.5)
        if np.isfinite(fit.get("beta", float("nan"))):
            xs = np.linspace(x_plot.min() * 0.8, x_plot.max() * 1.2, 100)
            ax2.plot(xs, fit["c"] * xs ** fit["beta"], "--", color=C_FIT, lw=1.5,
                     label=f"$\\Gamma = {fit['c']:.2f}\\,(1/2-H)^{{{fit['beta']:.2f}}}$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("$1/2 - H$", fontsize=12)
        ax2.set_ylabel("$\\Gamma(H)$", fontsize=12)
        ax2.set_title("Advantage Gap (log-log scale)", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(d / "fig_h_sweep_gamma_loglog.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_h_sweep_gamma_loglog.png'}", flush=True)

    def _fig_relative_gap(self, d: Path) -> None:
        rel = self.compute_relative_gap() * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.H_values, rel, "D-", color=C_GAM, lw=2, ms=7)
        imax = np.argmax(rel)
        ax.plot(self.H_values[imax], rel[imax], "*", color=C_FIT, ms=18, zorder=5,
                label=f"Max = {rel[imax]:.1f}% at $H={self.H_values[imax]:.2f}$")
        # Mark H=0.07
        i07 = np.argmin(np.abs(self.H_values - 0.07))
        ax.axhline(rel[i07], color="grey", ls=":", lw=0.8,
                   label=f"$H=0.07$: {rel[i07]:.1f}%")
        ax.set_xlabel("Hurst parameter $H$", fontsize=12)
        ax.set_ylabel("Relative advantage (%)", fontsize=12)
        ax.set_title("Deep Hedging Relative Improvement over BS Delta", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / "fig_h_sweep_relative_gap.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_h_sweep_relative_gap.png'}", flush=True)

    def _fig_phase_transition(self, d: Path, phase: dict) -> None:
        mask = (self.gamma > 0) & (self.H_values < 0.499)
        x = np.log(0.5 - self.H_values[mask])
        y = np.log(self.gamma[mask])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color=C_GAM, s=50, zorder=5, edgecolors="k", lw=0.5)

        # Single fit
        b_s, c_s = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 50)
        ax.plot(xs, b_s * xs + c_s, "--", color=C_FIT, lw=1.5,
                label=f"Single: $\\beta$={b_s:.2f}")

        if phase["significant"]:
            # Find break index in masked data
            bp = phase["break_point"]
            H_masked = self.H_values[mask]
            k = np.argmin(np.abs(H_masked - bp))
            b1, c1 = np.polyfit(x[:k], y[:k], 1)
            b2, c2 = np.polyfit(x[k:], y[k:], 1)
            ax.plot(x[:k], b1 * x[:k] + c1, "-", color="orange", lw=2,
                    label=f"Low $H$: $\\beta$={b1:.2f}")
            ax.plot(x[k:], b2 * x[k:] + c2, "-", color="teal", lw=2,
                    label=f"High $H$: $\\beta$={b2:.2f}")
            ax.axvline(x[k], color="grey", ls=":", lw=1)

        ax.set_xlabel("$\\ln(1/2 - H)$", fontsize=12)
        ax.set_ylabel("$\\ln\\,\\Gamma(H)$", fontsize=12)
        sig = "YES" if phase["significant"] else "NO"
        ax.set_title(f"Phase Transition Test (significant: {sig}, "
                     f"RSS improvement: {phase['improvement']:.0%})", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / "fig_h_sweep_phase_transition.png", dpi=300)
        plt.close(fig)
        print(f"  Saved {d / 'fig_h_sweep_phase_transition.png'}", flush=True)

    def _fig_summary(self, d: Path, fit: dict) -> None:
        rel = self.compute_relative_gap() * 100
        fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

        # 1: ES_95
        ax = axes[0]
        ax.plot(self.H_values, self.es95_bs, "o-", color=C_BS, lw=2, ms=5, label="BS Delta")
        ax.plot(self.H_values, self.es95_dh, "s-", color=C_DH, lw=2, ms=5, label="Deep Hedger")
        ax.fill_between(self.H_values, self.es95_dh, self.es95_bs, alpha=0.12, color=C_GAM)
        ax.set_ylabel("ES$_{0.95}$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2: Gamma linear
        ax = axes[1]
        ax.plot(self.H_values, self.gamma, "D-", color=C_GAM, lw=2, ms=6)
        if np.isfinite(fit.get("beta", float("nan"))):
            ax.plot(self.H_values, fit["fitted_gamma"], "--", color=C_FIT, lw=1.5)
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_ylabel("$\\Gamma(H)$")
        ax.grid(True, alpha=0.3)

        # 3: Gamma log-log
        ax = axes[2]
        mask = (self.gamma > 0) & (self.H_values < 0.499)
        xp = 0.5 - self.H_values[mask]
        yp = self.gamma[mask]
        ax.scatter(xp, yp, color=C_GAM, s=40, zorder=5, edgecolors="k", lw=0.4)
        if np.isfinite(fit.get("beta", float("nan"))):
            xs = np.linspace(xp.min() * 0.8, xp.max() * 1.2, 80)
            ax.plot(xs, fit["c"] * xs ** fit["beta"], "--", color=C_FIT, lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("$\\Gamma$ (log)")
        ax.grid(True, alpha=0.3, which="both")

        # 4: Relative gap
        ax = axes[3]
        ax.plot(self.H_values, rel, "D-", color=C_GAM, lw=2, ms=6)
        ax.set_ylabel("Relative gap (%)")
        ax.set_xlabel("Hurst parameter $H$")
        ax.grid(True, alpha=0.3)

        fig.suptitle("H-Sweep Summary", fontsize=14, y=0.995)
        fig.tight_layout()
        fig.savefig(d / "fig_h_sweep_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {d / 'fig_h_sweep_summary.png'}", flush=True)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_analysis_report(self, fit: dict, boot: dict, phase: dict) -> None:
        print("\n===== H-Sweep Analysis =====")
        print(f"Power law: Gamma(H) = c * (1/2 - H)^beta")
        beta_ci = boot.get("beta_ci", (float("nan"), float("nan")))
        c_ci = boot.get("c_ci", (float("nan"), float("nan")))
        print(f"  beta     = {fit['beta']:.3f}   [95% CI: {beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
        print(f"  c        = {fit['c']:.3f}   [95% CI: {c_ci[0]:.3f}, {c_ci[1]:.3f}]")
        print(f"  R^2      = {fit['r_squared']:.3f}")
        print(f"  n_points = {fit['n_points']}")
        print(f"\nPhase transition test:")
        print(f"  Break point:     H = {phase['break_point']:.2f}")
        print(f"  RSS improvement: {phase['improvement']:.1%}")
        print(f"  beta_low:        {phase['beta_low']:.3f}")
        print(f"  beta_high:       {phase['beta_high']:.3f}")
        print(f"  Significant:     {'YES' if phase['significant'] else 'NO'}")

        rel = self.compute_relative_gap() * 100
        i07 = np.argmin(np.abs(self.H_values - 0.07))
        print(f"\nSummary:")
        print(f"  Gamma(0.07)             = {self.gamma[i07]:.3f}")
        print(f"  Gamma(0.07)/ES95_BS     = {rel[i07]:.1f}%")
        print(f"  Max relative gap:         {rel.max():.1f}% at H={self.H_values[np.argmax(rel)]:.2f}")
        print(f"  Min relative gap:         {rel.min():.1f}% at H={self.H_values[np.argmin(rel)]:.2f}")
        print(f"  Mean Gamma across all H:  {self.gamma.mean():.3f}")

    def export_latex_table(self, path: str | Path = "figures/h_sweep_table.tex") -> None:
        path = Path(path)
        rel = self.compute_relative_gap() * 100
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{H-sweep: deep hedging advantage across roughness levels}",
            r"\begin{tabular}{c|rrrrc}",
            r"\hline",
            r"$H$ & ES$_{95}^{\text{BS}}$ & ES$_{95}^{\text{DH}}$ "
            r"& $\Gamma(H)$ & $\Gamma/\text{ES}^{\text{BS}}$ (\%) & std ratio \\",
            r"\hline",
        ]
        for i, r in enumerate(self.results):
            std_ratio = self.std_dh[i] / max(self.std_bs[i], 1e-12)
            lines.append(
                f"{self.H_values[i]:.2f} & {self.es95_bs[i]:.2f} & "
                f"{self.es95_dh[i]:.2f} & {self.gamma[i]:.3f} & "
                f"{rel[i]:.1f}\\% & {std_ratio:.3f} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        path.write_text("\n".join(lines))
        print(f"  Saved {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60, flush=True)
    print("  H-Sweep Analysis", flush=True)
    print("=" * 60, flush=True)

    analyser = HSweepAnalyser(FIGURE_DIR / "h_sweep_results.json")

    fit = analyser.fit_power_law()
    boot = analyser.bootstrap_confidence(n_bootstrap=1000)
    phase = analyser.test_phase_transition()

    analyser.print_analysis_report(fit, boot, phase)

    print("\n  Generating figures ...", flush=True)
    analyser.generate_all_figures(FIGURE_DIR)
    analyser.export_latex_table(FIGURE_DIR / "h_sweep_table.tex")

    print("\n" + "=" * 60, flush=True)
    print("  DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
