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
            data = json.load(f)
        # Handle both old (list) and new (dict with "results" key) schema
        if isinstance(data, list):
            return data
        return data.get("results", data)

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
# Standalone bootstrap functions (callable without class)
# ---------------------------------------------------------------------------

def bootstrap_power_law_slope(
    H_values: np.ndarray,
    gamma_values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2024,
) -> dict:
    """Panel bootstrap over regression points for the log-log slope.

    Resamples the (x, y) pairs with replacement and refits the log-log
    regression log Gamma(H) = log c + beta * log(1/2 - H).
    """
    rng = np.random.default_rng(seed)
    mask = (gamma_values > 0) & (H_values < 0.499)
    log_x = np.log(0.5 - H_values[mask])
    log_y = np.log(gamma_values[mask])
    n = len(log_x)

    if n < 3:
        nan = float("nan")
        return {"beta_hat": nan, "beta_se_ols": nan,
                "beta_ci_bootstrap_95": [nan, nan],
                "beta_ci_bootstrap_68": [nan, nan],
                "beta_samples_mean": nan, "beta_samples_std": nan,
                "r_squared_bootstrap_mean": nan,
                "n_bootstrap": n_bootstrap, "seed": seed}

    samples = np.empty(n_bootstrap)
    r2s = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        fit = np.polyfit(log_x[idx], log_y[idx], 1, cov=False)
        samples[b] = fit[0]
        y_fit = np.polyval(fit, log_x[idx])
        ss_res = np.sum((log_y[idx] - y_fit) ** 2)
        ss_tot = np.sum((log_y[idx] - np.mean(log_y[idx])) ** 2)
        r2s[b] = 1.0 - ss_res / max(ss_tot, 1e-30)

    beta_full, cov = np.polyfit(log_x, log_y, 1, cov=True)
    return {
        "beta_hat":             float(beta_full[0]),
        "beta_se_ols":          float(np.sqrt(cov[0, 0])),
        "beta_ci_bootstrap_95": [float(np.quantile(samples, 0.025)),
                                 float(np.quantile(samples, 0.975))],
        "beta_ci_bootstrap_68": [float(np.quantile(samples, 0.16)),
                                 float(np.quantile(samples, 0.84))],
        "beta_samples_mean":    float(samples.mean()),
        "beta_samples_std":     float(samples.std(ddof=1)),
        "r_squared_bootstrap_mean": float(np.nanmean(r2s)),
        "n_bootstrap":          n_bootstrap,
        "seed":                 seed,
    }


def compute_slope_noise_floor(
    H_values: np.ndarray,
    es_halfwidth_per_point: float,
    gamma_values: np.ndarray,
) -> dict:
    """Monte Carlo noise floor for the log-log slope.

    Translates per-point ES estimator noise into equivalent slope units.
    If |beta_hat| < beta_noise_floor, the slope is indistinguishable
    from zero given the current MC budget.
    """
    mask = (gamma_values > 0) & (H_values < 0.499)
    log_x = np.log(0.5 - H_values[mask])
    x_range = float(log_x.max() - log_x.min())
    gamma_median = float(np.median(np.abs(gamma_values[mask])))
    relative = es_halfwidth_per_point / gamma_median if gamma_median > 1e-12 else float("inf")
    beta_noise = 2.0 * relative / x_range if x_range > 1e-12 else float("inf")
    beta_full = float(np.polyfit(log_x, np.log(gamma_values[mask]), 1)[0])
    return {
        "es_halfwidth_per_point":  es_halfwidth_per_point,
        "x_range_log":             x_range,
        "gamma_median":            gamma_median,
        "relative_halfwidth":      relative,
        "beta_noise_floor":        float(beta_noise),
        "beta_hat":                beta_full,
        "beta_inside_noise_band":  abs(beta_full) < beta_noise,
    }


def _estimate_es_halfwidth(gamma_values: np.ndarray, n_test: int = 50_000,
                           alpha: float = 0.95) -> float:
    """Analytical estimate of 95% CI halfwidth for ES difference (Gamma).

    Uses the Normal approximation on the tail sample. For rBergomi PnL
    at the dissertation calibration, sigma_tail ~ 10 (empirical ballpark
    from the unified-baseline PnL distributions).

    Gamma = ES_BS - ES_DH involves two independent ES estimates, so the
    noise on the difference is sqrt(2) * SE(ES).
    """
    tail_n = int(n_test * (1.0 - alpha))
    # Conservative tail volatility: typical rBergomi tail std ~ 10
    sigma_tail = 10.0
    se_single = sigma_tail / math.sqrt(max(tail_n, 1))
    # Gamma involves two independent ES estimates
    return 1.96 * math.sqrt(2) * se_single


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import subprocess
    from datetime import datetime

    parser = argparse.ArgumentParser(description="H-sweep analysis.")
    parser.add_argument("--with-bootstrap", action="store_true",
                        help="Compute panel bootstrap and noise floor; persist to JSON.")
    args = parser.parse_args()

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

    if args.with_bootstrap:
        print("\n  Running panel bootstrap (10k replicates) ...", flush=True)
        panel = bootstrap_power_law_slope(
            analyser.H_values, analyser.gamma, n_bootstrap=10_000, seed=2024)

        print(f"    beta_hat = {panel['beta_hat']:.4f}")
        print(f"    beta_se_ols = {panel['beta_se_ols']:.4f}")
        print(f"    95% CI = [{panel['beta_ci_bootstrap_95'][0]:.4f}, "
              f"{panel['beta_ci_bootstrap_95'][1]:.4f}]")

        # Noise floor (analytical fallback — no per-H PnL tensors cached)
        halfwidth = _estimate_es_halfwidth(analyser.gamma)
        noise = compute_slope_noise_floor(
            analyser.H_values, halfwidth, analyser.gamma)
        print(f"    noise floor = {noise['beta_noise_floor']:.4f}")
        print(f"    |beta| inside noise band: {noise['beta_inside_noise_band']}")

        # Git SHA
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            sha = "unknown"

        # Persist to JSON
        json_path = FIGURE_DIR / "h_sweep_results.json"
        with open(json_path) as f:
            data = json.load(f)

        # If data is a list (old schema), wrap it
        if isinstance(data, list):
            data = {"results": data}

        data["bootstrap"] = {
            "panel_slope": panel,
            "noise_floor": noise,
            "per_h_bootstrap_available": False,
            "meta": {
                "source_script": "deep_hedging/experiments/h_sweep_analysis.py",
                "source_commit": sha,
                "generated_at": datetime.now().isoformat(),
            },
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Bootstrap block written to {json_path}")

        # LaTeX table
        tex_path = FIGURE_DIR / "h_sweep_bootstrap_table.tex"
        lo95, hi95 = panel["beta_ci_bootstrap_95"]
        inside_str = "Yes" if noise["beta_inside_noise_band"] else "No"
        tex_lines = [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            f"$\\hat\\beta$ (point estimate) & {panel['beta_hat']:.4f} \\\\",
            f"OLS standard error & $\\pm${panel['beta_se_ols']:.4f} \\\\",
            f"Bootstrap 95\\% CI for $\\hat\\beta$ & [{lo95:.4f}, {hi95:.4f}] \\\\",
            f"Bootstrap mean $R^2$ & {panel['r_squared_bootstrap_mean']:.3f} \\\\",
            f"MC noise floor $\\beta_{{\\mathrm{{noise}}}}$ & {noise['beta_noise_floor']:.4f} \\\\",
            f"$|\\hat\\beta|$ inside noise band & {inside_str} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        tex_path.write_text("\n".join(tex_lines))
        print(f"  Saved {tex_path}")

        # Bootstrap figure (separate from existing)
        _fig_gamma_loglog_bootstrap(analyser, fit, panel, noise, FIGURE_DIR)

    print("\n" + "=" * 60, flush=True)
    print("  DONE", flush=True)
    print("=" * 60, flush=True)


def _fig_gamma_loglog_bootstrap(
    analyser: HSweepAnalyser, fit: dict, panel: dict,
    noise: dict, save_dir: Path,
) -> None:
    """Log-log plot with bootstrap envelope and noise floor."""
    mask = (analyser.gamma > 0) & (analyser.H_values < 0.499)
    x_plot = 0.5 - analyser.H_values[mask]
    y_plot = analyser.gamma[mask]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x_plot, y_plot, color=C_GAM, s=60, zorder=5, edgecolors="k", lw=0.5)

    if np.isfinite(fit.get("beta", float("nan"))):
        xs = np.linspace(x_plot.min() * 0.8, x_plot.max() * 1.2, 100)
        y_fit = fit["c"] * xs ** fit["beta"]
        ax.plot(xs, y_fit, "--", color=C_FIT, lw=2,
                label=f"Fit: $\\beta$={fit['beta']:.3f}")

        # Bootstrap envelope: use 68% CI on beta
        lo68, hi68 = panel["beta_ci_bootstrap_68"]
        y_lo = fit["c"] * xs ** lo68
        y_hi = fit["c"] * xs ** hi68
        ax.fill_between(xs, y_lo, y_hi, alpha=0.15, color=C_FIT,
                        label="68% bootstrap envelope")
        lo95, hi95 = panel["beta_ci_bootstrap_95"]
        y_lo95 = fit["c"] * xs ** lo95
        y_hi95 = fit["c"] * xs ** hi95
        ax.fill_between(xs, y_lo95, y_hi95, alpha=0.07, color=C_FIT,
                        label="95% bootstrap envelope")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$1/2 - H$", fontsize=12)
    ax.set_ylabel("$\\Gamma(H)$", fontsize=12)
    ax.set_title("H-sweep: Power-law fit with bootstrap CIs", fontsize=13)

    # Annotation box
    lo95, hi95 = panel["beta_ci_bootstrap_95"]
    ax.text(0.02, 0.02,
            f"$\\hat{{\\beta}}$ = {panel['beta_hat']:.3f}\n"
            f"95% CI: [{lo95:.3f}, {hi95:.3f}]\n"
            f"Noise floor: {noise['beta_noise_floor']:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_dir / "fig_h_sweep_gamma_loglog_bootstrap.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {save_dir / 'fig_h_sweep_gamma_loglog_bootstrap.png'}")


if __name__ == "__main__":
    main()
