#!/usr/bin/env python
"""
Part A of Prompt 12: worst-case adversarial perturbation via PGD.

For each radius r in a normalised parameter space, finds the worst-case
perturbation direction eps* that maximises ES_95 for a given strategy
(BS delta or the flat deep hedger). Then cross-evaluates both strategies
at each worst-case direction.

Leverages the differentiable rBergomi simulator (Prompt 1) — this
experiment is literally impossible with NumPy-based rBergomi codes
in the literature.

Run:
    python -u -m deep_hedging.experiments.worst_case_adversarial
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
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import expected_shortfall, compute_all_metrics


# ─── Baseline calibration ──────────────────────────────────────
THETA_0: dict[str, float] = dict(H=0.07, eta=1.9, rho=-0.7, xi0=0.235 ** 2)
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100
SIGMA = float(math.sqrt(THETA_0["xi0"]))
MASTER_SEED = 2024

# Per-axis scale factors (one "sigma" ranges for typical calibrations)
SIGMA_SCALES: dict[str, float] = dict(H=0.1, eta=1.0, rho=0.3)

# Parameter box constraints (H, eta, rho)
PARAM_BOX: dict[str, tuple[float, float]] = dict(
    H=(0.01, 0.49),
    eta=(0.2, 4.0),
    rho=(-0.99, 0.0),
)

RADII: list[float] = [0.2, 0.5, 1.0, 2.0]
PGD_STEPS = 50
PGD_LR = 0.02
N_PATHS_ATTACK = 10_000
N_PATHS_EVAL = 30_000

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
HEDGER_PATH = FIGURE_DIR / "adversarial_baseline_hedger.pt"
RESULTS_PATH = FIGURE_DIR / "worst_case_adversarial.json"

# Colours
C_BS = "#2196F3"
C_DEEP = "#4CAF50"


# =======================================================================
# Helpers: simulator construction + evaluation
# =======================================================================

def _build_differentiable_sim(
    H: float, eta: float, rho: float, xi0: float,
    n_steps: int = N_STEPS, T_: float = T,
) -> DifferentiableRoughBergomi:
    """Build sim at (H, eta, rho) with parameters promoted to nn.Parameter."""
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T_, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    sim.volterra.make_H_parameter()
    sim.make_params_differentiable()
    return sim


def _evaluate_strategy_with_grad(
    strategy: str,
    H: float, eta: float, rho: float,
    n_paths: int, seed: int,
    baseline_hedger: DeepHedgerFNN | None,
    p0: float,
    alpha: float = 0.95,
) -> tuple[float, dict[str, float]]:
    """Forward + backward pass returning ES_95 and gradients on (H, eta, rho).

    Returns (es_scalar, grads_dict).
    """
    sim = _build_differentiable_sim(H, eta, rho, THETA_0["xi0"])

    g = torch.Generator().manual_seed(seed)
    Z_vol = torch.randn(n_paths, N_STEPS, 2, dtype=torch.float64, generator=g)
    Z_price = torch.randn(n_paths, N_STEPS, dtype=torch.float64, generator=g)

    S, _ = sim(Z_vol, Z_price, S0=S0)

    if strategy == "bs":
        bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
        deltas = bs.hedge_paths(S)
    elif strategy == "deep":
        if baseline_hedger is None:
            raise ValueError("deep strategy requires a baseline hedger")
        # Freeze hedger so only sim params carry grad
        for p in baseline_hedger.parameters():
            p.requires_grad_(False)
        baseline_hedger.eval()
        deltas = baseline_hedger.hedge_paths(S, T=T, S0=S0)
        deltas = deltas.to(S.dtype)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    payoff = compute_payoff(S, K, "call")
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, 0.0)
    es = expected_shortfall(pnl, alpha)
    es.backward()

    grads = {
        "H": float(sim.volterra._H.grad),
        "eta": float(sim._eta.grad),
        "rho": float(sim._rho.grad),
    }
    return float(es.detach()), grads


def _evaluate_strategy_no_grad(
    strategy: str,
    H: float, eta: float, rho: float,
    n_paths: int, seed: int,
    baseline_hedger: DeepHedgerFNN | None,
    p0: float,
) -> dict[str, Any]:
    """Unbiased evaluation on fresh paths (no gradient)."""
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=eta, rho=rho, xi0=THETA_0["xi0"],
    )
    with torch.no_grad():
        S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)

        if strategy == "bs":
            bs = BlackScholesDelta(sigma=SIGMA, K=K, T=T)
            deltas = bs.hedge_paths(S)
        elif strategy == "deep":
            baseline_hedger.eval()
            deltas = baseline_hedger.hedge_paths(S, T=T, S0=S0)
            deltas = deltas.to(S.dtype)
        else:
            raise ValueError(strategy)

        payoff = compute_payoff(S, K, "call")
        pnl = compute_hedging_pnl(S, deltas, payoff, p0, 0.0)
        metrics = compute_all_metrics(pnl)
    return {"metrics": metrics, "es_95": float(metrics["es_95"])}


# =======================================================================
# Projection onto the constraint set
# =======================================================================

def project_epsilon(
    eps_H: float, eps_eta: float, eps_rho: float,
    radius: float,
    theta_0: dict[str, float] = THETA_0,
    sigma_scales: dict[str, float] = SIGMA_SCALES,
    box: dict[str, tuple[float, float]] = PARAM_BOX,
) -> tuple[float, float, float]:
    """Project (eps_H, eps_eta, eps_rho) onto the normalised-ball ∩ box.

    Normalised norm:
        ||eps/sigma|| = sqrt((eps_H/sH)^2 + (eps_eta/sEta)^2 + (eps_rho/sR)^2)

    After projection, (theta_0 + eps) is guaranteed to lie inside the
    parameter box and the scaled norm is <= radius.
    """
    # 1) Shrink onto ball if outside
    sH = sigma_scales["H"]
    sE = sigma_scales["eta"]
    sR = sigma_scales["rho"]
    norm = math.sqrt((eps_H / sH) ** 2 + (eps_eta / sE) ** 2 + (eps_rho / sR) ** 2)
    if norm > radius and norm > 1e-12:
        scale = radius / norm
        eps_H *= scale
        eps_eta *= scale
        eps_rho *= scale

    # 2) Clamp each component individually onto the box
    lo, hi = box["H"]
    eps_H = max(lo - theta_0["H"], min(hi - theta_0["H"], eps_H))
    lo, hi = box["eta"]
    eps_eta = max(lo - theta_0["eta"], min(hi - theta_0["eta"], eps_eta))
    lo, hi = box["rho"]
    eps_rho = max(lo - theta_0["rho"], min(hi - theta_0["rho"], eps_rho))

    return eps_H, eps_eta, eps_rho


# =======================================================================
# WorstCaseAdversarialExperiment
# =======================================================================

class WorstCaseAdversarialExperiment:
    """PGD-style worst-case adversarial experiment on (H, eta, rho)."""

    def __init__(
        self,
        theta_0: dict[str, float] | None = None,
        radii: list[float] = RADII,
        sigma_scales: dict[str, float] | None = None,
        pgd_steps: int = PGD_STEPS,
        pgd_lr: float = PGD_LR,
        n_paths_attack: int = N_PATHS_ATTACK,
        n_paths_eval: int = N_PATHS_EVAL,
        alpha: float = 0.95,
        figures_dir: str | Path = FIGURE_DIR,
        device: torch.device | None = None,
    ) -> None:
        self.theta_0 = theta_0 if theta_0 is not None else dict(THETA_0)
        self.radii = list(radii)
        self.sigma_scales = sigma_scales if sigma_scales is not None else dict(SIGMA_SCALES)
        self.pgd_steps = int(pgd_steps)
        self.pgd_lr = float(pgd_lr)
        self.n_paths_attack = int(n_paths_attack)
        self.n_paths_eval = int(n_paths_eval)
        self.alpha = float(alpha)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cpu")
        self.baseline_hedger: DeepHedgerFNN | None = None
        self.p0_calibration: float | None = None

    # ──────────────────────────────────────────────────────────
    # Baseline hedger + calibration p0
    # ──────────────────────────────────────────────────────────

    def load_baseline_hedger(
        self, path: str | Path = HEDGER_PATH,
    ) -> DeepHedgerFNN:
        hedger = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        hedger.load_state_dict(state)
        for p in hedger.parameters():
            p.requires_grad_(False)
        hedger.eval()
        self.baseline_hedger = hedger.to(self.device)
        return self.baseline_hedger

    def compute_calibration_p0(self, n_paths: int = 100_000, seed: int = 999) -> float:
        sim = DifferentiableRoughBergomi(
            n_steps=N_STEPS, T=T,
            H=self.theta_0["H"], eta=self.theta_0["eta"], rho=self.theta_0["rho"],
            xi0=self.theta_0["xi0"],
        )
        with torch.no_grad():
            S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)
            payoff = compute_payoff(S, K, "call")
            p0 = float(payoff.mean())
        self.p0_calibration = p0
        return p0

    # ──────────────────────────────────────────────────────────
    # PGD attack
    # ──────────────────────────────────────────────────────────

    def attack_strategy(
        self,
        strategy: str,
        radius: float,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run PGD to find the worst-case eps for the given strategy."""
        if strategy == "deep" and self.baseline_hedger is None:
            raise RuntimeError("call load_baseline_hedger() first")
        if self.p0_calibration is None:
            raise RuntimeError("call compute_calibration_p0() first")
        p0 = self.p0_calibration

        # Evaluate baseline (eps = 0)
        _, base_grads = _evaluate_strategy_with_grad(
            strategy,
            H=self.theta_0["H"], eta=self.theta_0["eta"], rho=self.theta_0["rho"],
            n_paths=self.n_paths_attack,
            seed=MASTER_SEED + 7777,
            baseline_hedger=self.baseline_hedger,
            p0=p0,
            alpha=self.alpha,
        )
        es_baseline_attack = _evaluate_strategy_no_grad(
            strategy,
            H=self.theta_0["H"], eta=self.theta_0["eta"], rho=self.theta_0["rho"],
            n_paths=self.n_paths_attack,
            seed=MASTER_SEED + 8888,
            baseline_hedger=self.baseline_hedger,
            p0=p0,
        )["es_95"]

        # Initialise eps using the baseline gradient direction
        # (points "uphill" on ES_95, so it's a better start than zero)
        gH = base_grads["H"]
        gE = base_grads["eta"]
        gR = base_grads["rho"]
        sH = self.sigma_scales["H"]
        sE = self.sigma_scales["eta"]
        sR = self.sigma_scales["rho"]
        scaled_grad_norm = math.sqrt(
            (gH * sH) ** 2 + (gE * sE) ** 2 + (gR * sR) ** 2
        )
        if scaled_grad_norm > 1e-12:
            # Initial step magnitude ~ 0.3 * radius in scaled norm
            init_scale = 0.3 * radius / scaled_grad_norm
            eps_H = init_scale * gH * sH * sH
            eps_eta = init_scale * gE * sE * sE
            eps_rho = init_scale * gR * sR * sR
        else:
            eps_H = eps_eta = eps_rho = 0.0
        eps_H, eps_eta, eps_rho = project_epsilon(
            eps_H, eps_eta, eps_rho, radius,
            self.theta_0, self.sigma_scales, PARAM_BOX,
        )

        # Track best
        best_eps = (eps_H, eps_eta, eps_rho)
        best_es = es_baseline_attack
        iter_history: list[dict[str, float]] = []

        for step in range(self.pgd_steps):
            H_cur = self.theta_0["H"] + eps_H
            eta_cur = self.theta_0["eta"] + eps_eta
            rho_cur = self.theta_0["rho"] + eps_rho

            seed = MASTER_SEED + 1_000_000 + step * 137
            es_val, grads = _evaluate_strategy_with_grad(
                strategy,
                H=H_cur, eta=eta_cur, rho=rho_cur,
                n_paths=self.n_paths_attack,
                seed=seed,
                baseline_hedger=self.baseline_hedger,
                p0=p0,
            )
            iter_history.append({
                "step": step, "eps_H": eps_H, "eps_eta": eps_eta,
                "eps_rho": eps_rho, "es_95": es_val,
            })

            if es_val > best_es:
                best_es = es_val
                best_eps = (eps_H, eps_eta, eps_rho)

            # Gradient ASCENT (we want to maximise ES)
            # Scale the LR per-axis by sigma^2 so all axes move proportionally
            sH2 = self.sigma_scales["H"] ** 2
            sE2 = self.sigma_scales["eta"] ** 2
            sR2 = self.sigma_scales["rho"] ** 2

            eps_H = eps_H + self.pgd_lr * grads["H"] * sH2
            eps_eta = eps_eta + self.pgd_lr * grads["eta"] * sE2
            eps_rho = eps_rho + self.pgd_lr * grads["rho"] * sR2

            eps_H, eps_eta, eps_rho = project_epsilon(
                eps_H, eps_eta, eps_rho, radius,
                self.theta_0, self.sigma_scales, PARAM_BOX,
            )

            if verbose and (step % 10 == 0 or step == self.pgd_steps - 1):
                print(f"      pgd step {step+1:>3d}: es={es_val:.4f}  "
                      f"eps=({eps_H:+.4f}, {eps_eta:+.4f}, {eps_rho:+.4f})",
                      flush=True)

        # Unbiased eval on a larger set at the best eps
        eH, eE, eR = best_eps
        eval_data = _evaluate_strategy_no_grad(
            strategy,
            H=self.theta_0["H"] + eH,
            eta=self.theta_0["eta"] + eE,
            rho=self.theta_0["rho"] + eR,
            n_paths=self.n_paths_eval,
            seed=MASTER_SEED + 5_000_000,
            baseline_hedger=self.baseline_hedger,
            p0=p0,
        )
        # Re-evaluate the calibration baseline too
        baseline_eval = _evaluate_strategy_no_grad(
            strategy,
            H=self.theta_0["H"], eta=self.theta_0["eta"], rho=self.theta_0["rho"],
            n_paths=self.n_paths_eval,
            seed=MASTER_SEED + 5_000_001,
            baseline_hedger=self.baseline_hedger,
            p0=p0,
        )
        es_worst = eval_data["es_95"]
        es_base = baseline_eval["es_95"]
        deg = es_worst - es_base
        rel_deg = deg / max(es_base, 1e-12)

        return {
            "strategy": strategy,
            "radius": float(radius),
            "eps_best": {"H": float(eH), "eta": float(eE), "rho": float(eR)},
            "theta_perturbed": {
                "H": float(self.theta_0["H"] + eH),
                "eta": float(self.theta_0["eta"] + eE),
                "rho": float(self.theta_0["rho"] + eR),
            },
            "es95_baseline": float(es_base),
            "es95_worst_case": float(es_worst),
            "degradation": float(deg),
            "relative_degradation": float(rel_deg),
            "es95_best_attack_set": float(best_es),  # on n_paths_attack
            "iter_es_history": [float(h["es_95"]) for h in iter_history],
        }

    # ──────────────────────────────────────────────────────────
    # Cross-evaluation at each radius
    # ──────────────────────────────────────────────────────────

    def cross_evaluate(
        self,
        eps_bs: dict[str, float],
        eps_dh: dict[str, float],
        radius: float,
    ) -> dict[str, float]:
        """Evaluate BOTH strategies at BOTH worst-case directions."""
        assert self.baseline_hedger is not None
        assert self.p0_calibration is not None
        p0 = self.p0_calibration

        def _eval(strategy: str, eps: dict[str, float], seed: int) -> float:
            return _evaluate_strategy_no_grad(
                strategy,
                H=self.theta_0["H"] + eps["H"],
                eta=self.theta_0["eta"] + eps["eta"],
                rho=self.theta_0["rho"] + eps["rho"],
                n_paths=self.n_paths_eval,
                seed=seed,
                baseline_hedger=self.baseline_hedger,
                p0=p0,
            )["es_95"]

        seed_base = MASTER_SEED + 6_000_000 + int(radius * 10000)
        return {
            "BS_at_eps_BS": _eval("bs", eps_bs, seed_base + 0),
            "BS_at_eps_DH": _eval("bs", eps_dh, seed_base + 1),
            "DH_at_eps_BS": _eval("deep", eps_bs, seed_base + 2),
            "DH_at_eps_DH": _eval("deep", eps_dh, seed_base + 3),
        }

    # ──────────────────────────────────────────────────────────
    # Orchestration
    # ──────────────────────────────────────────────────────────

    def run_all_radii(self) -> dict[float, dict[str, Any]]:
        results: dict[float, dict[str, Any]] = {}
        for i, radius in enumerate(self.radii):
            print(f"\n--- Radius r={radius} ({i+1}/{len(self.radii)}) ---", flush=True)

            t0 = time.time()
            print(f"  Attack BS ...", flush=True)
            bs_attack = self.attack_strategy("bs", radius, verbose=False)
            print(f"    BS baseline={bs_attack['es95_baseline']:.3f}  "
                  f"worst={bs_attack['es95_worst_case']:.3f}  "
                  f"(+{bs_attack['relative_degradation']*100:.1f}%)  "
                  f"in {time.time()-t0:.0f}s", flush=True)

            t0 = time.time()
            print(f"  Attack deep hedger ...", flush=True)
            dh_attack = self.attack_strategy("deep", radius, verbose=False)
            print(f"    DH baseline={dh_attack['es95_baseline']:.3f}  "
                  f"worst={dh_attack['es95_worst_case']:.3f}  "
                  f"(+{dh_attack['relative_degradation']*100:.1f}%)  "
                  f"in {time.time()-t0:.0f}s", flush=True)

            print(f"  Cross-evaluating ...", flush=True)
            cross = self.cross_evaluate(
                bs_attack["eps_best"], dh_attack["eps_best"], radius,
            )

            results[radius] = {
                "eps_bs": bs_attack,
                "eps_dh": dh_attack,
                "cross": cross,
            }

            # Save incrementally
            self.save_results(results)

        return results

    # ──────────────────────────────────────────────────────────
    # Verdict + reporting
    # ──────────────────────────────────────────────────────────

    def compute_verdict(self, results: dict[float, dict]) -> dict[str, Any]:
        # DH own-worst ≤ BS own-worst at every radius?
        dh_le_bs = []
        for r, data in results.items():
            dh_own = data["eps_dh"]["es95_worst_case"]
            bs_own = data["eps_bs"]["es95_worst_case"]
            dh_le_bs.append(dh_own <= bs_own)

        all_dh_below = all(dh_le_bs)
        some_dh_below = any(dh_le_bs)

        # Relative degradation comparison
        rel_deg_comparison = {}
        for r, data in results.items():
            rel_deg_comparison[r] = {
                "bs": data["eps_bs"]["relative_degradation"],
                "dh": data["eps_dh"]["relative_degradation"],
                "dh_smaller": (data["eps_dh"]["relative_degradation"]
                               < data["eps_bs"]["relative_degradation"]),
            }
        n_rel_wins = sum(1 for v in rel_deg_comparison.values() if v["dh_smaller"])

        if all_dh_below and n_rel_wins >= len(self.radii) - 1:
            verdict = "strong H3 (worst case)"
            summary = (
                "Strong worst-case robustness: deep hedger's own-worst ES "
                "is below BS's own-worst at every radius AND its relative "
                "degradation is smaller at (almost) every radius."
            )
        elif all_dh_below:
            verdict = "moderate H3 (worst case)"
            summary = (
                "Moderate worst-case robustness: deep hedger's own-worst "
                "ES is below BS at every radius (absolute advantage "
                "preserved), but its relative degradation is not always "
                "smaller."
            )
        elif some_dh_below:
            verdict = "mixed H3 (worst case)"
            summary = (
                "Mixed worst-case robustness: deep hedger is below BS at "
                "some radii but not others. Graceful degradation fails "
                "at larger perturbations."
            )
        else:
            verdict = "H3 refuted (worst case)"
            summary = (
                "Worst-case H3 refuted: the deep hedger's own worst case "
                "exceeds BS's own worst case at at least one radius."
            )

        return {
            "verdict": verdict,
            "summary": summary,
            "dh_own_le_bs_own": dh_le_bs,
            "relative_comparison": rel_deg_comparison,
        }

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save_results(self, results: dict[float, dict]) -> None:
        payload: dict[str, Any] = {
            "config": {
                "theta_0": self.theta_0,
                "sigma_scales": self.sigma_scales,
                "radii": self.radii,
                "pgd_steps": self.pgd_steps,
                "pgd_lr": self.pgd_lr,
                "n_paths_attack": self.n_paths_attack,
                "n_paths_eval": self.n_paths_eval,
                "alpha": self.alpha,
            },
            "p0_calibration": self.p0_calibration,
            "by_radius": {str(r): v for r, v in results.items()},
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # Figures
    # ──────────────────────────────────────────────────────────

    def generate_figures(self, results: dict[float, dict]) -> None:
        print("\n  Generating figures ...", flush=True)
        self._fig_worst_case_radii(results)
        self._fig_worst_case_directions(results)
        self._fig_worst_case_cross_matrix(results)

    def _fig_worst_case_radii(self, results: dict[float, dict]) -> None:
        radii = sorted(results.keys())
        bs_own = [results[r]["eps_bs"]["es95_worst_case"] for r in radii]
        dh_own = [results[r]["eps_dh"]["es95_worst_case"] for r in radii]
        dh_at_eps_bs = [results[r]["cross"]["DH_at_eps_BS"] for r in radii]

        bs_base = results[radii[0]]["eps_bs"]["es95_baseline"]
        dh_base = results[radii[0]]["eps_dh"]["es95_baseline"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(bs_base, color=C_BS, ls=":", lw=1, alpha=0.7,
                   label=f"BS baseline = {bs_base:.2f}")
        ax.axhline(dh_base, color=C_DEEP, ls=":", lw=1, alpha=0.7,
                   label=f"DH baseline = {dh_base:.2f}")
        ax.plot(radii, bs_own, "o-", color=C_BS, lw=2, ms=10,
                label=r"BS at $\varepsilon^*_{\rm BS}$",
                markeredgecolor="black", markeredgewidth=0.5)
        ax.plot(radii, dh_own, "D-", color=C_DEEP, lw=2, ms=10,
                label=r"DH at $\varepsilon^*_{\rm DH}$",
                markeredgecolor="black", markeredgewidth=0.5)
        ax.plot(radii, dh_at_eps_bs, "D--", color=C_DEEP, lw=1.5, ms=8,
                alpha=0.7,
                label=r"DH at $\varepsilon^*_{\rm BS}$ (cross)")
        ax.set_xlabel(r"Perturbation radius $r$ (scaled)", fontsize=11)
        ax.set_ylabel("Worst-case ES$_{95}$", fontsize=11)
        ax.set_title("Worst-Case ES$_{95}$ vs Perturbation Radius", fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.figures_dir / "fig_worst_case_radii.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_worst_case_directions(self, results: dict[float, dict]) -> None:
        radii = sorted(results.keys())
        fig, axes = plt.subplots(1, len(radii), figsize=(4 * len(radii), 4.5), sharey=True)
        if len(radii) == 1:
            axes = [axes]

        for ax, r in zip(axes, radii):
            bs_eps = results[r]["eps_bs"]["eps_best"]
            dh_eps = results[r]["eps_dh"]["eps_best"]
            axes_names = ("H", "eta", "rho")
            bs_vals = [bs_eps[k] / self.sigma_scales[k] for k in axes_names]
            dh_vals = [dh_eps[k] / self.sigma_scales[k] for k in axes_names]
            x = np.arange(len(axes_names))
            w = 0.38
            ax.bar(x - w / 2, bs_vals, w, color=C_BS, label="BS",
                   edgecolor="black", lw=0.5)
            ax.bar(x + w / 2, dh_vals, w, color=C_DEEP, label="DH",
                   edgecolor="black", lw=0.5)
            ax.axhline(0, color="grey", lw=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([r"$H$", r"$\eta$", r"$\rho$"], fontsize=11)
            ax.set_title(f"$r={r}$", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8, loc="best")

        axes[0].set_ylabel(r"$\varepsilon / \sigma_{\rm scale}$", fontsize=11)
        fig.suptitle("Worst-Case $\\varepsilon$ Direction per Strategy",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        path = self.figures_dir / "fig_worst_case_directions.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    def _fig_worst_case_cross_matrix(self, results: dict[float, dict]) -> None:
        radii = sorted(results.keys())
        # Pick 2 representative radii: smallest and largest
        picks = [radii[0], radii[-1]] if len(radii) >= 2 else radii

        fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.5))
        if len(picks) == 1:
            axes = [axes]

        for ax, r in zip(axes, picks):
            cross = results[r]["cross"]
            M = np.array([
                [cross["BS_at_eps_BS"], cross["BS_at_eps_DH"]],
                [cross["DH_at_eps_BS"], cross["DH_at_eps_DH"]],
            ])
            im = ax.imshow(M, cmap="viridis_r", aspect="auto")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="white" if M[i, j] > M.mean() else "black",
                            fontsize=13, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([r"$\varepsilon^*_{\rm BS}$", r"$\varepsilon^*_{\rm DH}$"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["BS", "DH"])
            ax.set_title(f"$r = {r}$", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Cross-Evaluation: ES$_{95}$ at Each Worst-Case Direction",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        path = self.figures_dir / "fig_worst_case_cross_matrix.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {path.name}", flush=True)

    # ──────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────

    def print_report(self, results: dict[float, dict], verdict: dict) -> None:
        print("\n" + "=" * 70, flush=True)
        print("  WORST-CASE ADVERSARIAL REPORT", flush=True)
        print("=" * 70, flush=True)

        for r in sorted(results.keys()):
            data = results[r]
            cross = data["cross"]
            bs = data["eps_bs"]
            dh = data["eps_dh"]
            print(f"\n--- Radius r = {r} ---", flush=True)
            print(f"  BS worst case:  ES_95 = {bs['es95_worst_case']:.3f}  "
                  f"(+{bs['relative_degradation']*100:.1f}% vs baseline)", flush=True)
            print(f"  DH worst case:  ES_95 = {dh['es95_worst_case']:.3f}  "
                  f"(+{dh['relative_degradation']*100:.1f}% vs baseline)", flush=True)
            print(f"  eps*_BS = (H={bs['eps_best']['H']:+.4f}, "
                  f"eta={bs['eps_best']['eta']:+.4f}, "
                  f"rho={bs['eps_best']['rho']:+.4f})", flush=True)
            print(f"  eps*_DH = (H={dh['eps_best']['H']:+.4f}, "
                  f"eta={dh['eps_best']['eta']:+.4f}, "
                  f"rho={dh['eps_best']['rho']:+.4f})", flush=True)
            print(f"  Cross-eval:", flush=True)
            print(f"    BS at eps*_BS = {cross['BS_at_eps_BS']:.3f}", flush=True)
            print(f"    BS at eps*_DH = {cross['BS_at_eps_DH']:.3f}", flush=True)
            print(f"    DH at eps*_BS = {cross['DH_at_eps_BS']:.3f}", flush=True)
            print(f"    DH at eps*_DH = {cross['DH_at_eps_DH']:.3f}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("  VERDICT", flush=True)
        print("=" * 70, flush=True)
        print(f"  {verdict['verdict'].upper()}", flush=True)
        print(f"  {verdict['summary']}", flush=True)


# =======================================================================
# CLI
# =======================================================================

def main() -> None:
    exp = WorstCaseAdversarialExperiment()

    print("=" * 70, flush=True)
    print("  WORST-CASE ADVERSARIAL PERTURBATION — PART A OF PROMPT 12", flush=True)
    print("=" * 70, flush=True)
    print(f"  Theta_0: {exp.theta_0}", flush=True)
    print(f"  Radii: {exp.radii}", flush=True)
    print(f"  PGD steps: {exp.pgd_steps}  lr: {exp.pgd_lr}", flush=True)
    print(f"  N paths: attack={exp.n_paths_attack}, eval={exp.n_paths_eval}",
          flush=True)

    print("\n  Loading baseline hedger from Prompt 11 cache ...", flush=True)
    exp.load_baseline_hedger()

    print("  Computing calibration p_0 ...", flush=True)
    p0 = exp.compute_calibration_p0()
    print(f"    p_0 = {p0:.4f}", flush=True)

    results = exp.run_all_radii()
    verdict = exp.compute_verdict(results)

    exp.save_results(results)
    exp.generate_figures(results)
    exp.print_report(results, verdict)


if __name__ == "__main__":
    main()
