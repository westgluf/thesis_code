#!/usr/bin/env python
"""
Phase M — Perturbation Robustness Comprehensive Extension (Block 5).

Six sub-experiments characterising the robustness basin of the deep
hedger around the canonical rough-Bergomi calibration:

  M.1  Extended radius range          (7 radii × 6 axis-dir × 5 seeds = 210 evals)
  M.2  Higher-resolution axis sweeps  (3 axes × 15 grid × 5 seeds   = 225 evals)
  M.3  Joint 3D PGD                   (5 radii × 5 seeds            =  25 PGDs)
  M.4  Targeted attacks on DH         (3 radii × 3 seeds × 2 modes  =  18 PGDs)
  M.5  Objective-dependent robustness (4 obj × 5 seeds train + eval = 20+450 cells)
  M.6  Hessian eigenstructure         (deterministic FD)

Run from repo root:
    python -u -m deep_hedging.experiments.perturbation_extended --setup
    python -u -m deep_hedging.experiments.perturbation_extended --M1
    python -u -m deep_hedging.experiments.perturbation_extended --M2
    python -u -m deep_hedging.experiments.perturbation_extended --M3
    python -u -m deep_hedging.experiments.perturbation_extended --M4
    python -u -m deep_hedging.experiments.perturbation_extended --M5
    python -u -m deep_hedging.experiments.perturbation_extended --M6

    python -u -m deep_hedging.experiments.perturbation_extended --repro-M1
    ... (same for M2, M3, M4, M5)

Reproducibility: each --repro-MX runs a single seed/cell in a fresh
subprocess and writes to a separate L*_repro.json (never overwrites
the main results file).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.hedging.deep_hedger import DeepHedgerFNN, train_deep_hedger
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import (
    compute_all_metrics,
    expected_shortfall,
    entropic_risk,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "perturbation_v2"
FIG_DIR = REPO_ROOT / "figures" / "perturbation_v2"
M5_CHECKPOINTS_DIR = FIG_DIR  # M.5 saves trained checkpoints in figures/perturbation_v2/

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_DH_PATH = REPO_ROOT / "figures" / "unified_dh_rbergomi_hedger.pt"

# ─── Baseline calibration (from Phase B canonical) ────────────────────
H_BL = 0.07
ETA_BL = 1.9
RHO_BL = -0.7
XI0_BL = 0.235 ** 2  # 0.055225
SIGMA_BS = math.sqrt(XI0_BL)  # 0.235
S0 = 100.0
K = 100.0
T = 1.0
N_STEPS = 100

# Per-axis normalisation (from Phase M spec §3.3)
SIGMA_AXES: dict[str, float] = dict(H=0.05, eta=0.5, rho=0.2)

# Parameter bounds (from Phase M spec §3.4)
PARAM_BOX: dict[str, tuple[float, float]] = dict(
    H=(0.02, 0.49),
    eta=(0.01, 4.0),
    rho=(-0.99, 0.99),
)

# Test-set sizes
N_PATHS_EVAL = 50_000
N_PATHS_PGD = 10_000

# Reference benchmarks (5-seed canonical, Phase B)
CANONICAL_BS_5SEED = 11.5921
CANONICAL_DH_5SEED = 10.4442

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"

def _meta() -> dict[str, Any]:
    return {
        "script": "deep_hedging/experiments/perturbation_extended.py",
        "git_commit": _git_commit_sha(),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "baseline": {
            "H": H_BL, "eta": ETA_BL, "rho": RHO_BL, "xi0": XI0_BL,
            "S0": S0, "K": K, "T": T, "n_steps": N_STEPS,
        },
        "sigma_axes": dict(SIGMA_AXES),
        "param_box": {k: list(v) for k, v in PARAM_BOX.items()},
    }

def _strip_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _strip_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)

def _save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def _agg(values: list[float]) -> dict[str, Any]:
    """Aggregate per-seed values to mean / std / SE / 95% CI / min / max / n."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mn = float(arr.mean())
    if n > 1:
        sd = float(arr.std(ddof=1))
        se = sd / math.sqrt(n)
    else:
        sd = 0.0
        se = 0.0
    return {
        "mean": mn,
        "std": sd,
        "se": se,
        "ci95_lower": mn - 1.96 * se,
        "ci95_upper": mn + 1.96 * se,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "all_values": arr.tolist(),
        "n": n,
    }

def _load_canonical_dh() -> DeepHedgerFNN:
    """Load the canonical rough-Bergomi-trained deep hedger."""
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
    state = torch.load(CANONICAL_DH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def _clamp_to_box(H: float, eta: float, rho: float) -> tuple[float, float, float]:
    """Clamp parameters to the physically-valid box."""
    H = max(PARAM_BOX["H"][0], min(PARAM_BOX["H"][1], H))
    eta = max(PARAM_BOX["eta"][0], min(PARAM_BOX["eta"][1], eta))
    rho = max(PARAM_BOX["rho"][0], min(PARAM_BOX["rho"][1], rho))
    return H, eta, rho

def _normalised_radius(dH: float, deta: float, drho: float) -> float:
    """Compute radius in the normalised-3D coordinate system."""
    return math.sqrt(
        (dH / SIGMA_AXES["H"]) ** 2
        + (deta / SIGMA_AXES["eta"]) ** 2
        + (drho / SIGMA_AXES["rho"]) ** 2
    )

def _project_to_ball_and_box(
    dH: float, deta: float, drho: float, radius: float,
) -> tuple[float, float, float]:
    """Project (dH, deta, drho) into the normalised-r ball ∩ parameter box."""
    norm = _normalised_radius(dH, deta, drho)
    if norm > radius and norm > 1e-12:
        scale = radius / norm
        dH *= scale
        deta *= scale
        drho *= scale
    # Box clamp on absolute parameter values
    dH = max(PARAM_BOX["H"][0] - H_BL, min(PARAM_BOX["H"][1] - H_BL, dH))
    deta = max(PARAM_BOX["eta"][0] - ETA_BL, min(PARAM_BOX["eta"][1] - ETA_BL, deta))
    drho = max(PARAM_BOX["rho"][0] - RHO_BL, min(PARAM_BOX["rho"][1] - RHO_BL, drho))
    return dH, deta, drho

# --------------------------------------------------------------------------
# Path generation + evaluation primitives
# --------------------------------------------------------------------------

def _simulate_perturbed(
    H: float, eta: float, rho: float, xi0: float = XI0_BL,
    n_paths: int = N_PATHS_EVAL, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate rough-Bergomi at perturbed (H, eta, rho); return (S, V)."""
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    with torch.no_grad():
        S, V, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)
    return S, V

def _evaluate_strategies_at(
    H: float, eta: float, rho: float, seed: int,
    canonical_dh: DeepHedgerFNN,
    p0_override: float | None = None,
    n_paths: int = N_PATHS_EVAL,
) -> dict[str, Any]:
    """Evaluate canonical DH and BS delta on paths at perturbed (H,eta,rho).

    p0 is computed from a separate independent batch at the SAME perturbed
    parameters (so it correctly reflects the price under the perturbed model).
    Use p0_override to fix p0 across cells (e.g. for axis-aligned sweeps where
    we want to isolate the hedge effect from the calibration drift).

    Returns: { "p0": float, "dh": metrics_dict, "bs": metrics_dict, "gap_dh_minus_bs": float }
    """
    # Evaluation paths
    S, _ = _simulate_perturbed(H, eta, rho, n_paths=n_paths, seed=seed)
    payoff = compute_payoff(S, K, "call")

    # p0: MC under the perturbed model on a separate batch
    if p0_override is None:
        S_p0, _ = _simulate_perturbed(
            H, eta, rho, n_paths=n_paths, seed=seed + 100_000,
        )
        p0 = float(compute_payoff(S_p0, K, "call").mean())
        del S_p0
        gc.collect()
    else:
        p0 = float(p0_override)

    # DH
    canonical_dh.eval()
    with torch.no_grad():
        deltas_dh = canonical_dh.hedge_paths(S.float(), T, S0).to(S.dtype)
    pnl_dh = compute_hedging_pnl(S, deltas_dh, payoff, p0, 0.0)
    m_dh = compute_all_metrics(pnl_dh)

    # BS
    bs = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
    deltas_bs = bs.hedge_paths(S)
    pnl_bs = compute_hedging_pnl(S, deltas_bs, payoff, p0, 0.0)
    m_bs = compute_all_metrics(pnl_bs)

    return {
        "H": H, "eta": eta, "rho": rho, "p0": p0,
        "dh": m_dh, "bs": m_bs,
        "gap_dh_minus_bs_es95": float(m_dh["es_95"] - m_bs["es_95"]),
    }

def _evaluate_with_grad(
    H: float, eta: float, rho: float, seed: int,
    canonical_dh: DeepHedgerFNN,
    p0: float,
    objective: str = "dh",  # "dh" | "bs" | "dh_minus_bs"
    n_paths: int = N_PATHS_PGD,
) -> tuple[float, dict[str, float]]:
    """Compute (objective_value, gradient_dict) at perturbed (H, eta, rho).

    objective:
      "dh"          → ES_0.95(DH).  Gradient ∂ES_DH/∂(H,eta,rho).
      "bs"          → ES_0.95(BS).  Gradient ∂ES_BS/∂(H,eta,rho).
      "dh_minus_bs" → ES_DH − ES_BS. PGD on this maximises DH − BS gap.
    """
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=eta, rho=rho, xi0=XI0_BL,
    )
    sim.volterra.make_H_parameter()
    sim.make_params_differentiable()

    g = torch.Generator().manual_seed(seed)
    Z_vol = torch.randn(n_paths, N_STEPS, 2, dtype=torch.float64, generator=g)
    Z_price = torch.randn(n_paths, N_STEPS, dtype=torch.float64, generator=g)

    S, _ = sim(Z_vol, Z_price, S0=S0)
    payoff = compute_payoff(S, K, "call")

    if objective in ("dh", "dh_minus_bs"):
        for p in canonical_dh.parameters():
            p.requires_grad_(False)
        canonical_dh.eval()
        deltas_dh = canonical_dh.hedge_paths(S.float(), T, S0).to(S.dtype)
        pnl_dh = compute_hedging_pnl(S, deltas_dh, payoff, p0, 0.0)
        es_dh = expected_shortfall(pnl_dh, 0.95)

    if objective in ("bs", "dh_minus_bs"):
        bs = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
        deltas_bs = bs.hedge_paths(S)
        pnl_bs = compute_hedging_pnl(S, deltas_bs, payoff, p0, 0.0)
        es_bs = expected_shortfall(pnl_bs, 0.95)

    if objective == "dh":
        loss = es_dh
    elif objective == "bs":
        loss = es_bs
    elif objective == "dh_minus_bs":
        loss = es_dh - es_bs
    else:
        raise ValueError(objective)

    loss.backward()
    grads = {
        "H": float(sim.volterra._H.grad.item()),
        "eta": float(sim._eta.grad.item()),
        "rho": float(sim._rho.grad.item()),
    }
    return float(loss.detach().item()), grads

# --------------------------------------------------------------------------
# M.1 — Extended radius range (axis-aligned, no PGD optimisation)
# --------------------------------------------------------------------------

M1_RADII = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
M1_AXES = ["H", "eta", "rho"]
M1_DIRECTIONS = ["+", "-"]
M1_SEEDS = [8001, 8002, 8003, 8004, 8005]

def _axis_perturb(axis: str, direction: str, radius: float) -> tuple[float, float, float]:
    """Compute axis-aligned perturbation (H, eta, rho) at given radius."""
    sgn = +1.0 if direction == "+" else -1.0
    dH = deta = drho = 0.0
    if axis == "H":
        dH = sgn * radius * SIGMA_AXES["H"]
    elif axis == "eta":
        deta = sgn * radius * SIGMA_AXES["eta"]
    elif axis == "rho":
        drho = sgn * radius * SIGMA_AXES["rho"]
    else:
        raise ValueError(f"unknown axis {axis!r}")
    H, eta, rho = _clamp_to_box(H_BL + dH, ETA_BL + deta, RHO_BL + drho)
    return H, eta, rho

def run_M1_extended_radius(
    radii: list[float] | None = None,
    axes: list[str] | None = None,
    seeds: list[int] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.1: axis-aligned perturbations at extended radii."""
    if radii is None:
        radii = list(M1_RADII)
    if axes is None:
        axes = list(M1_AXES)
    if seeds is None:
        seeds = list(M1_SEEDS)

    print("=" * 70, flush=True)
    print(
        f"  M.1 — Extended radius "
        f"({len(radii)} radii × {len(axes)} axes × 2 dir × {len(seeds)} seeds = "
        f"{len(radii)*len(axes)*2*len(seeds)} evals)",
        flush=True,
    )
    print("=" * 70, flush=True)

    canonical_dh = _load_canonical_dh()
    intermediate_path = out_path or (OUT_DIR / "M1_extended_radius.json")

    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            out = existing.get("results", {})
            n_existing = sum(
                1 for ax_dict in out.values()
                for d_dict in (ax_dict.values() if isinstance(ax_dict, dict) else [])
                for r_dict in (d_dict.values() if isinstance(d_dict, dict) else [])
                for s, m in (r_dict.get("per_seed", {}).items() if isinstance(r_dict, dict) else [])
                if isinstance(m, dict) and "dh" in m
            )
            print(f"  RESUME: loaded {n_existing} existing evaluations", flush=True)
        except Exception as exc:
            print(f"  RESUME failed ({exc}); starting fresh", flush=True)
            out = {}

    for axis in axes:
        if axis not in out:
            out[axis] = {}
        for direction in M1_DIRECTIONS:
            if direction not in out[axis]:
                out[axis][direction] = {}
            for r in radii:
                key = f"{r:g}"
                if key not in out[axis][direction]:
                    out[axis][direction][key] = {"radius": r, "per_seed": {}, "aggregate": {}}
                cell = out[axis][direction][key]
                # Compute the perturbed point ONCE per (axis, dir, r) — same for all seeds
                Hp, etap, rhop = _axis_perturb(axis, direction, r)
                cell["H"] = Hp
                cell["eta"] = etap
                cell["rho"] = rhop
                for seed in seeds:
                    if str(seed) in cell["per_seed"] and "dh" in cell["per_seed"][str(seed)]:
                        continue
                    print(
                        f"    axis={axis} dir={direction} r={r} "
                        f"(H={Hp:.4f}, eta={etap:.4f}, rho={rhop:.4f}) "
                        f"seed={seed} ...",
                        flush=True,
                    )
                    t0 = time.time()
                    try:
                        rec = _evaluate_strategies_at(
                            Hp, etap, rhop, seed=seed,
                            canonical_dh=canonical_dh,
                            n_paths=N_PATHS_EVAL,
                        )
                        cell["per_seed"][str(seed)] = rec
                        wall = time.time() - t0
                        print(
                            f"      DH ES = {rec['dh']['es_95']:.4f}  "
                            f"BS ES = {rec['bs']['es_95']:.4f}  "
                            f"gap = {rec['gap_dh_minus_bs_es95']:+.4f}  "
                            f"({wall:.1f}s)",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"      ERROR seed {seed}: {exc}", flush=True)
                        cell["per_seed"][str(seed)] = {"error": str(exc)}
                # Per-cell aggregate
                dh_vals = [r2["dh"]["es_95"] for r2 in cell["per_seed"].values()
                            if isinstance(r2, dict) and "dh" in r2]
                bs_vals = [r2["bs"]["es_95"] for r2 in cell["per_seed"].values()
                            if isinstance(r2, dict) and "bs" in r2]
                gap_vals = [r2["gap_dh_minus_bs_es95"] for r2 in cell["per_seed"].values()
                            if isinstance(r2, dict) and "gap_dh_minus_bs_es95" in r2]
                cell["aggregate"] = {
                    "dh_es95": _agg(dh_vals),
                    "bs_es95": _agg(bs_vals),
                    "gap": _agg(gap_vals),
                }
                _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                            intermediate_path)

    # Crossover analysis: smallest r* such that any axis-direction has DH >= BS
    r_star = None
    crossover_cell = None
    for r in sorted(radii):
        for axis in axes:
            for direction in M1_DIRECTIONS:
                cell = out.get(axis, {}).get(direction, {}).get(f"{r:g}", {})
                ag = cell.get("aggregate", {})
                if (ag.get("gap", {}).get("mean", -1) >= 0) and r_star is None:
                    r_star = r
                    crossover_cell = (axis, direction, r,
                                       ag["dh_es95"]["mean"], ag["bs_es95"]["mean"])
                    break
            if r_star is not None: break
        if r_star is not None: break

    crossover = (
        {"r_star": r_star, "axis_direction_radius_dh_bs": crossover_cell}
        if r_star is not None else
        {"r_star": None, "note": "no crossover up to r=" + str(max(radii))}
    )

    final = {
        "meta": _meta(),
        "results": _strip_for_json(out),
        "crossover_analysis": crossover,
    }
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# M.2 — Higher-resolution axis sweeps
# --------------------------------------------------------------------------

# 15-point grids spanning approximately ±3σ from baseline, slightly asymmetric
# to respect physical bounds (especially H near 0)
M2_GRIDS: dict[str, list[float]] = {
    "H":   [round(0.07 + d, 4) for d in
            (-0.05, -0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0,
             0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)],
    "eta": [round(1.9 + d, 4) for d in
            (-1.5, -1.0, -0.7, -0.5, -0.3, -0.15, -0.05, 0.0,
             0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 1.5)],
    "rho": [round(-0.7 + d, 4) for d in
            (-0.25, -0.20, -0.15, -0.10, -0.05, -0.025, -0.01, 0.0,
             0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60)],
}
M2_SEEDS = [8101, 8102, 8103, 8104, 8105]

def run_M2_axis_sweep(
    grids: dict[str, list[float]] | None = None,
    seeds: list[int] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.2: 15-point axis sweeps × 5 seeds."""
    if grids is None:
        grids = {ax: list(g) for ax, g in M2_GRIDS.items()}
    if seeds is None:
        seeds = list(M2_SEEDS)

    n_evals = sum(len(g) for g in grids.values()) * len(seeds)
    print("=" * 70, flush=True)
    print(f"  M.2 — Axis sweeps "
          f"(3 axes × 15 grid × {len(seeds)} seeds = {n_evals} evals)", flush=True)
    print("=" * 70, flush=True)

    canonical_dh = _load_canonical_dh()
    intermediate_path = out_path or (OUT_DIR / "M2_axis_sweep.json")

    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            out = existing.get("results", {})
        except Exception:
            out = {}

    for axis in grids:
        if axis not in out:
            out[axis] = {}
        for val in grids[axis]:
            key = f"{val:.4f}"
            if key not in out[axis]:
                out[axis][key] = {"axis": axis, "value": val,
                                    "per_seed": {}, "aggregate": {}}
            cell = out[axis][key]
            for seed in seeds:
                if str(seed) in cell["per_seed"] and "dh" in cell["per_seed"][str(seed)]:
                    continue
                # Build perturbed parameter vector
                if axis == "H":
                    Hp, etap, rhop = _clamp_to_box(val, ETA_BL, RHO_BL)
                elif axis == "eta":
                    Hp, etap, rhop = _clamp_to_box(H_BL, val, RHO_BL)
                elif axis == "rho":
                    Hp, etap, rhop = _clamp_to_box(H_BL, ETA_BL, val)
                print(
                    f"    axis={axis} val={val} seed={seed} "
                    f"(H={Hp:.4f}, eta={etap:.4f}, rho={rhop:.4f}) ...",
                    flush=True,
                )
                t0 = time.time()
                try:
                    rec = _evaluate_strategies_at(
                        Hp, etap, rhop, seed=seed,
                        canonical_dh=canonical_dh,
                        n_paths=N_PATHS_EVAL,
                    )
                    cell["per_seed"][str(seed)] = rec
                    wall = time.time() - t0
                    print(
                        f"      DH ES = {rec['dh']['es_95']:.4f}  "
                        f"BS ES = {rec['bs']['es_95']:.4f}  "
                        f"gap = {rec['gap_dh_minus_bs_es95']:+.4f}  "
                        f"({wall:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"      ERROR: {exc}", flush=True)
                    cell["per_seed"][str(seed)] = {"error": str(exc)}
            dh_vals = [r["dh"]["es_95"] for r in cell["per_seed"].values()
                        if isinstance(r, dict) and "dh" in r]
            bs_vals = [r["bs"]["es_95"] for r in cell["per_seed"].values()
                        if isinstance(r, dict) and "bs" in r]
            gap_vals = [r["gap_dh_minus_bs_es95"] for r in cell["per_seed"].values()
                        if isinstance(r, dict) and "gap_dh_minus_bs_es95" in r]
            cell["aggregate"] = {
                "dh_es95": _agg(dh_vals),
                "bs_es95": _agg(bs_vals),
                "gap": _agg(gap_vals),
            }
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)

    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# M.3 — Joint 3D PGD
# --------------------------------------------------------------------------

M3_RADII = [1.0, 2.0, 3.0, 4.0, 5.0]
M3_SEEDS = [8201, 8202, 8203, 8204, 8205]
M3_PGD_STEPS = 30
M3_PGD_LR = 0.05  # In normalised coordinates; will scale per-axis

def _pgd_joint(
    canonical_dh: DeepHedgerFNN,
    radius: float, seed: int,
    objective: str = "dh",
    n_steps: int = M3_PGD_STEPS,
    lr: float = M3_PGD_LR,
    n_paths: int = N_PATHS_PGD,
) -> dict[str, Any]:
    """3D PGD: maximise <objective> over (dH, deta, drho) within radius."""
    # State in normalised coords (so step is uniform across axes)
    delta_norm = np.zeros(3)  # (dH/sH, deta/sEta, drho/sRho)
    history = []

    # p0 at baseline (we keep p0 fixed during PGD to avoid bias from a moving target)
    S_p0, _ = _simulate_perturbed(
        H_BL, ETA_BL, RHO_BL,
        n_paths=n_paths, seed=seed + 200_000,
    )
    p0 = float(compute_payoff(S_p0, K, "call").mean())
    del S_p0
    gc.collect()

    for step in range(n_steps):
        # Perturbed parameters
        dH = delta_norm[0] * SIGMA_AXES["H"]
        deta = delta_norm[1] * SIGMA_AXES["eta"]
        drho = delta_norm[2] * SIGMA_AXES["rho"]
        H_pert = H_BL + dH
        eta_pert = ETA_BL + deta
        rho_pert = RHO_BL + drho
        # Use a different seed each step for unbiased gradients
        step_seed = seed * 1000 + step
        try:
            es_val, grads = _evaluate_with_grad(
                H_pert, eta_pert, rho_pert, seed=step_seed,
                canonical_dh=canonical_dh, p0=p0,
                objective=objective,
                n_paths=n_paths,
            )
        except Exception as exc:
            history.append({"step": step, "error": str(exc)})
            break
        # Convert grad back to normalised coords (chain rule)
        grad_norm = np.array([
            grads["H"] * SIGMA_AXES["H"],
            grads["eta"] * SIGMA_AXES["eta"],
            grads["rho"] * SIGMA_AXES["rho"],
        ])
        # Take ascent step in normalised coords
        delta_norm = delta_norm + lr * grad_norm
        # Project onto normalised ball
        nm = float(np.linalg.norm(delta_norm))
        if nm > radius and nm > 1e-12:
            delta_norm = delta_norm * (radius / nm)
        # Box clamping (back-transform to physical coords)
        dH = delta_norm[0] * SIGMA_AXES["H"]
        deta = delta_norm[1] * SIGMA_AXES["eta"]
        drho = delta_norm[2] * SIGMA_AXES["rho"]
        Hp, etap, rhop = _clamp_to_box(H_BL + dH, ETA_BL + deta, RHO_BL + drho)
        # Re-encode possibly-clamped position
        delta_norm = np.array([
            (Hp - H_BL) / SIGMA_AXES["H"],
            (etap - ETA_BL) / SIGMA_AXES["eta"],
            (rhop - RHO_BL) / SIGMA_AXES["rho"],
        ])
        history.append({
            "step": step, "es": es_val,
            "delta_norm": delta_norm.tolist(),
            "H": Hp, "eta": etap, "rho": rhop,
        })

    # Final unbiased eval (50k paths)
    final_dH = delta_norm[0] * SIGMA_AXES["H"]
    final_deta = delta_norm[1] * SIGMA_AXES["eta"]
    final_drho = delta_norm[2] * SIGMA_AXES["rho"]
    Hf, etaf, rhof = _clamp_to_box(
        H_BL + final_dH, ETA_BL + final_deta, RHO_BL + final_drho,
    )
    final_eval = _evaluate_strategies_at(
        Hf, etaf, rhof, seed=seed + 300_000,
        canonical_dh=canonical_dh, p0_override=p0,
        n_paths=N_PATHS_EVAL,
    )

    return {
        "radius": radius,
        "seed": seed,
        "objective": objective,
        "p0": p0,
        "final": final_eval,
        "delta_norm_final": delta_norm.tolist(),
        "norm_final": float(np.linalg.norm(delta_norm)),
        "history_tail": history[-5:] if history else [],
        "n_steps_completed": len(history),
    }

def run_M3_joint(
    radii: list[float] | None = None,
    seeds: list[int] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.3: joint 3D PGD attacks."""
    if radii is None:
        radii = list(M3_RADII)
    if seeds is None:
        seeds = list(M3_SEEDS)

    print("=" * 70, flush=True)
    print(f"  M.3 — Joint 3D PGD ({len(radii)} radii × {len(seeds)} seeds = "
          f"{len(radii)*len(seeds)} attacks)", flush=True)
    print("=" * 70, flush=True)

    canonical_dh = _load_canonical_dh()
    intermediate_path = out_path or (OUT_DIR / "M3_joint_attacks.json")

    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            out = existing.get("results", {})
        except Exception:
            out = {}

    for r in radii:
        rkey = f"{r:g}"
        if rkey not in out:
            out[rkey] = {"radius": r, "per_seed": {}, "aggregate": {}}
        cell = out[rkey]
        for seed in seeds:
            if str(seed) in cell["per_seed"] and "final" in cell["per_seed"][str(seed)]:
                continue
            print(f"\n    r={r}  seed={seed}  (DH-attack PGD)...", flush=True)
            t0 = time.time()
            rec = _pgd_joint(
                canonical_dh, radius=r, seed=seed,
                objective="dh",
            )
            cell["per_seed"][str(seed)] = rec
            wall = time.time() - t0
            fin = rec["final"]
            print(
                f"      final  DH ES = {fin['dh']['es_95']:.4f}  "
                f"BS ES = {fin['bs']['es_95']:.4f}  "
                f"gap = {fin['gap_dh_minus_bs_es95']:+.4f}  "
                f"|Δ|={rec['norm_final']:.3f}  "
                f"({wall:.1f}s)",
                flush=True,
            )
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)
        # Aggregate
        dh_es = [r2["final"]["dh"]["es_95"] for r2 in cell["per_seed"].values()
                  if isinstance(r2, dict) and "final" in r2]
        bs_es = [r2["final"]["bs"]["es_95"] for r2 in cell["per_seed"].values()
                  if isinstance(r2, dict) and "final" in r2]
        gap = [r2["final"]["gap_dh_minus_bs_es95"] for r2 in cell["per_seed"].values()
                if isinstance(r2, dict) and "final" in r2]
        cell["aggregate"] = {
            "dh_es95": _agg(dh_es),
            "bs_es95": _agg(bs_es),
            "gap": _agg(gap),
        }
        _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                    intermediate_path)

    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# M.4 — Targeted attacks on DH
# --------------------------------------------------------------------------

M4_RADII = [1.0, 2.0, 3.0]
M4_SEEDS = [8301, 8302, 8303]

def run_M4_targeted(
    radii: list[float] | None = None,
    seeds: list[int] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.4: PGD with objective = ES_DH − ES_BS (find DH-vulnerable directions)."""
    if radii is None:
        radii = list(M4_RADII)
    if seeds is None:
        seeds = list(M4_SEEDS)

    print("=" * 70, flush=True)
    print(f"  M.4 — Targeted DH attacks ({len(radii)} radii × {len(seeds)} seeds × 2 modes = "
          f"{len(radii)*len(seeds)*2} attacks)", flush=True)
    print("=" * 70, flush=True)

    canonical_dh = _load_canonical_dh()
    intermediate_path = out_path or (OUT_DIR / "M4_targeted_attacks.json")

    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            out = existing.get("results", {})
        except Exception:
            out = {}

    # Two modes: "dh_targeted" maximises DH-BS gap; "dh_favorable" minimises it (= maximise BS-DH).
    # Implementation: same _pgd_joint but with different objective.
    #   "dh_targeted" → ascend ES_DH - ES_BS
    #   "dh_favorable" → ascend ES_BS - ES_DH (so we negate the gradient by using objective="dh_minus_bs"
    #     and a sign-flipped lr)
    for mode in ("dh_targeted", "dh_favorable"):
        if mode not in out:
            out[mode] = {}
        for r in radii:
            rkey = f"{r:g}"
            if rkey not in out[mode]:
                out[mode][rkey] = {"radius": r, "per_seed": {}, "aggregate": {}}
            cell = out[mode][rkey]
            for seed in seeds:
                if str(seed) in cell["per_seed"] and "final" in cell["per_seed"][str(seed)]:
                    continue
                print(f"\n    mode={mode} r={r} seed={seed} ...", flush=True)
                t0 = time.time()
                # Use same PGD structure as M.3 but with dh_minus_bs objective.
                # For dh_targeted, lr is positive (ascend gap).
                # For dh_favorable, use negative lr (descend gap).
                lr = M3_PGD_LR if mode == "dh_targeted" else -M3_PGD_LR
                rec = _pgd_joint(
                    canonical_dh, radius=r, seed=seed,
                    objective="dh_minus_bs",
                    lr=lr,
                )
                rec["mode"] = mode
                cell["per_seed"][str(seed)] = rec
                wall = time.time() - t0
                fin = rec["final"]
                print(
                    f"      final  DH ES = {fin['dh']['es_95']:.4f}  "
                    f"BS ES = {fin['bs']['es_95']:.4f}  "
                    f"gap = {fin['gap_dh_minus_bs_es95']:+.4f}  "
                    f"(H={fin['H']:.4f}, eta={fin['eta']:.4f}, rho={fin['rho']:.4f})  "
                    f"({wall:.1f}s)",
                    flush=True,
                )
                _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                            intermediate_path)
            dh_es = [r2["final"]["dh"]["es_95"] for r2 in cell["per_seed"].values()
                      if isinstance(r2, dict) and "final" in r2]
            bs_es = [r2["final"]["bs"]["es_95"] for r2 in cell["per_seed"].values()
                      if isinstance(r2, dict) and "final" in r2]
            gap = [r2["final"]["gap_dh_minus_bs_es95"] for r2 in cell["per_seed"].values()
                    if isinstance(r2, dict) and "final" in r2]
            cell["aggregate"] = {
                "dh_es95": _agg(dh_es),
                "bs_es95": _agg(bs_es),
                "gap": _agg(gap),
            }
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)

    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# M.5 — Objective-dependent robustness
# --------------------------------------------------------------------------

M5_OBJECTIVES = ["es_090", "es_095", "es_099", "mse", "entropic"]
M5_SEEDS = [8401, 8402, 8403, 8404, 8405]
M5_RADII = [1.0, 2.0, 3.0]

def _objective_to_risk_fn(objective: str):
    """Return a risk function (Tensor → scalar Tensor) corresponding to <objective>."""
    if objective == "es_090":
        return lambda pnl: expected_shortfall(pnl, 0.90)
    if objective == "es_095":
        return lambda pnl: expected_shortfall(pnl, 0.95)
    if objective == "es_099":
        return lambda pnl: expected_shortfall(pnl, 0.99)
    if objective == "mse":
        # Mean-squared P&L (penalises both gains and losses; classical hedging objective)
        return lambda pnl: (pnl ** 2).mean()
    if objective == "entropic":
        return lambda pnl: entropic_risk(pnl, lam=1.0)
    raise ValueError(objective)

def _train_dh_with_objective(
    objective: str, seed: int,
    n_train: int = 80_000, n_val: int = 20_000,
    epochs: int = 200, patience: int = 30,
) -> tuple[DeepHedgerFNN, float]:
    """Train DH on canonical rough Bergomi paths with the given risk objective."""
    # Generate paths
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H_BL, eta=ETA_BL, rho=RHO_BL, xi0=XI0_BL,
    )
    with torch.no_grad():
        S_train, _, _ = sim.simulate(n_paths=n_train, S0=S0, seed=seed)
        S_val, _, _ = sim.simulate(n_paths=n_val, S0=S0, seed=seed + 1)
    # p0
    with torch.no_grad():
        S_p0, _, _ = sim.simulate(n_paths=n_train, S0=S0, seed=seed + 2)
        p0 = float(compute_payoff(S_p0, K, "call").mean())
        del S_p0
    gc.collect()

    # Seed protocol
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)

    risk_fn = _objective_to_risk_fn(objective)
    t0 = time.time()
    history = train_deep_hedger(
        model, S_train, S_val,
        K=K, T=T, S0=S0, p0=p0,
        cost_lambda=0.0, risk_fn=risk_fn,
        lr=1e-3, batch_size=2048,
        epochs=epochs, patience=patience,
        verbose=False,
    )
    train_time = time.time() - t0
    return model, train_time, history, p0

def run_M5_objective_robustness(
    objectives: list[str] | None = None,
    seeds: list[int] | None = None,
    radii: list[float] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.5: train DH with 4 objectives (skip canonical es_095) and evaluate
    robustness via M.1-style sweeps."""
    if objectives is None:
        objectives = list(M5_OBJECTIVES)
    if seeds is None:
        seeds = list(M5_SEEDS)
    if radii is None:
        radii = list(M5_RADII)

    print("=" * 70, flush=True)
    print(f"  M.5 — Objective robustness "
          f"({len(objectives)} obj × {len(seeds)} seeds × "
          f"{len(radii)} radii × 6 axis-dir = "
          f"{len(objectives)*len(seeds)*len(radii)*6} eval cells; "
          f"{(len(objectives)-1)*len(seeds)} trainings)",
          flush=True)
    print("=" * 70, flush=True)

    intermediate_path = out_path or (OUT_DIR / "M5_objective_robustness.json")
    canonical_dh = _load_canonical_dh()

    out: dict[str, Any] = {}
    if intermediate_path.exists():
        try:
            existing = json.load(intermediate_path.open())
            out = existing.get("results", {})
        except Exception:
            out = {}

    for objective in objectives:
        if objective not in out:
            out[objective] = {"per_seed": {}}
        for seed in seeds:
            seed_key = str(seed)
            if seed_key in out[objective]["per_seed"] and \
                    out[objective]["per_seed"][seed_key].get("complete", False):
                print(f"  {objective} seed={seed} SKIP (already complete)", flush=True)
                continue
            ckpt_path = M5_CHECKPOINTS_DIR / f"dh_{objective}_seed{seed}.pt"

            if objective == "es_095":
                # Reuse canonical checkpoint for all 5 seeds (canonical is the same
                # network regardless of seed; it was trained on a different seeded run).
                model = canonical_dh
                train_time = 0.0
                best_epoch = -1
                p0_train = None
            elif ckpt_path.exists():
                print(f"  {objective} seed={seed} loading checkpoint ...", flush=True)
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model = DeepHedgerFNN(input_dim=4, hidden_dim=128, n_res_blocks=2)
                model.load_state_dict(state)
                train_time = 0.0
                best_epoch = -1
                p0_train = None
            else:
                print(f"  {objective} seed={seed} TRAINING ...", flush=True)
                t0 = time.time()
                model, train_time, history, p0_train = _train_dh_with_objective(
                    objective, seed,
                )
                best_epoch = int(history.get("best_epoch", -1))
                torch.save(model.state_dict(), ckpt_path)
                print(f"    trained in {train_time/60:.1f} min "
                      f"(best epoch {best_epoch}); checkpoint saved",
                      flush=True)

            # Run M.1-style axis-aligned sweep at radii × 6 axis-dir × eval seeds
            cell = out[objective]["per_seed"].setdefault(seed_key, {
                "objective": objective, "seed": seed,
                "train_time_s": train_time,
                "best_epoch": best_epoch,
                "axis_sweep": {},
            })
            cell["train_time_s"] = train_time
            cell["best_epoch"] = best_epoch
            sweep = cell.setdefault("axis_sweep", {})

            eval_seed_base = seed * 10
            for r in radii:
                rkey = f"{r:g}"
                if rkey not in sweep:
                    sweep[rkey] = {"per_axis_dir": {}}
                for axis in M1_AXES:
                    for direction in M1_DIRECTIONS:
                        key = f"{axis}{direction}"
                        if key in sweep[rkey]["per_axis_dir"]:
                            continue
                        Hp, etap, rhop = _axis_perturb(axis, direction, r)
                        try:
                            rec = _evaluate_strategies_at(
                                Hp, etap, rhop, seed=eval_seed_base,
                                canonical_dh=model,
                                n_paths=N_PATHS_EVAL,
                            )
                            sweep[rkey]["per_axis_dir"][key] = {
                                "axis": axis, "direction": direction,
                                "H": Hp, "eta": etap, "rho": rhop,
                                "dh_es95": float(rec["dh"]["es_95"]),
                                "bs_es95": float(rec["bs"]["es_95"]),
                                "gap": float(rec["gap_dh_minus_bs_es95"]),
                            }
                        except Exception as exc:
                            sweep[rkey]["per_axis_dir"][key] = {"error": str(exc)}
                # worst-case across 6 axis-dir at this radius
                vals = [v["dh_es95"] for v in sweep[rkey]["per_axis_dir"].values()
                        if isinstance(v, dict) and "dh_es95" in v]
                if vals:
                    sweep[rkey]["worst_dh_es95"] = max(vals)
                _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                            intermediate_path)
            cell["complete"] = True
            print(f"  {objective} seed={seed} sweep complete: "
                  f"worst@r=2 = {sweep['2']['worst_dh_es95']:.4f}",
                  flush=True)
            _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                        intermediate_path)
            del model
            gc.collect()

        # Aggregate over seeds at each radius
        agg = out[objective].setdefault("aggregate_per_radius", {})
        for r in radii:
            rkey = f"{r:g}"
            worst_vals = []
            for seed_key in (str(s) for s in seeds):
                cell = out[objective]["per_seed"].get(seed_key, {})
                w = cell.get("axis_sweep", {}).get(rkey, {}).get("worst_dh_es95")
                if w is not None:
                    worst_vals.append(w)
            if worst_vals:
                agg[rkey] = _agg(worst_vals)
        _save_json({"meta": _meta(), "results": _strip_for_json(out)},
                    intermediate_path)

    final = {"meta": _meta(), "results": _strip_for_json(out)}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# M.6 — Hessian eigenstructure
# --------------------------------------------------------------------------

def _es_at(
    H: float, eta: float, rho: float, strategy: str,
    canonical_dh: DeepHedgerFNN,
    n_paths: int = 30_000, seed: int = 9001,
    p0: float | None = None,
) -> float:
    """Compute ES_0.95 for {bs, dh} strategy at perturbed (H, eta, rho)."""
    sim = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H, eta=eta, rho=rho, xi0=XI0_BL,
    )
    with torch.no_grad():
        S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)
    if p0 is None:
        with torch.no_grad():
            S_p0, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed + 1)
            p0 = float(compute_payoff(S_p0, K, "call").mean())
            del S_p0
        gc.collect()
    payoff = compute_payoff(S, K, "call")
    if strategy == "bs":
        bs = BlackScholesDelta(sigma=SIGMA_BS, K=K, T=T)
        deltas = bs.hedge_paths(S)
    else:
        with torch.no_grad():
            deltas = canonical_dh.hedge_paths(S.float(), T, S0).to(S.dtype)
    pnl = compute_hedging_pnl(S, deltas, payoff, p0, 0.0)
    es = expected_shortfall(pnl, 0.95)
    return float(es.item())

def _hessian_fd(
    canonical_dh: DeepHedgerFNN,
    strategy: str,
    h_factor: float = 0.01,
    n_paths: int = 30_000,
    seed: int = 9001,
    p0: float | None = None,
) -> np.ndarray:
    """Symmetric finite-difference Hessian of ES_0.95 wrt (H, eta, rho).

    Using central differences with step h_axis = h_factor * SIGMA_AXES[axis]:
        ∂²f/∂x∂y ≈ (f(x+h, y+k) - f(x+h, y-k) - f(x-h, y+k) + f(x-h, y-k)) / (4*h*k)
    """
    axes_order = ["H", "eta", "rho"]
    h = {ax: h_factor * SIGMA_AXES[ax] for ax in axes_order}
    base = {"H": H_BL, "eta": ETA_BL, "rho": RHO_BL}

    # Compute p0 once at baseline (consistent across all FD evals)
    if p0 is None:
        with torch.no_grad():
            sim_bl = DifferentiableRoughBergomi(
                n_steps=N_STEPS, T=T, H=H_BL, eta=ETA_BL, rho=RHO_BL, xi0=XI0_BL,
            )
            S_p0, _, _ = sim_bl.simulate(n_paths=n_paths, S0=S0, seed=seed - 1)
            p0 = float(compute_payoff(S_p0, K, "call").mean())
            del S_p0
        gc.collect()

    def f_at(dH: float, deta: float, drho: float) -> float:
        return _es_at(
            base["H"] + dH, base["eta"] + deta, base["rho"] + drho,
            strategy, canonical_dh, n_paths=n_paths, seed=seed, p0=p0,
        )

    H_mat = np.zeros((3, 3))
    # Diagonal entries via central differences
    for i, axi in enumerate(axes_order):
        h_i = h[axi]
        d_p = [0.0, 0.0, 0.0]; d_p[i] = h_i
        d_m = [0.0, 0.0, 0.0]; d_m[i] = -h_i
        f_pp = f_at(*d_p)
        f_mm = f_at(*d_m)
        f_0 = f_at(0.0, 0.0, 0.0)
        H_mat[i, i] = (f_pp - 2 * f_0 + f_mm) / (h_i ** 2)
    # Off-diagonal via central-mixed differences
    for i in range(3):
        for j in range(i + 1, 3):
            axi, axj = axes_order[i], axes_order[j]
            h_i, h_j = h[axi], h[axj]
            d_pp = [0.0, 0.0, 0.0]; d_pp[i] = h_i; d_pp[j] = h_j
            d_pm = [0.0, 0.0, 0.0]; d_pm[i] = h_i; d_pm[j] = -h_j
            d_mp = [0.0, 0.0, 0.0]; d_mp[i] = -h_i; d_mp[j] = h_j
            d_mm = [0.0, 0.0, 0.0]; d_mm[i] = -h_i; d_mm[j] = -h_j
            val = (f_at(*d_pp) - f_at(*d_pm) - f_at(*d_mp) + f_at(*d_mm)) \
                  / (4 * h_i * h_j)
            H_mat[i, j] = val
            H_mat[j, i] = val
    # Symmetrise
    H_mat = 0.5 * (H_mat + H_mat.T)
    return H_mat

def run_M6_hessian(
    h_factors: list[float] | None = None,
    seed: int = 9001,
    n_paths: int = 30_000,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """M.6: Hessian eigenstructure analysis at baseline calibration."""
    if h_factors is None:
        h_factors = [0.005, 0.01, 0.02]

    print("=" * 70, flush=True)
    print(f"  M.6 — Hessian eigenstructure ({len(h_factors)} step sizes × 2 strategies)",
          flush=True)
    print("=" * 70, flush=True)

    canonical_dh = _load_canonical_dh()
    intermediate_path = out_path or (OUT_DIR / "M6_hessian.json")

    # p0 at baseline
    sim_bl = DifferentiableRoughBergomi(
        n_steps=N_STEPS, T=T, H=H_BL, eta=ETA_BL, rho=RHO_BL, xi0=XI0_BL,
    )
    with torch.no_grad():
        S_p0, _, _ = sim_bl.simulate(n_paths=n_paths, S0=S0, seed=seed - 1)
        p0 = float(compute_payoff(S_p0, K, "call").mean())
        del S_p0
    gc.collect()

    out: dict[str, Any] = {"p0": p0, "results": {}}
    for strategy in ("bs", "dh"):
        out["results"][strategy] = {}
        for h_factor in h_factors:
            print(f"\n  {strategy} h_factor={h_factor} ...", flush=True)
            t0 = time.time()
            H_mat = _hessian_fd(
                canonical_dh, strategy=strategy,
                h_factor=h_factor, n_paths=n_paths, seed=seed, p0=p0,
            )
            wall = time.time() - t0
            evals, evecs = np.linalg.eigh(H_mat)  # ascending order
            # Sort descending
            order = np.argsort(-evals)
            evals = evals[order]
            evecs = evecs[:, order]
            print(f"    eigenvalues = {evals}  ({wall:.1f}s)", flush=True)
            out["results"][strategy][f"{h_factor:g}"] = {
                "h_factor": h_factor,
                "hessian": H_mat.tolist(),
                "eigenvalues": evals.tolist(),
                "eigenvectors": evecs.tolist(),
            }
            _save_json({"meta": _meta(), **out}, intermediate_path)

    # Compare DH vs BS top-1 eigenvectors at h_factor=0.01 (middle sample)
    h_ref = "0.01"
    if h_ref in out["results"]["dh"] and h_ref in out["results"]["bs"]:
        v_dh = np.array(out["results"]["dh"][h_ref]["eigenvectors"])[:, 0]
        v_bs = np.array(out["results"]["bs"][h_ref]["eigenvectors"])[:, 0]
        cos = float(abs(np.dot(v_dh, v_bs) / (np.linalg.norm(v_dh) * np.linalg.norm(v_bs))))
        ratio = float(out["results"]["dh"][h_ref]["eigenvalues"][0]
                       / out["results"]["bs"][h_ref]["eigenvalues"][0])
        out["comparison"] = {
            "h_factor_ref": h_ref,
            "top1_eigenvector_cosine_DH_BS": cos,
            "top1_eigenvalue_ratio_DH_over_BS": ratio,
            "top1_eigenvector_DH": v_dh.tolist(),
            "top1_eigenvector_BS": v_bs.tolist(),
        }

    final = {"meta": _meta(), **out}
    _save_json(final, intermediate_path)
    return final

# --------------------------------------------------------------------------
# Setup verifier
# --------------------------------------------------------------------------

def setup_verify() -> None:
    """Sanity-check infrastructure before launching long runs."""
    print("=" * 70, flush=True)
    print("  M setup verification", flush=True)
    print("=" * 70, flush=True)
    print(f"  REPO_ROOT       = {REPO_ROOT}", flush=True)
    print(f"  OUT_DIR         = {OUT_DIR}", flush=True)
    print(f"  FIG_DIR         = {FIG_DIR}", flush=True)
    print(f"  CANONICAL_DH    = {'OK' if CANONICAL_DH_PATH.exists() else 'MISSING'} "
          f"({CANONICAL_DH_PATH})", flush=True)
    canonical_dh = _load_canonical_dh()
    print(f"  Loaded DH params: {sum(p.numel() for p in canonical_dh.parameters())}",
          flush=True)
    # Quick eval to confirm pipeline works
    rec = _evaluate_strategies_at(
        H_BL, ETA_BL, RHO_BL, seed=2024,
        canonical_dh=canonical_dh,
        n_paths=10_000,
    )
    print(f"  Baseline (H={H_BL}, eta={ETA_BL}, rho={RHO_BL}, seed=2024, N=10k):",
          flush=True)
    print(f"    p0 = {rec['p0']:.4f}", flush=True)
    print(f"    DH ES_0.95 = {rec['dh']['es_95']:.4f}", flush=True)
    print(f"    BS ES_0.95 = {rec['bs']['es_95']:.4f}", flush=True)
    print(f"    gap        = {rec['gap_dh_minus_bs_es95']:+.4f}", flush=True)

# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------

def repro_M1(out_path: Path | None = None) -> None:
    out_path = out_path or (OUT_DIR / "M1_repro.json")
    res = run_M1_extended_radius(
        radii=[2.0], axes=["H"], seeds=[8001], out_path=out_path,
    )
    _save_json(res, out_path)

def repro_M2(out_path: Path | None = None) -> None:
    out_path = out_path or (OUT_DIR / "M2_repro.json")
    res = run_M2_axis_sweep(
        grids={"eta": [2.4]}, seeds=[8101], out_path=out_path,
    )
    _save_json(res, out_path)

def repro_M3(out_path: Path | None = None) -> None:
    out_path = out_path or (OUT_DIR / "M3_repro.json")
    res = run_M3_joint(radii=[2.0], seeds=[8201], out_path=out_path)
    _save_json(res, out_path)

def repro_M4(out_path: Path | None = None) -> None:
    out_path = out_path or (OUT_DIR / "M4_repro.json")
    # Single mode (dh_targeted), single radius, single seed
    canonical_dh = _load_canonical_dh()
    rec = _pgd_joint(
        canonical_dh, radius=2.0, seed=8301,
        objective="dh_minus_bs", lr=M3_PGD_LR,
    )
    _save_json({"meta": _meta(), "result": _strip_for_json(rec)}, out_path)

def repro_M5(out_path: Path | None = None) -> None:
    out_path = out_path or (OUT_DIR / "M5_repro.json")
    res = run_M5_objective_robustness(
        objectives=["mse"], seeds=[8401], radii=[2.0], out_path=out_path,
    )
    _save_json(res, out_path)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--M1", action="store_true")
    parser.add_argument("--M2", action="store_true")
    parser.add_argument("--M3", action="store_true")
    parser.add_argument("--M4", action="store_true")
    parser.add_argument("--M5", action="store_true")
    parser.add_argument("--M6", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repro-M1", action="store_true")
    parser.add_argument("--repro-M2", action="store_true")
    parser.add_argument("--repro-M3", action="store_true")
    parser.add_argument("--repro-M4", action="store_true")
    parser.add_argument("--repro-M5", action="store_true")
    parser.add_argument("--repro-output", type=str, default=None)
    args = parser.parse_args()

    if args.setup:
        setup_verify()
        return

    if args.repro_M1:
        repro_M1(Path(args.repro_output) if args.repro_output else None); return
    if args.repro_M2:
        repro_M2(Path(args.repro_output) if args.repro_output else None); return
    if args.repro_M3:
        repro_M3(Path(args.repro_output) if args.repro_output else None); return
    if args.repro_M4:
        repro_M4(Path(args.repro_output) if args.repro_output else None); return
    if args.repro_M5:
        repro_M5(Path(args.repro_output) if args.repro_output else None); return

    if args.all or args.M1: run_M1_extended_radius()
    if args.all or args.M2: run_M2_axis_sweep()
    if args.all or args.M3: run_M3_joint()
    if args.all or args.M4: run_M4_targeted()
    if args.all or args.M5: run_M5_objective_robustness()
    if args.all or args.M6: run_M6_hessian()

if __name__ == "__main__":
    main()
