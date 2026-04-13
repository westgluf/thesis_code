#!/usr/bin/env python
"""
H2 grid extension: BS delta at finer frequency x cost grid.

Extends Part A of Prompt 9 (Pareto front) to finer resolution on
BOTH axes, using BS delta only (no training). Reuses existing
Prompt 9 Part A cells where the grids overlap.

Input:
    figures/pareto_part_A_results.json   (Prompt 9 Part A scalar results)

Outputs:
    figures/h2_grid_extension.json       (full 6x7 grid)
    figures/h2_grid_extension.log        (run log)

Extended grid:
    freq_values = [25, 50, 100, 200, 400, 800]
    cost_values = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010]

Run:
    python -u -m deep_hedging.experiments.h2_grid_extension
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.delta_hedger import BlackScholesDelta
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics
from deep_hedging.utils.config import get_device, set_global_seed


# ─── Fixed rBergomi parameters (match Prompt 9 Part A) ─────────
RBG_PARAMS: dict[str, float] = dict(
    H=0.07,
    eta=1.9,
    rho=-0.7,
    xi0=0.235 ** 2,
)
S0 = 100.0
K = 100.0
T = 1.0
SIGMA_ASSUMED = float(np.sqrt(RBG_PARAMS["xi0"]))  # sqrt(xi0) ~ 0.235
N_TEST = 50_000
MASTER_SEED = 2024

# ─── Extended grid ─────────────────────────────────────────────
FREQ_VALUES: list[int] = [25, 50, 100, 200, 400, 800]
COST_VALUES: list[float] = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010]

# Prompt 9 Part A grid (for reuse)
PROMPT_9_FREQ: set[int] = {50, 100, 200}
PROMPT_9_COST: set[float] = {0.0, 0.001, 0.002}

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
PROMPT_9_JSON = FIGURE_DIR / "pareto_part_A_results.json"
OUT_JSON = FIGURE_DIR / "h2_grid_extension.json"


# =======================================================================
# Loader for Prompt 9 cells
# =======================================================================

def _match_cost_key(cost_dict: dict, target_cost: float, tol: float = 1e-9) -> str | None:
    """Find the string key in cost_dict that represents ``target_cost``."""
    for k in cost_dict:
        try:
            if abs(float(k) - target_cost) < tol:
                return k
        except (TypeError, ValueError):
            continue
    return None


def load_prompt_9_cells(path: Path) -> dict[int, dict[float, dict]]:
    """Load BS delta cells from Prompt 9 Part A that overlap with our grid.

    Returns nested dict ``{n_steps: {cost_lambda: {...}}}``.
    """
    if not path.exists():
        print(f"  Prompt 9 JSON not found at {path} — will compute all cells fresh",
              flush=True)
        return {}

    with open(path) as f:
        data = json.load(f)

    if "bs" not in data:
        return {}

    out: dict[int, dict[float, dict]] = {}
    bs_tree = data["bs"]
    for n_key, cost_dict in bs_tree.items():
        try:
            n_steps = int(n_key)
        except (TypeError, ValueError):
            continue
        if n_steps not in PROMPT_9_FREQ:
            continue

        out[n_steps] = {}
        for target_cost in PROMPT_9_COST:
            k = _match_cost_key(cost_dict, target_cost)
            if k is None:
                continue
            cell = cost_dict[k]
            # Defensive copy of scalar metric fields only
            out[n_steps][target_cost] = {
                "metrics": dict(cell.get("metrics", {})),
                "mean_turnover": float(cell.get("mean_turnover", 0.0)),
                "source": "prompt_9",
            }
    return out


# =======================================================================
# Path generation and BS delta evaluation
# =======================================================================

def generate_paths_for_freq(
    n_steps: int, device: torch.device, n_paths: int = N_TEST,
) -> dict[str, Any]:
    """Simulate rBergomi paths at the given frequency.

    Seed: ``MASTER_SEED + n_steps`` (per spec).
    """
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T,
        H=RBG_PARAMS["H"], eta=RBG_PARAMS["eta"],
        rho=RBG_PARAMS["rho"], xi0=RBG_PARAMS["xi0"],
    )
    seed = MASTER_SEED + n_steps
    S, _, _ = sim.simulate(n_paths=n_paths, S0=S0, seed=seed)

    payoff = compute_payoff(S, K, "call")
    # Use test-set mean as p0 (same convention as BS delta evaluation
    # cell in Prompt 9 where no training data is involved).
    p0 = float(payoff.mean())

    return {"S": S, "payoff": payoff, "p0": p0, "n_steps": n_steps, "seed": seed}


def _mean_turnover(deltas: Tensor) -> float:
    """Mean over paths of sum_k |delta_k - delta_{k-1}|, delta_{-1} = 0."""
    batch = deltas.shape[0]
    dtype, device = deltas.dtype, deltas.device
    delta_prev = torch.cat(
        [torch.zeros(batch, 1, dtype=dtype, device=device), deltas[:, :-1]],
        dim=1,
    )
    return float((deltas - delta_prev).abs().sum(dim=1).mean())


def evaluate_bs_delta_at_costs(
    data: dict[str, Any],
    cost_values: list[float],
    *,
    K: float = K,
    T: float = T,
    sigma: float = SIGMA_ASSUMED,
    skip_costs: set[float] | None = None,
    keep_pnl: bool = False,
) -> dict[float, dict[str, Any]]:
    """Compute BS delta ONCE, evaluate at all requested cost levels.

    Any cost in ``skip_costs`` is not computed.
    """
    skip_costs = skip_costs or set()
    S = data["S"]
    payoff = data["payoff"]
    p0 = data["p0"]

    hedger = BlackScholesDelta(sigma=sigma, K=K, T=T)
    deltas = hedger.hedge_paths(S)  # (batch, n_steps) — cost-independent
    turnover = _mean_turnover(deltas)

    out: dict[float, dict[str, Any]] = {}
    for cost in cost_values:
        if cost in skip_costs:
            continue
        pnl = compute_hedging_pnl(S, deltas, payoff, p0, cost)
        metrics = compute_all_metrics(pnl)
        cell: dict[str, Any] = {
            "metrics": metrics,
            "mean_turnover": turnover,
            "p0": p0,
            "source": "fresh",
        }
        if keep_pnl:
            cell["pnl"] = pnl.detach().float().cpu()
        out[cost] = cell

    # Drop intermediate tensors before returning
    del deltas
    return out


# =======================================================================
# Main grid driver
# =======================================================================

def run_extended_grid(
    freq_values: list[int] = FREQ_VALUES,
    cost_values: list[float] = COST_VALUES,
    reuse_prompt_9: bool = True,
    save_pnl_tensors: bool = False,
    device: torch.device | None = None,
) -> dict[int, dict[float, dict[str, Any]]]:
    """Run the full frequency x cost BS delta grid."""
    device = device or torch.device("cpu")

    prompt_9_cells: dict[int, dict[float, dict]] = {}
    if reuse_prompt_9:
        prompt_9_cells = load_prompt_9_cells(PROMPT_9_JSON)
        if prompt_9_cells:
            total_reused = sum(len(v) for v in prompt_9_cells.values())
            print(f"  Loaded {total_reused} cells from Prompt 9 Part A", flush=True)

    results: dict[int, dict[float, dict[str, Any]]] = {}

    for n_steps in freq_values:
        print(f"\n  n_steps={n_steps}", flush=True)
        t0 = time.time()

        # Pre-populate from Prompt 9 if available
        row: dict[float, dict[str, Any]] = {}
        if reuse_prompt_9 and n_steps in prompt_9_cells:
            row.update(prompt_9_cells[n_steps])
            skip = set(row.keys())
            print(f"    reusing costs {sorted(skip)}", flush=True)
        else:
            skip = set()

        costs_to_compute = [c for c in cost_values if c not in skip]
        if costs_to_compute:
            data = generate_paths_for_freq(n_steps, device=device)
            print(f"    generated {N_TEST} paths "
                  f"(seed={data['seed']}, p0={data['p0']:.4f})", flush=True)

            fresh_cells = evaluate_bs_delta_at_costs(
                data, costs_to_compute,
                skip_costs=skip,
                keep_pnl=save_pnl_tensors,
            )
            for cost, cell in fresh_cells.items():
                row[cost] = cell

            # Optionally save PnL tensors
            if save_pnl_tensors:
                for cost, cell in fresh_cells.items():
                    if "pnl" in cell:
                        fname = f"h2_ext_n{n_steps}_cost{cost:.4f}_bs_pnl.pt"
                        torch.save(cell["pnl"], FIGURE_DIR / fname)
                        # Remove from dict to avoid JSON serialisation issues
                        cell.pop("pnl", None)

            del data
            gc.collect()
        else:
            print(f"    all costs already loaded from Prompt 9", flush=True)

        # Ensure row is sorted by cost
        row_sorted = {c: row[c] for c in cost_values if c in row}
        results[n_steps] = row_sorted

        dt = time.time() - t0
        es_snapshot = " ".join(
            f"{row_sorted[c]['metrics']['es_95']:6.2f}"
            for c in cost_values
        )
        print(f"    ES_95: [{es_snapshot}]  ({dt:.1f}s)", flush=True)

    return results


# =======================================================================
# Pretty printing
# =======================================================================

def print_full_grid_table(
    results: dict[int, dict[float, dict]],
    freq_values: list[int] = FREQ_VALUES,
    cost_values: list[float] = COST_VALUES,
) -> None:
    """Print the full ES_95 grid with row-wise minima highlighted."""
    # Header
    header = "  n_steps \\ lambda   " + "".join(
        f"{c:>9.4f}" for c in cost_values
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    for n in freq_values:
        row_dict = results.get(n, {})
        # Find minimum in row
        values = [row_dict[c]["metrics"]["es_95"] for c in cost_values if c in row_dict]
        min_val = min(values) if values else None

        row_str = f"  {n:>6d}           "
        for c in cost_values:
            if c not in row_dict:
                row_str += f"{'  —  ':>9s}"
                continue
            v = row_dict[c]["metrics"]["es_95"]
            marker = "*" if min_val is not None and abs(v - min_val) < 1e-6 else " "
            row_str += f"{v:>8.3f}{marker}"
        print(row_str, flush=True)


def print_turnover_table(
    results: dict[int, dict[float, dict]],
    freq_values: list[int] = FREQ_VALUES,
    cost_values: list[float] = COST_VALUES,
) -> None:
    """Turnover depends only on frequency for BS delta."""
    print(f"\n  BS delta mean turnover (cost-independent):", flush=True)
    for n in freq_values:
        row_dict = results.get(n, {})
        turnovers = {
            c: row_dict[c]["mean_turnover"]
            for c in cost_values if c in row_dict
        }
        if turnovers:
            vals = list(turnovers.values())
            avg = sum(vals) / len(vals)
            print(f"    n={n:>4d}:  turnover = {avg:.3f}", flush=True)


# =======================================================================
# Reversal detection
# =======================================================================

def detect_reversal(
    results: dict[int, dict[float, dict]],
    freq_values: list[int] = FREQ_VALUES,
    cost_values: list[float] = COST_VALUES,
    tol: float = 0.01,
) -> dict[str, Any]:
    """Find the ES_95-minimising n_steps at each cost level.

    A reversal is declared at cost level c if the optimal n_steps is
    strictly less than the largest tested n_steps AND the ES_95 at the
    largest n_steps exceeds the minimum by at least ``tol``.

    Returns a dict with diagnostic fields and a human-readable verdict.
    """
    min_freq_by_cost: dict[float, int] = {}
    reversal_detected: dict[float, bool] = {}
    saturation: dict[float, bool] = {}

    largest_n = max(freq_values)
    smallest_n = min(freq_values)

    reversal_costs: list[float] = []

    for c in cost_values:
        es_by_n = {
            n: results[n][c]["metrics"]["es_95"]
            for n in freq_values if c in results.get(n, {})
        }
        if not es_by_n:
            continue

        min_n = min(es_by_n, key=lambda n: es_by_n[n])
        min_val = es_by_n[min_n]
        es_at_largest = es_by_n[largest_n]

        min_freq_by_cost[c] = min_n

        is_reversal = (
            min_n < largest_n and (es_at_largest - min_val) > tol
        )
        reversal_detected[c] = is_reversal
        if is_reversal:
            reversal_costs.append(c)

        # Saturation: even if the largest n is the minimiser, check if
        # the marginal improvement between the second-largest and largest
        # is <= tol.
        sorted_n = sorted(es_by_n.keys())
        if len(sorted_n) >= 2:
            diff_last = es_by_n[sorted_n[-2]] - es_by_n[sorted_n[-1]]
            saturation[c] = (not is_reversal) and (diff_last < tol)
        else:
            saturation[c] = False

    reversal_threshold: float | None = (
        min(reversal_costs) if reversal_costs else None
    )

    # Verdict
    if reversal_threshold is not None:
        verdict = "strong H2"
        summary = (
            f"Strong H2 confirmed: reversal observed starting at "
            f"lambda = {reversal_threshold:.4f}"
        )
    elif any(saturation.values()):
        verdict = "moderate H2"
        summary = (
            "Moderate H2 (saturation): no explicit reversal, but marginal "
            "frequency benefit vanishes at high cost"
        )
    else:
        verdict = "weak H2"
        summary = (
            "Weak H2: ES_95 decreases monotonically with frequency at "
            "all tested cost levels"
        )

    return {
        "min_freq_by_cost": min_freq_by_cost,
        "reversal_detected": reversal_detected,
        "saturation": saturation,
        "reversal_cost_threshold": reversal_threshold,
        "verdict": verdict,
        "summary": summary,
    }


# =======================================================================
# Persistence
# =======================================================================

def _strip_for_json(obj: Any) -> Any:
    """Recursively convert to JSON-safe form."""
    if isinstance(obj, torch.Tensor):
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


def save_results(
    results: dict[int, dict[float, dict]],
    detection: dict[str, Any],
    out_path: Path = OUT_JSON,
) -> None:
    """Save scalar results and detection analysis to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "freq_values": FREQ_VALUES,
            "cost_values": COST_VALUES,
            "n_test": N_TEST,
            "rbergomi_params": RBG_PARAMS,
            "S0": S0, "K": K, "T": T,
            "sigma_assumed": SIGMA_ASSUMED,
            "master_seed": MASTER_SEED,
        },
        "grid": _strip_for_json(results),
        "detection": _strip_for_json(detection),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {out_path}", flush=True)


# =======================================================================
# Main
# =======================================================================

def main() -> None:
    device = get_device()
    set_global_seed(MASTER_SEED)

    print("=" * 70, flush=True)
    print("  H2 GRID EXTENSION  —  BS delta only", flush=True)
    print("=" * 70, flush=True)
    print(f"  Frequencies: {FREQ_VALUES}", flush=True)
    print(f"  Costs:       {COST_VALUES}", flush=True)
    print(f"  Grid size:   {len(FREQ_VALUES)} x {len(COST_VALUES)} = "
          f"{len(FREQ_VALUES) * len(COST_VALUES)} cells", flush=True)
    print(f"  N_test:      {N_TEST}", flush=True)
    print(f"  Device:      {device}", flush=True)

    # ── Pre-validate n_steps=800 ──
    print("\n  Pre-validating n_steps=800 (100 paths, dry run) ...", flush=True)
    try:
        sim_pre = DifferentiableRoughBergomi(
            n_steps=800, T=T,
            H=RBG_PARAMS["H"], eta=RBG_PARAMS["eta"],
            rho=RBG_PARAMS["rho"], xi0=RBG_PARAMS["xi0"],
        )
        S_pre, _, _ = sim_pre.simulate(n_paths=100, S0=S0, seed=42)
        if torch.isnan(S_pre).any() or torch.isinf(S_pre).any():
            raise RuntimeError("n_steps=800 produced NaN/Inf")
        print(f"    OK: shape {tuple(S_pre.shape)}, "
              f"finite={bool(torch.isfinite(S_pre).all())}", flush=True)
        del sim_pre, S_pre
        gc.collect()
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)
        raise

    # ── Run the grid ──
    t0 = time.time()
    results = run_extended_grid(device=device)
    elapsed = time.time() - t0
    print(f"\n  Total grid time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # ── Pretty-print ──
    print("\n" + "=" * 70, flush=True)
    print("  FULL 6x7 GRID  —  ES_95 (row-minimum marked with *)", flush=True)
    print("=" * 70, flush=True)
    print_full_grid_table(results)

    print_turnover_table(results)

    # ── Reversal detection ──
    print("\n" + "=" * 70, flush=True)
    print("  REVERSAL DETECTION", flush=True)
    print("=" * 70, flush=True)
    detection = detect_reversal(results)

    print(f"  Optimal n_steps by cost level:", flush=True)
    for cost in COST_VALUES:
        if cost not in detection["min_freq_by_cost"]:
            continue
        opt_n = detection["min_freq_by_cost"][cost]
        is_rev = detection["reversal_detected"][cost]
        is_sat = detection["saturation"][cost]
        tag = ""
        if is_rev:
            tag = "  [REVERSAL]"
        elif is_sat:
            tag = "  [SATURATED]"
        print(f"    lambda = {cost:.4f}  ->  n_steps = {opt_n}{tag}", flush=True)

    print(f"\n  VERDICT: {detection['verdict']}", flush=True)
    print(f"  {detection['summary']}", flush=True)

    if detection["reversal_cost_threshold"] is not None:
        print(f"  Reversal threshold: lambda >= "
              f"{detection['reversal_cost_threshold']:.4f}", flush=True)

    # ── Save ──
    save_results(results, detection, OUT_JSON)
    print("\n  Done.", flush=True)


if __name__ == "__main__":
    main()
