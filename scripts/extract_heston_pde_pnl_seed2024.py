"""Extract per-path P&L for True Heston PDE delta on canonical rB seed-2024 paths.

Output: ``results/canonical_v2/heston_pde_pnl_seed2024.npy`` — 50,000 float64
P&L values, one per test path.

This is needed for Section 6.3.1 figure regeneration so the histograms /
Q-Q plots / metrics bars use the True Heston PDE Delta (not the Plug-in
Delta currently shown in the canonical_v2 figures).

The path generation, train/val/test split, and ``p0`` Monte Carlo premium
are all replicated exactly as in
``deep_hedging/experiments/run_section_6_3_baseline.py``
``Section63Experiment.generate_data(seed=2024)``.

Run from repo root::

    python scripts/extract_heston_pde_pnl_seed2024.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_hedging.core.rough_bergomi import DifferentiableRoughBergomi
from deep_hedging.hedging.heston_pde_delta import HestonPDEDelta
from deep_hedging.objectives.pnl import compute_payoff, compute_hedging_pnl
from deep_hedging.objectives.risk_measures import compute_all_metrics

def main() -> None:
    # Canonical rough Bergomi parameters
    H = 0.07
    eta = 1.9
    rho = -0.7
    xi0 = 0.235 ** 2  # 0.055225
    T = 1.0
    n_steps = 100
    S0 = 100.0
    K = 100.0
    seed = 2024

    n_train = 80_000
    n_val = 20_000
    n_test = 50_000
    total = n_train + n_val + n_test

    # ---- Step 1: generate canonical rB paths exactly as Section63Experiment ----
    print(f"Generating {total:,} rough Bergomi paths "
          f"(H={H}, eta={eta}, rho={rho}, seed={seed}) ...", flush=True)
    sim = DifferentiableRoughBergomi(
        n_steps=n_steps, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
    )
    S_all, V_all, _ = sim.simulate(n_paths=total, S0=S0, seed=seed)

    n1 = n_train
    n2 = n_train + n_val
    S_train = S_all[:n1]
    S_test = S_all[n2:]
    V_test = V_all[n2:]

    print(f"  S_train shape={tuple(S_train.shape)}", flush=True)
    print(f"  S_test  shape={tuple(S_test.shape)}", flush=True)

    # p0 from training-set MC
    payoff_train = compute_payoff(S_train, K, "call")
    p0 = float(payoff_train.mean())
    print(f"  p0 (MC option price on training set) = {p0:.10f}", flush=True)
    print(f"  Expected from baseline_seed2024_full.json: 8.0553676232", flush=True)

    # ---- Step 2: load Heston calibration ----
    calib_path = REPO_ROOT / "results" / "heston_pde" / "calibration_data.json"
    with open(calib_path) as f:
        calib = json.load(f)
    hp = calib["heston_params"]
    print(f"\nHeston calibration:", flush=True)
    for k in ["kappa", "theta", "sigma_v", "rho", "V0"]:
        print(f"  {k} = {hp[k]}", flush=True)

    # ---- Step 3: solve Heston PDE ----
    print("\nSolving Heston PDE (this may take 1–2 minutes)...", flush=True)
    heston_pde = HestonPDEDelta(
        kappa=hp["kappa"],
        theta=hp["theta"],
        sigma_v=hp["sigma_v"],
        rho=hp["rho"],
        V0=hp["V0"],
        K=K,
        T=T,
    )
    print("  PDE solved.", flush=True)

    # ---- Step 4: compute deltas on test paths ----
    print(f"\nEvaluating Heston PDE delta on {n_test:,} test paths...", flush=True)
    deltas = heston_pde.hedge_paths(S_test, V_test)
    print(f"  deltas shape={tuple(deltas.shape)}", flush=True)

    # ---- Step 5: compute per-path P&L ----
    payoff_test = compute_payoff(S_test, K, "call")
    pnl = compute_hedging_pnl(
        S=S_test, deltas=deltas, payoff=payoff_test, p0=p0, cost_lambda=0.0,
    )
    pnl_np = pnl.detach().cpu().numpy().astype(np.float64)
    print(f"  pnl shape={pnl_np.shape}, mean={pnl_np.mean():.6f}, "
          f"std={pnl_np.std():.6f}", flush=True)

    # ---- Step 6: verify summary metrics ----
    metrics = compute_all_metrics(pnl)
    print("\nSummary metrics on seed-2024 test set:", flush=True)
    for k in ["es_95", "es_99", "var_95", "std_pnl", "mean_pnl",
              "skewness", "kurtosis"]:
        if k in metrics:
            print(f"  {k}: {float(metrics[k]):.6f}", flush=True)

    # Loss-based ES_0.95 (positive = loss)
    loss = -pnl_np
    es_95_from_array = float(loss[loss >= np.quantile(loss, 0.95)].mean())
    print(f"\n  ES_0.95 from per-path array: {es_95_from_array:.4f}", flush=True)
    print(f"  Compare seed-6024 canonical Heston PDE: 13.5236", flush=True)
    print(f"  Compare 5-seed mean Heston PDE: 13.4470 ± 0.0857", flush=True)

    # ---- Step 7: save ----
    output_dir = REPO_ROOT / "results" / "canonical_v2"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / "heston_pde_pnl_seed2024.npy"
    np.save(out_path, pnl_np)
    print(f"\nSaved {len(pnl_np):,} per-path P&L values "
          f"({pnl_np.nbytes / 1024:.1f} KB) to {out_path}", flush=True)

if __name__ == "__main__":
    main()
