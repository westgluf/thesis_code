"""Compute the standard-deviation ratio of DH ES_0.95 across the η axis
versus the H axis in the M.2 axis sweep.

Source: results/perturbation_v2/M2_axis_sweep.json

Output: prints per-axis (mean, std) of DH ES_0.95 across the 15 grid
points along each axis, plus the std ratio η / H.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
M2 = json.load(open(REPO_ROOT / "results" / "perturbation_v2" / "M2_axis_sweep.json"))

def collect_axis(axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis_values, dh_es95_means) for the 15 points along the axis."""
    axis_data = M2["results"][axis]
    keys = sorted(float(k) for k in axis_data.keys())
    dh_es = []
    bs_es = []
    for v in keys:
        # Find matching key (string format may differ slightly)
        cell = None
        for k, val in axis_data.items():
            if abs(float(k) - v) < 1e-9:
                cell = val
                break
        agg = cell.get("aggregate", cell)
        dh_es.append(agg["dh_es95"]["mean"])
        bs_es.append(agg["bs_es95"]["mean"])
    return np.array(keys), np.array(dh_es), np.array(bs_es)

for axis in ("H", "eta", "rho"):
    keys, dh, bs = collect_axis(axis)
    print(f"\n=== Axis: {axis} (15 grid points) ===")
    print(f"  values:    {keys}")
    print(f"  DH ES_95:  min={dh.min():.4f} max={dh.max():.4f} std={dh.std(ddof=1):.4f}")
    print(f"  BS ES_95:  min={bs.min():.4f} max={bs.max():.4f} std={bs.std(ddof=1):.4f}")
    print(f"  range_DH:  {dh.max() - dh.min():.4f}")

# Ratio η / H
_, dh_H, _ = collect_axis("H")
_, dh_eta, _ = collect_axis("eta")
_, dh_rho, _ = collect_axis("rho")

std_H = dh_H.std(ddof=1)
std_eta = dh_eta.std(ddof=1)
std_rho = dh_rho.std(ddof=1)

print("\n=== Cross-axis comparison (DH ES_0.95 std across 15 grid points) ===")
print(f"  std along η: {std_eta:.4f}")
print(f"  std along H: {std_H:.4f}")
print(f"  std along ρ: {std_rho:.4f}")
print(f"  ratio η / H: {std_eta / std_H:.2f}")
print(f"  ratio η / ρ: {std_eta / std_rho:.2f}")
print(f"  ratio H  / ρ: {std_H / std_rho:.2f}")

range_H = dh_H.max() - dh_H.min()
range_eta = dh_eta.max() - dh_eta.min()
range_rho = dh_rho.max() - dh_rho.min()
print(f"\n  range η / H: {range_eta / range_H:.2f}")
print(f"  range η / ρ: {range_eta / range_rho:.2f}")
