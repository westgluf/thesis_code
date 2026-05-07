"""Generate Figure 5.V (markovian benchmark sample paths) for Section 5.2.

Three panels:
  (a) 50 GBM sample paths at sigma=0.235
  (b) 50 Heston sample paths (S only)
  (c) 50 Heston variance paths (V only)

Run from repo root:
    python scripts/generate_5_2_figure.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from deep_hedging.core.gbm import GBM
from deep_hedging.core.heston import Heston

# Canonical parameters
T = 1.0
n_steps = 100
n_paths = 50
S0 = 100.0
sigma_gbm = 0.235

# GBM
gbm = GBM(n_steps=n_steps, T=T, sigma=sigma_gbm)
S_gbm, _, t_grid = gbm.simulate(n_paths=n_paths, S0=S0, seed=2024)

# Heston with calibrated params
calib_path = os.path.join(REPO_ROOT, "results", "heston_pde", "calibration_data.json")
with open(calib_path) as f:
    calib = json.load(f)
hp = calib["heston_params"]
heston = Heston(
    n_steps=n_steps, T=T,
    v0=hp["V0"], kappa=hp["kappa"], theta=hp["theta"],
    sigma_v=hp["sigma_v"], rho=hp["rho"],
)
S_heston, V_heston, _ = heston.simulate(n_paths=n_paths, S0=S0, seed=2024)

# Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
t_np = t_grid.cpu().numpy()

# Panel (a): GBM
axes[0].plot(t_np, S_gbm.cpu().numpy().T, lw=0.5, alpha=0.5, color="steelblue")
axes[0].axhline(S0, color="black", lw=0.7, ls="--", alpha=0.4)
axes[0].set_title("(a) GBM sample paths ($\\sigma = 0.235$)")
axes[0].set_xlabel("$t$")
axes[0].set_ylabel("$S_t$")
axes[0].grid(alpha=0.3)

# Panel (b): Heston S
axes[1].plot(t_np, S_heston.cpu().numpy().T, lw=0.5, alpha=0.5, color="darkgreen")
axes[1].axhline(S0, color="black", lw=0.7, ls="--", alpha=0.4)
axes[1].set_title("(b) Heston sample paths ($S_t$)")
axes[1].set_xlabel("$t$")
axes[1].set_ylabel("$S_t$")
axes[1].set_ylim(axes[0].get_ylim())  # match y-axis with GBM
axes[1].grid(alpha=0.3)

# Panel (c): Heston V
axes[2].plot(t_np, V_heston.cpu().numpy().T, lw=0.5, alpha=0.5, color="firebrick")
axes[2].axhline(hp["V0"], color="black", lw=0.7, ls="--", alpha=0.4,
                label=f"$V_0 = {hp['V0']:.4f}$")
axes[2].set_title("(c) Heston variance paths ($V_t$)")
axes[2].set_xlabel("$t$")
axes[2].set_ylabel("$V_t$")
axes[2].legend(loc="upper right", fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()

out_latex = os.path.join(REPO_ROOT, "latex_package", "figures", "5_2_markovian_paths.png")
out_repo = os.path.join(REPO_ROOT, "figures", "5_2_markovian_paths.png")
os.makedirs(os.path.dirname(out_latex), exist_ok=True)
os.makedirs(os.path.dirname(out_repo), exist_ok=True)
plt.savefig(out_latex, dpi=170, bbox_inches="tight")
plt.savefig(out_repo, dpi=170, bbox_inches="tight")
print(f"Saved: {out_latex}")
print(f"        {out_repo}")
