"""Compute Kendall τ between rebalancing frequency n and ES_0.95 at each
proportional-cost level λ from the H2 grid extension.

Source (resolved at runtime, in priority order):
    1. figures/h2_grid_extension.json   (live local recompute path)
    2. archive/legacy_figures_data/h2_grid_extension.json
       (the post-Phase-2 archived copy used for Tab. 9 reproduction)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parent.parent

_CANDIDATE_PATHS = [
    REPO_ROOT / "figures" / "h2_grid_extension.json",
    REPO_ROOT / "archive" / "legacy_figures_data" / "h2_grid_extension.json",
]
_h2_path = next((p for p in _CANDIDATE_PATHS if p.exists()), None)
if _h2_path is None:
    raise SystemExit(
        "h2_grid_extension.json not found in either:\n"
        + "\n".join(f"  - {p}" for p in _CANDIDATE_PATHS)
        + "\nTo regenerate, run:\n"
        "  python -m deep_hedging.experiments.h2_grid_extension"
    )
H2 = json.load(open(_h2_path))

# At each λ, walk through all n values; collect ES_0.95 for the BS strategy
# (the H2 reversal is a delta-strategy phenomenon under proportional cost).

ns = sorted(int(k) for k in H2["grid"].keys())
all_lams = sorted({float(k) for n in H2["grid"].values() for k in n.keys()})

print(f"n values: {ns}")
print(f"λ values: {all_lams}")
print()
print(f"{'λ':>10s}  {'τ_BS':>8s}  {'τ_Leland':>10s}  {'n*_BS':>6s}  {'n*_Leland':>10s}")
for lam in all_lams:
    bs_pairs = []
    ld_pairs = []
    for n in ns:
        cells = H2["grid"][str(n)]
        # Find matching key
        key = None
        for k in cells:
            if abs(float(k) - lam) < 1e-9:
                key = k
                break
        if key is None:
            continue
        cell = cells[key]
        bs_es = (cell.get("BS") or {}).get("metrics", {}).get("es_95")
        ld_es = (cell.get("Leland") or {}).get("metrics", {}).get("es_95")
        if bs_es is not None:
            bs_pairs.append((n, bs_es))
        if ld_es is not None:
            ld_pairs.append((n, ld_es))

    if len(bs_pairs) >= 2:
        ns_bs = np.array([p[0] for p in bs_pairs])
        es_bs = np.array([p[1] for p in bs_pairs])
        tau_bs, _ = kendalltau(ns_bs, es_bs)
        n_star_bs = ns_bs[np.argmin(es_bs)]
    else:
        tau_bs = np.nan
        n_star_bs = -1

    if len(ld_pairs) >= 2:
        ns_ld = np.array([p[0] for p in ld_pairs])
        es_ld = np.array([p[1] for p in ld_pairs])
        tau_ld, _ = kendalltau(ns_ld, es_ld)
        n_star_ld = ns_ld[np.argmin(es_ld)]
    else:
        tau_ld = np.nan
        n_star_ld = -1

    print(f"{lam:>10.4f}  {tau_bs:>+8.3f}  {tau_ld:>+10.3f}  "
          f"{n_star_bs:>6d}  {n_star_ld:>10d}")

# Also print full BS grid to see the reversal
print()
print("BS ES_0.95 grid:")
print(f"{'n':>6s}  " + "  ".join(f"{lam:>9.4f}" for lam in all_lams))
for n in ns:
    cells = H2["grid"][str(n)]
    row = [f"{n:>6d}"]
    for lam in all_lams:
        key = None
        for k in cells:
            if abs(float(k) - lam) < 1e-9:
                key = k
                break
        if key is None:
            row.append("    ---  ")
            continue
        cell = cells[key]
        bs_es = (cell.get("BS") or {}).get("metrics", {}).get("es_95")
        if bs_es is None:
            row.append("    ---  ")
        else:
            row.append(f"{bs_es:>9.4f}")
    print("  ".join(row))
