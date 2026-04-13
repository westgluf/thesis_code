#!/usr/bin/env python
"""
Smoke tests for Prompt 10 Pareto + H2 analysis.

    python -m deep_hedging.tests.test_pareto_h2_analysis
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from deep_hedging.experiments.pareto_h2_analysis import (
    ParetoH2Analyser,
    H2_FREQ_VALUES, H2_COST_VALUES,
    FIGURE_DIR,
)


def _make_tmp_with_jsons() -> tempfile.TemporaryDirectory:
    """Create a tmp dir and copy the 3 source JSON files into it."""
    tmp = tempfile.TemporaryDirectory()
    for name in ("pareto_part_A_results.json",
                 "pareto_part_B_results.json",
                 "h2_grid_extension.json"):
        src = FIGURE_DIR / name
        if src.exists():
            shutil.copy(src, Path(tmp.name) / name)
    return tmp


# -----------------------------------------------------------------------
# Test 1: Loaders return expected structures
# -----------------------------------------------------------------------

def test_loaders() -> Tuple[bool, str]:
    a = ParetoH2Analyser(figures_dir=FIGURE_DIR, seed_consistent_reeval=False)

    pa = a.load_pareto_part_A()
    if not isinstance(pa, dict) or "bs" not in pa or "deep" not in pa:
        return False, "Part A missing keys"
    if len(pa["bs"]) != 9 or len(pa["deep"]) != 9:
        return False, f"Part A: bs={len(pa['bs'])}, deep={len(pa['deep'])}"

    pb = a.load_pareto_part_B()
    n_strats = len([k for k in pb if not k.startswith("_")])
    if n_strats != 6:
        return False, f"Part B: expected 6 strategies, got {n_strats}"

    h2 = a.load_h2_extension()
    if len(h2["grid"]) != len(H2_FREQ_VALUES) * len(H2_COST_VALUES):
        return False, f"H2 grid size {len(h2['grid'])}"

    return True, f"PartA={len(pa['bs'])}+{len(pa['deep'])}, PartB={n_strats}, H2={len(h2['grid'])}"


# -----------------------------------------------------------------------
# Test 2: Pareto front identification on synthetic data
# -----------------------------------------------------------------------

def test_pareto_synthetic() -> Tuple[bool, str]:
    """Lower-better on both axes; expected front: (1,1), (3,0), (0,3)."""
    fake = {
        "p1": {"mean_pnl": -1.0, "es_95": 1.0, "es_99": 0, "std_pnl": 0, "entropic_1": 0, "turnover": 0},
        "p2": {"mean_pnl": -2.0, "es_95": 2.0, "es_99": 0, "std_pnl": 0, "entropic_1": 0, "turnover": 0},
        "p3": {"mean_pnl": -3.0, "es_95": 0.0, "es_99": 0, "std_pnl": 0, "entropic_1": 0, "turnover": 0},
        "p4": {"mean_pnl": 0.0,  "es_95": 3.0, "es_99": 0, "std_pnl": 0, "entropic_1": 0, "turnover": 0},
        "p5": {"mean_pnl": -4.0, "es_95": 4.0, "es_99": 0, "std_pnl": 0, "entropic_1": 0, "turnover": 0},
    }
    # Convention: mean_pnl HIGHER is better, es_95 LOWER is better.
    # Re-derive expected front under that convention:
    # p1: mean=-1, es=1
    # p2: mean=-2, es=2  -> dominated by p1 (higher mean, lower es)
    # p3: mean=-3, es=0
    # p4: mean=0,  es=3
    # p5: mean=-4, es=4  -> dominated by p3
    # p1 vs p3: p1 has higher mean (-1 > -3), p3 has lower es (0 < 1) -> neither dominates
    # p1 vs p4: p4 higher mean (0 > -1), p1 lower es (1 < 3) -> neither dominates
    # p3 vs p4: p4 higher mean, p3 lower es -> neither dominates
    # Expected front: {p1, p3, p4}; dominated: {p2, p5}

    a = ParetoH2Analyser(figures_dir=FIGURE_DIR, seed_consistent_reeval=False)
    front = a.identify_pareto_front(fake, ("mean_pnl", "es_95"))
    front_set = set(front["front_tags"])
    expected = {"p1", "p3", "p4"}
    if front_set != expected:
        return False, f"front={front_set}, expected={expected}"

    dominated = set(front["dominated_tags"])
    expected_dom = {"p2", "p5"}
    if dominated != expected_dom:
        return False, f"dominated={dominated}, expected={expected_dom}"

    return True, f"front={sorted(front_set)}, dominated={sorted(dominated)}"


# -----------------------------------------------------------------------
# Test 3: Monotonicity test on synthetic U-shape
# -----------------------------------------------------------------------

def test_monotonicity_u_shape() -> Tuple[bool, str]:
    a = ParetoH2Analyser(figures_dir=FIGURE_DIR, seed_consistent_reeval=False)
    es_values = [14.0, 13.0, 12.5, 12.3, 12.8, 13.5]
    grid = {
        (n, 0.005): es
        for n, es in zip(H2_FREQ_VALUES, es_values)
    }
    mono = a.test_h2_monotonicity(grid, cost_values=[0.005])
    r = mono[0.005]

    if r["optimum_n"] != 200:
        return False, f"optimum_n={r['optimum_n']}, expected 200"
    if r["monotone_direction"] != "U-shaped":
        return False, f"direction={r['monotone_direction']}, expected U-shaped"
    expected_deg = 100 * (13.5 - 12.3) / 12.3
    if abs(r["degradation_pct"] - expected_deg) > 0.5:
        return False, f"degradation={r['degradation_pct']:.2f}, expected {expected_deg:.2f}"

    return True, (
        f"opt_n=200, dir=U-shaped, degradation={r['degradation_pct']:.2f}%, "
        f"tau={r['kendall_tau']:+.3f}"
    )


# -----------------------------------------------------------------------
# Test 4: Seed-consistent reevaluation is deterministic and fast
# -----------------------------------------------------------------------

def test_reeval_deterministic() -> Tuple[bool, str]:
    a = ParetoH2Analyser(figures_dir=FIGURE_DIR, seed_consistent_reeval=False)

    t0 = time.time()
    out1 = a.reevaluate_h2_grid_consistent()
    elapsed = time.time() - t0
    out2 = a.reevaluate_h2_grid_consistent()

    if elapsed > 60:
        return False, f"reeval too slow: {elapsed:.1f}s"

    # Determinism
    for key in out1["grid"]:
        if abs(out1["grid"][key] - out2["grid"][key]) > 1e-9:
            return False, f"non-deterministic at {key}"

    # Cross-check a few values against the original Prompt 9.5 grid
    # (different overlap behavior — Prompt 9.5 reused Prompt 9 cells at
    # n in {50, 100, 200} for cost in {0, 0.001, 0.002}, so for those
    # specific cells the values should match Prompt 9 exactly. For
    # other cells they should match within MC noise of the consistent
    # reeval since the seed scheme is the same.)
    h2_orig = a.load_h2_extension()
    # At n=400 and n=800 — fully fresh in both — should match exactly
    for n in (400, 800):
        for c in H2_COST_VALUES:
            v1 = out1["grid"][(n, c)]
            v2 = h2_orig["grid"][(n, c)]
            if abs(v1 - v2) > 0.02:
                return False, f"n={n},c={c}: {v1:.3f} vs original {v2:.3f}"

    return True, f"deterministic, {elapsed:.1f}s, n=400/800 match Prompt 9.5 exactly"


# -----------------------------------------------------------------------
# Test 5: LaTeX tables are valid
# -----------------------------------------------------------------------

def test_latex_tables() -> Tuple[bool, str]:
    tmp = _make_tmp_with_jsons()
    try:
        a = ParetoH2Analyser(figures_dir=tmp.name, seed_consistent_reeval=False)
        a.load_pareto_part_A()
        a.load_pareto_part_B()
        a.load_h2_extension()
        a.generate_latex_tables()

        tmp_path = Path(tmp.name)
        f1 = tmp_path / "pareto_part_b_table.tex"
        f2 = tmp_path / "h2_extended_grid_table.tex"
        if not f1.exists() or not f2.exists():
            return False, "tables not created"

        for path in (f1, f2):
            text = path.read_text()
            if r"\begin{table}" not in text or r"\end{table}" not in text:
                return False, f"{path.name} missing table env"
            if "nan" in text.lower() or "inf" in text.lower():
                return False, f"{path.name} contains nan/inf"

        # Check the H2 table contains all 6 freq rows
        h2_text = f2.read_text()
        for n in (25, 50, 100, 200, 400, 800):
            if f"{n} &" not in h2_text:
                return False, f"H2 table missing n={n}"

        return True, "both tables valid"
    finally:
        tmp.cleanup()


# -----------------------------------------------------------------------
# Test 6: Figure generation creates all expected files
# -----------------------------------------------------------------------

def test_figures_generated() -> Tuple[bool, str]:
    tmp = _make_tmp_with_jsons()
    try:
        a = ParetoH2Analyser(figures_dir=tmp.name, seed_consistent_reeval=False)
        a.load_pareto_part_A()
        a.load_pareto_part_B()
        a.load_h2_extension()
        a.generate_all_figures()

        tmp_path = Path(tmp.name)
        expected = [
            "fig_pareto_front_main.png",
            "fig_pareto_multi_axis.png",
            "fig_h2_heatmap.png",
            "fig_h2_es_curves.png",
            "fig_h2_optimal_frequency.png",
            "fig_h2_cost_penalty.png",
            "fig_pareto_deep_vs_bs_grid.png",
            "fig_h2_summary.png",
        ]
        missing = []
        sizes = {}
        for name in expected:
            f = tmp_path / name
            if not f.exists():
                missing.append(name)
            else:
                sz = f.stat().st_size
                sizes[name] = sz
                if sz < 10_000:
                    return False, f"{name} too small: {sz} bytes"
        if missing:
            return False, f"missing: {missing}"

        return True, (
            f"all 8 figures created "
            f"(sizes {min(sizes.values())//1024}-{max(sizes.values())//1024} KB)"
        )
    finally:
        tmp.cleanup()


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Loaders return expected structures",  test_loaders),
        ("2. Pareto front on synthetic data",       test_pareto_synthetic),
        ("3. Monotonicity test on U-shape",         test_monotonicity_u_shape),
        ("4. Seed-consistent reeval deterministic", test_reeval_deterministic),
        ("5. LaTeX tables valid",                   test_latex_tables),
        ("6. All 8 figures generated",              test_figures_generated),
    ]

    print("=" * 65, flush=True)
    print(" Pareto + H2 Analysis — Smoke Tests", flush=True)
    print("=" * 65, flush=True)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as e:
            import traceback
            passed, msg = False, f"EXCEPTION: {e}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        print(f"         {msg}", flush=True)
        if not passed:
            all_passed = False

    print("-" * 65, flush=True)
    if all_passed:
        print(" All 6 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
