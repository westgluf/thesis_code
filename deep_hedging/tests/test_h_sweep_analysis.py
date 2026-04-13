#!/usr/bin/env python
"""
Quick tests for the H-sweep analysis module.

    python -m deep_hedging.tests.test_h_sweep_analysis
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from deep_hedging.experiments.h_sweep_analysis import HSweepAnalyser

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS_PATH = FIGURE_DIR / "h_sweep_results.json"


def _have_results() -> bool:
    return RESULTS_PATH.exists()


# ---------------------------------------------------------------------------
# Test 1: Load results — shapes, no NaN
# ---------------------------------------------------------------------------

def test_load_results() -> Tuple[bool, str]:
    """Load JSON, check arrays match expected number of H values."""
    if not _have_results():
        return False, f"Missing {RESULTS_PATH}"
    a = HSweepAnalyser(RESULTS_PATH)
    n = len(a.H_values)
    sorted_ok = all(a.H_values[i] <= a.H_values[i + 1] for i in range(n - 1))
    nan_ok = (np.isfinite(a.gamma).all() and np.isfinite(a.es95_bs).all()
              and np.isfinite(a.es95_dh).all())
    passed = n >= 5 and sorted_ok and nan_ok
    return passed, f"n={n}, sorted={sorted_ok}, no_nan={nan_ok}"


# ---------------------------------------------------------------------------
# Test 2: Power-law fit on synthetic data
# ---------------------------------------------------------------------------

def test_synthetic_fit() -> Tuple[bool, str]:
    """Fit known β=1.5, c=3.0 + small noise → recover parameters."""
    rng = np.random.default_rng(123)
    H_syn = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    true_beta, true_c = 1.5, 3.0
    gamma_syn = true_c * (0.5 - H_syn) ** true_beta * (1 + 0.02 * rng.standard_normal(len(H_syn)))

    # Build a fake JSON
    records = []
    for h, g in zip(H_syn, gamma_syn):
        records.append({
            "H": float(h), "gamma": float(g), "p0": 8.0,
            "training_time_s": 1.0, "best_epoch": 10, "best_val_risk": 5.0,
            "bs_metrics": {"es_95": float(g + 5), "es_99": float(g + 10),
                           "std_pnl": 4.0, "mean_pnl": 0.0, "var_95": 5.0,
                           "entropic_1": 10.0, "max_loss": 20.0, "min_pnl": -20.0,
                           "skewness": -1.0, "kurtosis": 5.0},
            "dh_metrics": {"es_95": 5.0, "es_99": 10.0, "std_pnl": 3.5,
                           "mean_pnl": 0.0, "var_95": 4.0, "entropic_1": 8.0,
                           "max_loss": 15.0, "min_pnl": -15.0,
                           "skewness": -0.5, "kurtosis": 4.0},
        })

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        tmp = Path(f.name)

    a = HSweepAnalyser(tmp)
    fit = a.fit_power_law(exclude_h05=True)
    tmp.unlink()

    beta_ok = abs(fit["beta"] - true_beta) < 0.3
    c_ok = abs(fit["c"] - true_c) < 1.0
    r2_ok = fit["r_squared"] > 0.9

    passed = beta_ok and c_ok and r2_ok
    return passed, (
        f"beta={fit['beta']:.3f} (target {true_beta}), "
        f"c={fit['c']:.3f} (target {true_c}), "
        f"R2={fit['r_squared']:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Power-law fit on actual data
# ---------------------------------------------------------------------------

def test_actual_fit() -> Tuple[bool, str]:
    """Fit on real sweep data — β should be finite."""
    if not _have_results():
        return False, f"Missing {RESULTS_PATH}"
    a = HSweepAnalyser(RESULTS_PATH)
    fit = a.fit_power_law()
    ok = math.isfinite(fit["beta"]) and math.isfinite(fit["r_squared"])
    passed = ok and fit["beta"] > -5.0
    return passed, (
        f"beta={fit['beta']:.3f}, c={fit['c']:.3f}, "
        f"R2={fit['r_squared']:.3f}, n={fit['n_points']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Bootstrap CIs bracket point estimate
# ---------------------------------------------------------------------------

def test_bootstrap() -> Tuple[bool, str]:
    """CI should contain the point estimate."""
    if not _have_results():
        return False, f"Missing {RESULTS_PATH}"
    a = HSweepAnalyser(RESULTS_PATH)
    fit = a.fit_power_law()
    boot = a.bootstrap_confidence(n_bootstrap=200, seed=42)
    lo, hi = boot["beta_ci"]
    bracket = lo <= fit["beta"] <= hi
    passed = bracket and math.isfinite(lo)
    return passed, f"beta={fit['beta']:.3f}, CI=[{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# Test 5: Phase transition test returns valid dict
# ---------------------------------------------------------------------------

def test_phase_transition() -> Tuple[bool, str]:
    """Phase transition test produces expected keys."""
    if not _have_results():
        return False, f"Missing {RESULTS_PATH}"
    a = HSweepAnalyser(RESULTS_PATH)
    ph = a.test_phase_transition()
    keys_ok = {"break_point", "rss_single", "rss_piecewise",
               "improvement", "significant"}.issubset(ph.keys())
    finite_ok = math.isfinite(ph["rss_single"])
    passed = keys_ok and finite_ok
    return passed, (
        f"break_H={ph['break_point']:.2f}, improvement={ph['improvement']:.1%}, "
        f"significant={ph['significant']}"
    )


# ---------------------------------------------------------------------------
# Test 6: All figures generate
# ---------------------------------------------------------------------------

def test_figures_generate() -> Tuple[bool, str]:
    """All analysis figures created without error."""
    if not _have_results():
        return False, f"Missing {RESULTS_PATH}"
    a = HSweepAnalyser(RESULTS_PATH)
    a.generate_all_figures(FIGURE_DIR)

    expected = [
        "fig_h_sweep_es95_es99.png",
        "fig_h_sweep_gamma_loglog.png",
        "fig_h_sweep_relative_gap.png",
        "fig_h_sweep_phase_transition.png",
        "fig_h_sweep_summary.png",
    ]
    missing = [f for f in expected if not (FIGURE_DIR / f).exists()]
    passed = len(missing) == 0
    return passed, f"missing={missing}" if missing else "all 5 figures saved"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Load results + shapes",     test_load_results),
        ("2. Synthetic power-law fit",    test_synthetic_fit),
        ("3. Actual data fit",            test_actual_fit),
        ("4. Bootstrap CIs",             test_bootstrap),
        ("5. Phase transition test",      test_phase_transition),
        ("6. All figures generate",       test_figures_generate),
    ]

    print("=" * 60, flush=True)
    print(" H-Sweep Analysis — Validation", flush=True)
    print("=" * 60, flush=True)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        print(f"         {msg}", flush=True)
        if not passed:
            all_passed = False

    print("-" * 60, flush=True)
    if all_passed:
        print(" All 6 tests PASSED.", flush=True)
    else:
        print(" Some tests FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
