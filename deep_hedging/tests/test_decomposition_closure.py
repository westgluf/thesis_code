"""Tests for the closed decomposition (build_decomposition.py).

Uses synthetic stub data to test the arithmetic without expensive training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from deep_hedging.experiments import build_decomposition as bd


def _synthetic_diagnostic_controls_json(tmp: Path) -> Path:
    """Minimal stub with realistic magnitudes for unit-testing the math."""
    stub = {
        "A": {
            "H": 0.07, "eta": 0.0, "risk_label": "es",
            "es95_bs": 10.50, "es95_dh": 10.27, "gamma": 0.23,
        },
        "A_prime": {
            "H": 0.07, "eta": 0.0, "risk_label": "mse",
            "es95_bs": 10.50, "es95_dh": 10.40, "gamma": 0.10,
        },
        "C": {
            "bs":     {"es95": 11.117},
            "dh_mse": {"es95": 10.756},
            "dh_es":  {"es95": 10.298},
        },
        "D": [
            {"H": 0.05, "eta": 0.5, "gamma": 0.344},
            {"H": 0.05, "eta": 1.9, "gamma": 0.618},
            {"H": 0.05, "eta": 3.0, "gamma": 0.350},
            {"H": 0.2,  "eta": 0.5, "gamma": 0.464},
            {"H": 0.2,  "eta": 1.9, "gamma": 0.503},
            {"H": 0.2,  "eta": 3.0, "gamma": 0.313},
            {"H": 0.5,  "eta": 0.5, "gamma": 0.456},
            {"H": 0.5,  "eta": 1.9, "gamma": 0.658},
            {"H": 0.5,  "eta": 3.0, "gamma": 0.362},
        ],
    }
    p = tmp / "stub.json"
    p.write_text(json.dumps(stub))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_factorial_components_sum_to_total(tmp_path: Path) -> Tuple[bool, str]:
    in_path = _synthetic_diagnostic_controls_json(tmp_path)
    out_path = tmp_path / "out.json"
    bd.build_from_paths(in_path, out_path)

    out = json.loads(out_path.read_text())
    decomp = out["decomposition"]
    total = decomp["Gamma_total"]
    parts = (
        decomp["Gamma_architecture"]
        + decomp["Gamma_objective"]
        + decomp["Gamma_stoch_vol"]
        + decomp["Gamma_roughness"]
        + decomp["Gamma_interaction_total"]
    )
    ok = abs(parts - total) < 1e-9 and abs(decomp["closure_residual"]) < 1e-9
    return ok, f"sum={parts:.6f}, total={total:.6f}, residual={decomp['closure_residual']:.2e}"


def test_anova_fractions_sum_to_one(tmp_path: Path) -> Tuple[bool, str]:
    in_path = _synthetic_diagnostic_controls_json(tmp_path)
    out_path = tmp_path / "out.json"
    bd.build_from_paths(in_path, out_path)

    anova = json.loads(out_path.read_text())["grid_3x3_anova"]
    frac_sum = anova["f_H"] + anova["f_eta"] + anova["f_interaction"]
    ok = abs(frac_sum - 1.0) < 1e-12
    return ok, f"f_H={anova['f_H']:.4f}, f_eta={anova['f_eta']:.4f}, f_int={anova['f_interaction']:.4f}, sum={frac_sum:.12f}"


def test_refuses_without_A_prime(tmp_path: Path) -> Tuple[bool, str]:
    stub = {"A": {"gamma": 0.23, "es95_bs": 1.0, "es95_dh": 0.77}, "C": {}, "D": []}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(stub))
    try:
        bd.build_from_paths(p, tmp_path / "out.json")
        return False, "Should have raised KeyError"
    except KeyError as e:
        ok = "A_prime" in str(e)
        return ok, f"Correctly raised KeyError: {e}"


def test_percentages_sum_to_one_hundred(tmp_path: Path) -> Tuple[bool, str]:
    in_path = _synthetic_diagnostic_controls_json(tmp_path)
    out_path = tmp_path / "out.json"
    bd.build_from_paths(in_path, out_path)

    pcts = json.loads(out_path.read_text())["decomposition"]["percentages_of_total"]
    total_pct = sum(pcts.values())
    ok = abs(total_pct - 100.0) < 1e-6
    return ok, f"Sum of percentages = {total_pct:.6f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    import tempfile

    tests = [
        ("1. 5-component sum == Gamma_total",   None),
        ("2. ANOVA fractions sum to 1.0",        None),
        ("3. Refuses without A_prime",           None),
        ("4. Percentages sum to 100%",           None),
    ]

    test_fns = [
        test_factorial_components_sum_to_total,
        test_anova_fractions_sum_to_one,
        test_refuses_without_A_prime,
        test_percentages_sum_to_one_hundred,
    ]

    print("=" * 70)
    print(" Decomposition Closure — Tests")
    print("=" * 70)

    all_passed = True
    for (name, _), fn in zip(tests, test_fns):
        try:
            with tempfile.TemporaryDirectory() as td:
                passed, msg = fn(Path(td))
        except Exception as exc:
            import traceback
            passed, msg = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(f" All {len(tests)} tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
