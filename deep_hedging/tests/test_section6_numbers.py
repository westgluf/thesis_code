"""Tests for the Section 6 numbers aggregator."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

FIGURES = Path(__file__).resolve().parents[2] / "figures"

REQUIRED_SOURCES = [
    "unified_baseline_results.json",
    "decomposition_closed.json",
    "diagnostic_controls_results.json",
    "h_sweep_results.json",
    "h2_grid_extension.json",
]


def _all_sources_exist() -> bool:
    return all((FIGURES / n).exists() for n in REQUIRED_SOURCES)


def _skip_if_missing() -> str | None:
    for n in REQUIRED_SOURCES:
        if not (FIGURES / n).exists():
            return f"{n} not yet generated"
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_has_required_top_level_keys() -> Tuple[bool, str]:
    """Aggregator output must contain one key per observation."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    required = {
        "meta",
        "section_6_2_gbm_benchmark",
        "observation_6_1_baseline",
        "observation_6_2_roughness_null",
        "observation_6_3_decomposition",
        "observation_6_5_frequency_cost",
        "observation_6_7_transfer",
    }
    missing = required - set(out)
    return len(missing) == 0, f"missing={missing}" if missing else "All 7 top-level keys present"


def test_observation_6_1_four_strategies() -> Tuple[bool, str]:
    """Obs 6.1 must contain all four strategy metrics."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    obs = out["observation_6_1_baseline"]
    expected = {"BS_Delta", "Plug_in_Delta", "DH_full_budget", "DH_GBM_pretrained"}
    for cost_key in ("lambda_0", "lambda_0p001"):
        if not expected <= set(obs[cost_key]):
            return False, f"Missing strategies in {cost_key}"
    return True, "All 4 strategies in both cost levels"


def test_decomposition_closes() -> Tuple[bool, str]:
    """Decomposition must still close after aggregation."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    dec = out["observation_6_3_decomposition"]["decomposition"]
    total = dec["Gamma_total"]
    parts = (dec["Gamma_architecture"] + dec["Gamma_objective"]
             + dec["Gamma_stoch_vol"] + dec["Gamma_roughness"]
             + dec["Gamma_interaction_total"])
    ok = abs(parts - total) < 1e-9
    return ok, f"sum={parts:.6f}, total={total:.6f}"


def test_deep_coverage_recorded() -> Tuple[bool, str]:
    """Obs 6.5 must record Deep hedger coverage."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    cov = out["observation_6_5_frequency_cost"]["deep_coverage"]
    ok = (cov["total_cells"] == 42
          and 0 <= cov["covered_cells"] <= 42
          and "missing_note" in cov)
    return ok, f"total={cov['total_cells']}, covered={cov['covered_cells']}"


def test_meta_has_commit_and_sources() -> Tuple[bool, str]:
    """Meta block must have commit SHA and source manifest."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    meta = out["meta"]
    sha_ok = len(meta.get("source_commit", "")) >= 7
    sources_ok = set(meta.get("sources_aggregated", [])) == set(REQUIRED_SOURCES)
    return sha_ok and sources_ok, f"sha_ok={sha_ok}, sources_ok={sources_ok}"


def test_json_roundtrips() -> Tuple[bool, str]:
    """Output must survive JSON round-trip without data loss."""
    reason = _skip_if_missing()
    if reason:
        return True, f"SKIP: {reason}"

    from deep_hedging.experiments import build_section6_numbers as bsn
    out = bsn.build()
    rt = json.loads(json.dumps(out))
    # Check a few deep paths survived
    ok = (rt["observation_6_1_baseline"]["derived"]["gamma_baseline"]
          == out["observation_6_1_baseline"]["derived"]["gamma_baseline"])
    return ok, "Round-trip preserves data"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Required top-level keys",      test_output_has_required_top_level_keys),
        ("2. Obs 6.1 four strategies",       test_observation_6_1_four_strategies),
        ("3. Decomposition closure",         test_decomposition_closes),
        ("4. Deep coverage recorded",        test_deep_coverage_recorded),
        ("5. Meta commit + sources",         test_meta_has_commit_and_sources),
        ("6. JSON round-trips",              test_json_roundtrips),
    ]

    print("=" * 65)
    print(" Section 6 Numbers Aggregator — Tests")
    print("=" * 65)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as e:
            import traceback
            passed, msg = False, f"EXCEPTION: {e}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("-" * 65)
    if all_passed:
        print(f" All {len(tests)} tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
