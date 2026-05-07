#!/usr/bin/env python
"""
Tests for Block 1.1 — Convergence Sweep.

Five tests covering: imports, tiny run, output-collision guard,
Richardson fit on synthetic data, and no-modification of existing files.

Run:
    python -m deep_hedging.tests.test_block1_convergence
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(name: str, passed: bool, msg: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {msg}")

# ---------------------------------------------------------------------------
# Test 1: Imports
# ---------------------------------------------------------------------------

def test_imports() -> Tuple[bool, str]:
    """Verify the convergence module imports and main() exists."""
    try:
        from deep_hedging.experiments.block1_convergence import main
        from deep_hedging.experiments.block1_hardware import record_hardware_and_versions

        if not callable(main):
            return False, "main is not callable"
        if not callable(record_hardware_and_versions):
            return False, "record_hardware_and_versions is not callable"

        return True, "All imports OK; main() and record_hardware_and_versions() found"
    except Exception as exc:
        return False, f"Import error: {exc}"

# ---------------------------------------------------------------------------
# Test 2: Tiny run
# ---------------------------------------------------------------------------

def test_tiny_run() -> Tuple[bool, str]:
    """Run with minimal config into a temp directory; verify outputs."""
    from deep_hedging.experiments.block1_convergence import main

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results" / "block1"
        figures_dir = Path(tmpdir) / "figures" / "block1"

        output = main(
            n_grid=[50, 100],
            n_paths=100,
            n_paths_p0=200,
            seeds=[7001],
            results_dir=results_dir,
            figures_dir=figures_dir,
            dry_run=False,
        )

        # Check output dict
        if not isinstance(output, dict):
            return False, "main() did not return a dict"
        for key in ["meta", "config", "per_seed_results", "aggregated",
                     "richardson", "convergence_flags"]:
            if key not in output:
                return False, f"Missing key '{key}' in output dict"

        # Check JSON file
        json_path = results_dir / "convergence_sweep.json"
        if not json_path.exists():
            return False, f"JSON not created: {json_path}"
        data = json.loads(json_path.read_text())
        if "per_seed_results" not in data:
            return False, "JSON missing per_seed_results"

        # Check figures exist and are non-empty
        for fig_name in ["convergence_curves.png", "richardson_fit.png",
                         "relative_changes_table.png"]:
            fig_path = figures_dir / fig_name
            if not fig_path.exists():
                return False, f"Figure not created: {fig_name}"
            if fig_path.stat().st_size < 100:
                return False, f"Figure too small ({fig_path.stat().st_size}B): {fig_name}"

        # Check README
        readme_path = results_dir / "README.md"
        if not readme_path.exists():
            return False, "README.md not created"
        readme_text = readme_path.read_text()
        if "Block 1.1" not in readme_text:
            return False, "README.md missing 'Block 1.1' header"

        return True, (
            f"Tiny run OK: JSON={json_path.stat().st_size}B, "
            f"3 figures created, README present"
        )

# ---------------------------------------------------------------------------
# Test 3: Output-collision guard
# ---------------------------------------------------------------------------

def test_collision_guard() -> Tuple[bool, str]:
    """Verify FileExistsError if an output file already exists."""
    from deep_hedging.experiments.block1_convergence import main

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results" / "block1"
        figures_dir = Path(tmpdir) / "figures" / "block1"
        results_dir.mkdir(parents=True)
        figures_dir.mkdir(parents=True)

        # Plant a dummy collision file
        dummy = results_dir / "convergence_sweep.json"
        dummy.write_text("{}")

        try:
            main(
                n_grid=[50, 100],
                n_paths=50,
                n_paths_p0=50,
                seeds=[7001],
                results_dir=results_dir,
                figures_dir=figures_dir,
            )
            return False, "Expected FileExistsError but main() succeeded"
        except FileExistsError as exc:
            if "convergence_sweep.json" in str(exc):
                return True, f"Correctly raised FileExistsError: {exc}"
            return False, f"FileExistsError but wrong file: {exc}"
        except Exception as exc:
            return False, f"Wrong exception type: {type(exc).__name__}: {exc}"

# ---------------------------------------------------------------------------
# Test 4: Richardson fit on synthetic data
# ---------------------------------------------------------------------------

def test_richardson_synthetic() -> Tuple[bool, str]:
    """Verify Richardson fit recovers known α from synthetic data."""
    from deep_hedging.experiments.block1_convergence import fit_richardson

    # Generate synthetic: ES(n) = 10.0 + 2.0 * n^{-0.55} + noise
    rng = np.random.default_rng(42)
    n_values = [50, 100, 200, 400, 800, 1600]
    true_alpha = 0.55
    true_es_inf = 10.0
    true_C = 2.0
    metric_values = [
        true_es_inf + true_C * n**(-true_alpha) + rng.normal(0, 0.01)
        for n in n_values
    ]

    result = fit_richardson(n_values, metric_values)

    alpha_hat = result["alpha_hat"]
    es_inf = result["ES_inf"]

    alpha_ok = 0.4 <= alpha_hat <= 0.7
    es_inf_ok = 9.9 <= es_inf <= 10.1

    msg = (
        f"α̂ = {alpha_hat:.4f} (expected ~0.55, range [0.4,0.7]={'OK' if alpha_ok else 'FAIL'}), "
        f"ES(∞) = {es_inf:.4f} (expected ~10.0, range [9.9,10.1]={'OK' if es_inf_ok else 'FAIL'})"
    )

    return alpha_ok and es_inf_ok, msg

# ---------------------------------------------------------------------------
# Test 5: No modification of existing files
# ---------------------------------------------------------------------------

def test_no_existing_file_modification() -> Tuple[bool, str]:
    """Verify core simulator files are unchanged after a tiny experiment."""
    repo_root = Path(__file__).resolve().parents[2]
    watched_files = [
        repo_root / "deep_hedging" / "core" / "volterra.py",
        repo_root / "deep_hedging" / "core" / "rough_bergomi.py",
        repo_root / "deep_hedging" / "objectives" / "risk_measures.py",
        repo_root / "deep_hedging" / "objectives" / "pnl.py",
        repo_root / "deep_hedging" / "hedging" / "delta_hedger.py",
    ]

    # Record mtimes before
    mtimes_before = {}
    for f in watched_files:
        if f.exists():
            mtimes_before[str(f)] = os.path.getmtime(f)

    # Run a tiny experiment
    from deep_hedging.experiments.block1_convergence import main

    with tempfile.TemporaryDirectory() as tmpdir:
        main(
            n_grid=[50, 100],
            n_paths=50,
            n_paths_p0=50,
            seeds=[7001],
            results_dir=Path(tmpdir) / "results" / "block1",
            figures_dir=Path(tmpdir) / "figures" / "block1",
        )

    # Check mtimes unchanged
    modified = []
    for f in watched_files:
        if f.exists():
            mtime_after = os.path.getmtime(f)
            if str(f) in mtimes_before:
                if mtime_after != mtimes_before[str(f)]:
                    modified.append(f.name)

    if modified:
        return False, f"Files modified: {', '.join(modified)}"

    return True, f"All {len(watched_files)} watched files unchanged"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("1. Imports", test_imports),
        ("2. Tiny run", test_tiny_run),
        ("3. Output-collision guard", test_collision_guard),
        ("4. Richardson fit (synthetic)", test_richardson_synthetic),
        ("5. No modification of existing files", test_no_existing_file_modification),
    ]

    print("=" * 70)
    print(" Block 1.1 — Convergence Sweep Tests")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            passed, msg = fn()
        except Exception as exc:
            passed, msg = False, f"EXCEPTION: {exc}"
        _print_result(name, passed, msg)
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(" All 5 tests PASSED.")
    else:
        print(" Some tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
