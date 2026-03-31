from __future__ import annotations

import argparse
import json
import os
import py_compile
import shutil
from pathlib import Path

from src.config import load_yaml
from src.paths import (
    arrays_debug_path,
    baseline_hist_plot_path,
    baseline_metrics_bsprice_path,
    baseline_metrics_mcprice_path,
    baseline_tail_plot_path,
    best_state_path,
    feature_norm_path,
    get_archive_dir,
    get_run_dir,
    hist_plot_path,
    last_state_path,
    latest_baseline_archive_metrics_path,
    metrics_bs_path,
    metrics_nn_path,
    new_baseline_archive_metrics_path,
    run_cfg_path,
    tail_plot_path,
    train_log_path,
)
from src.run_gbm_baseline import main as run_gbm_baseline_main
from src.train_deephedge_gbm import main as train_deephedge_gbm_main


KEYS = ["std_PL", "ES_loss_0.95", "VaR_loss_0.95", "ES_loss_0.99", "VaR_loss_0.99"]
TOL = 1e-10
REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_main() -> int:
    for pycache_dir in REPO_ROOT.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)
    for ds_store_path in REPO_ROOT.rglob(".DS_Store"):
        if ds_store_path.is_file():
            ds_store_path.unlink(missing_ok=True)
    print("clean OK")
    return 0


def compile_main() -> int:
    for py_path in _iter_python_files():
        py_compile.compile(str(py_path), doraise=True)
    print("compile OK")
    return 0


def smoke_main() -> int:
    clean_main()
    compile_main()
    run_gbm_baseline_main()
    train_deephedge_gbm_main()

    cfg = _load_active_cfg()
    run_dir = get_run_dir(cfg)

    _require_files(
        [
            baseline_metrics_mcprice_path(),
            baseline_metrics_bsprice_path(),
            baseline_hist_plot_path(),
            baseline_tail_plot_path(),
            metrics_bs_path(run_dir),
            metrics_nn_path(run_dir),
            hist_plot_path(run_dir),
            tail_plot_path(run_dir),
            arrays_debug_path(run_dir),
            best_state_path(run_dir),
            last_state_path(run_dir),
            feature_norm_path(run_dir),
            train_log_path(run_dir),
            run_cfg_path(run_dir),
        ]
    )

    _seed_guard_baseline_if_missing(run_dir)

    status = guard_main()
    if status != 0:
        return status

    print("smoke OK")
    return 0


def guard_main() -> int:
    baseline_path = latest_baseline_archive_metrics_path()
    if baseline_path is None:
        print(f"ERROR: no baseline in {get_archive_dir()}. Create one first.")
        return 1

    cfg = _load_active_cfg()
    run_dir = get_run_dir(cfg)

    print("Using baseline:", baseline_path)

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_deephedge_gbm_main()

    current_metrics_path = metrics_nn_path(run_dir)
    if not current_metrics_path.exists():
        print("ERROR: current run did not produce", current_metrics_path)
        return 1

    base = _load_json(baseline_path)
    cur = _load_json(current_metrics_path)

    print("BASE:", {key: base.get(key) for key in KEYS})
    print("CUR: ", {key: cur.get(key) for key in KEYS})

    bad = _worse(base, cur)
    if bad:
        print("")
        print("FAIL: metrics worsened:")
        for key, old_value, new_value in bad:
            print(f"  {key}: {old_value} -> {new_value}")
        return 2

    print("")
    print("PASS: metrics not worse than baseline.")
    print("guard OK")
    return 0


def clean_cli() -> int:
    return clean_main()


def compile_cli() -> int:
    return compile_main()


def smoke_cli() -> int:
    return smoke_main()


def guard_cli() -> int:
    return guard_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.tools_cli")
    parser.add_argument("command", choices=("clean", "compile", "smoke", "guard"))
    args = parser.parse_args(argv)

    if args.command == "clean":
        return clean_main()
    if args.command == "compile":
        return compile_main()
    if args.command == "smoke":
        return smoke_main()
    return guard_main()


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for py_path in (REPO_ROOT / "src").rglob("*.py"):
        if "__pycache__" in py_path.parts or "_archive" in py_path.parts:
            continue
        files.append(py_path)
    for py_path in (REPO_ROOT / "tools").glob("*.py"):
        files.append(py_path)
    return sorted(files)


def _load_active_cfg() -> dict:
    cfg_path = os.environ.get("GBM_CFG", "configs/gbm_es95.yaml")
    return load_yaml(cfg_path)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print("ERROR: missing required artifacts:")
        for path in missing:
            print(" ", path)
        raise SystemExit(1)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_guard_baseline_if_missing(run_dir: Path) -> None:
    if latest_baseline_archive_metrics_path() is not None:
        return
    baseline_path = new_baseline_archive_metrics_path()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_nn_path(run_dir), baseline_path)
    print("Seeded guard baseline:", baseline_path)


def _worse(base: dict, cur: dict) -> list[tuple[str, float, float]]:
    bad: list[tuple[str, float, float]] = []
    for key in KEYS:
        if key in base and key in cur and (cur[key] - base[key]) > TOL:
            bad.append((key, base[key], cur[key]))
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
