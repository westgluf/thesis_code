from __future__ import annotations

import argparse
import json
import math
import os
import py_compile
import shutil
from pathlib import Path

import torch

from src.config import load_yaml
from src.deep_hedging_model import MLPHedge
from src.hedge_core import rollout_strategy
from src.objectives import (
    CVaRObjective,
    EntropicRiskObjective,
    MeanVarianceObjective,
    build_objective,
)
from src.paths import (
    aggregated_by_method_path,
    arrays_debug_path,
    baseline_hist_plot_path,
    baseline_metrics_bsprice_path,
    baseline_metrics_mcprice_path,
    baseline_tail_plot_path,
    benchmark_root_dir,
    benchmark_run_dir,
    best_state_path,
    feature_norm_path,
    get_archive_dir,
    get_run_dir,
    hist_plot_path,
    last_state_path,
    latest_baseline_archive_metrics_path,
    metrics_bs_path,
    metrics_nn_path,
    metrics_summary_path,
    paired_comparisons_path,
    scenario_summary_path,
    manifest_runs_csv_path,
    manifest_runs_json_path,
    new_baseline_archive_metrics_path,
    pl_bs_array_path,
    pl_nn_array_path,
    run_cfg_path,
    run_meta_path,
    seed_info_path,
    seed_level_metrics_path,
    summary_rows_csv_path,
    summary_rows_json_path,
    tail_plot_path,
    train_curve_path,
    train_log_path,
    turnover_bs_array_path,
    turnover_nn_array_path,
    win_summary_path,
)
from src.rebuild_benchmark_statistics import rebuild_benchmark_statistics
from src.run_gbm_baseline import main as run_gbm_baseline_main
from src.train_deephedge_gbm import main as train_deephedge_gbm_main
from src.world_gbm import make_gbm_dataset, make_gbm_robust_dataset, policy_input_dim
from src.benchmark_repro import benchmark_run_id


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
    _run_objective_smoke_checks()
    _run_feature_contract_smoke_checks()
    _run_robust_dataset_smoke_checks()
    run_gbm_baseline_main()
    train_deephedge_gbm_main()

    cfg = _load_active_cfg()
    run_dir = get_run_dir(cfg)
    benchmark_dir = benchmark_run_dir(run_dir, benchmark_run_id(cfg))
    rebuild_benchmark_statistics(run_dir, bootstrap_resamples=100)

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
            best_state_path(benchmark_dir),
            last_state_path(benchmark_dir),
            feature_norm_path(benchmark_dir),
            train_log_path(benchmark_dir),
            metrics_bs_path(benchmark_dir),
            metrics_nn_path(benchmark_dir),
            arrays_debug_path(benchmark_dir),
            hist_plot_path(benchmark_dir),
            tail_plot_path(benchmark_dir),
            run_cfg_path(benchmark_dir),
            seed_info_path(benchmark_dir),
            run_meta_path(benchmark_dir),
            metrics_summary_path(benchmark_dir),
            pl_bs_array_path(benchmark_dir),
            pl_nn_array_path(benchmark_dir),
            turnover_bs_array_path(benchmark_dir),
            turnover_nn_array_path(benchmark_dir),
            train_curve_path(benchmark_dir),
            manifest_runs_json_path(run_dir),
            manifest_runs_csv_path(run_dir),
            summary_rows_json_path(run_dir),
            summary_rows_csv_path(run_dir),
            seed_level_metrics_path(run_dir),
            aggregated_by_method_path(run_dir),
            paired_comparisons_path(run_dir),
            win_summary_path(run_dir),
            scenario_summary_path(run_dir),
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

    _reset_run_dir(run_dir)

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


def _reset_run_dir(run_dir: Path) -> None:
    benchmark_dir = benchmark_root_dir(run_dir)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in run_dir.iterdir():
        if child == benchmark_dir:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def _run_feature_contract_smoke_checks() -> None:
    torch.manual_seed(0)
    base_kwargs = {
        "S0": 1.0,
        "sigma_true": 0.2,
        "T": 1.0,
        "n": 5,
        "K": 1.0,
        "N_train": 8,
        "N_val": 4,
        "N_test": 4,
        "seed": 7,
    }
    expected_dims = {"B": 2, "C": 4, "D": 3}

    for feature_set, feature_dim in expected_dims.items():
        dataset = make_gbm_dataset(**base_kwargs, feature_set=feature_set, sigma_in=0.2)

        for split_name in ("F_tr", "F_va", "F_te"):
            feats = dataset[split_name]
            if feats.shape[-1] != feature_dim:
                raise AssertionError(
                    f"{feature_set} {split_name} expected last dim {feature_dim}, got {feats.shape[-1]}"
                )

        if dataset["feature_dim"] != feature_dim:
            raise AssertionError(
                f"{feature_set} feature_dim expected {feature_dim}, got {dataset['feature_dim']}"
            )

        model = MLPHedge(in_dim=policy_input_dim(feature_set), hidden=8, depth=1)
        feats_te = torch.tensor(dataset["F_te"], dtype=torch.float32)
        deltas = rollout_strategy(model, feats_te)
        expected_shape = (dataset["F_te"].shape[0], base_kwargs["n"])
        if tuple(deltas.shape) != expected_shape:
            raise AssertionError(
                f"{feature_set} rollout expected shape {expected_shape}, got {tuple(deltas.shape)}"
            )


def _run_objective_smoke_checks() -> None:
    pl = torch.tensor([0.10, -0.20, 0.00, 0.30], dtype=torch.float32)

    cvar = CVaRObjective(alpha=0.95, w0=0.05)
    cvar_value = float(cvar(pl).detach().cpu())
    loss = -pl
    expected_cvar = float(0.05 + torch.relu(loss - 0.05).mean().item() / (1.0 - 0.95))
    if not math.isclose(cvar_value, expected_cvar, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"CVaR objective mismatch: expected {expected_cvar}, got {cvar_value}")

    mean_variance = MeanVarianceObjective(lambda_mv=2.0)
    mv_value = float(mean_variance(pl).detach().cpu())
    expected_mv = float(-pl.mean().item() + 0.5 * 2.0 * pl.var(unbiased=False).item())
    if not math.isclose(mv_value, expected_mv, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"Mean-variance objective mismatch: expected {expected_mv}, got {mv_value}")

    entropic_small = EntropicRiskObjective(gamma=2.0)
    entropic_small_value = float(entropic_small(pl).detach().cpu())
    expected_entropic_small = float(torch.log(torch.exp(-2.0 * pl).mean()).item() / 2.0)
    if not math.isclose(entropic_small_value, expected_entropic_small, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(
            f"Entropic objective mismatch: expected {expected_entropic_small}, got {entropic_small_value}"
        )

    pl_large = torch.tensor([4.0, -3.0, 2.0, -1.5], dtype=torch.float32)
    entropic = EntropicRiskObjective(gamma=7.5)
    entropic_value = entropic(pl_large)
    if not torch.isfinite(entropic_value):
        raise AssertionError("Entropic objective produced a non-finite value")

    objective = build_objective(name="entropic", gamma=2.0)
    if not isinstance(objective, EntropicRiskObjective):
        raise AssertionError("build_objective did not construct an EntropicRiskObjective")


def _run_robust_dataset_smoke_checks() -> None:
    base_kwargs = {
        "S0": 1.0,
        "sigma_true": 0.2,
        "robust_sigmas": (0.15, 0.20, 0.25),
        "T": 1.0,
        "n": 5,
        "K": 1.0,
        "N_train": 9,
        "N_val": 6,
        "N_test": 4,
        "seed": 11,
    }
    dataset = make_gbm_robust_dataset(**base_kwargs, feature_set="D", sigma_in_eval=0.2)
    if dataset["F_tr"].shape != (9, 5, 3):
        raise AssertionError(f"robust F_tr shape mismatch: expected (9, 5, 3), got {dataset['F_tr'].shape}")
    if dataset["F_va"].shape != (6, 5, 3):
        raise AssertionError(f"robust F_va shape mismatch: expected (6, 5, 3), got {dataset['F_va'].shape}")
    if dataset["F_te"].shape != (4, 5, 3):
        raise AssertionError(f"robust F_te shape mismatch: expected (4, 5, 3), got {dataset['F_te'].shape}")


if __name__ == "__main__":
    raise SystemExit(main())
