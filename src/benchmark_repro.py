from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.config import get
from src.logging_utils import write_csv_rows, write_json_file, write_run_config
from src.paths import (
    arrays_debug_path,
    benchmark_run_dir,
    benchmark_runs_dir,
    benchmark_spec_json_path,
    best_state_path,
    feature_norm_path,
    hist_plot_path,
    last_state_path,
    manifest_runs_csv_path,
    manifest_runs_json_path,
    metrics_bs_path,
    metrics_nn_path,
    metrics_summary_path,
    pl_bs_array_path,
    pl_nn_array_path,
    run_cfg_path,
    run_meta_path,
    seed_info_path,
    summary_rows_csv_path,
    summary_rows_json_path,
    tail_plot_path,
    train_curve_path,
    train_log_path,
    turnover_bs_array_path,
    turnover_nn_array_path,
)
from src.plots import plot_train_val_curves


@dataclass(frozen=True)
class BenchmarkRunContext:
    run_id: str
    benchmark_run_dir: Path
    created_at_utc: str
    config_hash: str


TrainingRegime = str
RunMode = str
DEFAULT_ROBUST_SIGMAS: tuple[float, ...] = (0.15, 0.20, 0.25)


def canonical_training_regime(name: str | None) -> TrainingRegime:
    raw = "oracle" if name is None else str(name).strip().lower()
    if raw not in {"oracle", "robust"}:
        raise ValueError(f"unknown training regime {name!r}; expected 'oracle' or 'robust'")
    return raw


def robust_sigmas_from_cfg(cfg: Mapping[str, Any]) -> tuple[float, ...]:
    raw = get(cfg, "benchmark.robust_sigmas", DEFAULT_ROBUST_SIGMAS)
    if isinstance(raw, (list, tuple)):
        values = tuple(float(value) for value in raw)
    else:
        values = (float(raw),)
    if not values:
        raise ValueError("benchmark.robust_sigmas must contain at least one volatility")
    return values


def deep_hedge_method_name(training_regime: str | None) -> str:
    return f"deep_hedge_{canonical_training_regime(training_regime)}"


def canonical_run_mode(name: str | None) -> RunMode:
    raw = "debug" if name is None else str(name).strip().lower()
    if raw not in {"benchmark", "debug", "smoke"}:
        raise ValueError(f"unknown run mode {name!r}; expected 'benchmark', 'debug', or 'smoke'")
    return raw


def benchmark_run_mode_from_cfg(cfg: Mapping[str, Any]) -> RunMode:
    return canonical_run_mode(get(cfg, "benchmark.run_mode", "debug"))


def benchmark_is_eligible_from_cfg(cfg: Mapping[str, Any]) -> bool:
    raw = get(cfg, "benchmark.is_benchmark_eligible", None)
    if raw is None:
        return benchmark_run_mode_from_cfg(cfg) == "benchmark"
    return _coerce_bool(raw)


def benchmark_campaign_id_from_cfg(cfg: Mapping[str, Any]) -> str:
    raw = str(get(cfg, "benchmark.campaign_id", "main_benchmark")).strip()
    return raw or "main_benchmark"


def benchmark_campaign_role_from_cfg(cfg: Mapping[str, Any]) -> str:
    raw = str(get(cfg, "benchmark.campaign_role", "main")).strip().lower()
    return raw or "main"


def effective_robust_sigmas(
    training_regime: str | None,
    robust_sigmas: Sequence[float] | tuple[float, ...],
) -> tuple[float, ...]:
    if canonical_training_regime(training_regime) != "robust":
        return ()
    return tuple(float(value) for value in robust_sigmas)


def benchmark_scenario_id(
    *,
    training_regime: str,
    feature_set: str,
    objective_name: str,
    sigma_true: float,
    sigma_bar: float,
    lambda_cost: float,
    robust_sigmas: Sequence[float] | tuple[float, ...] = (),
) -> str:
    sigmas = effective_robust_sigmas(training_regime, robust_sigmas)
    robust_slug = "none" if not sigmas else ",".join(_float_slug(value) for value in sigmas)
    return (
        f"train={canonical_training_regime(training_regime)}"
        f"|feat={str(feature_set).strip()}"
        f"|obj={str(objective_name).strip()}"
        f"|sigtrue={_float_slug(float(sigma_true))}"
        f"|sigbar={_float_slug(float(sigma_bar))}"
        f"|lam={_float_slug(float(lambda_cost))}"
        f"|robust={robust_slug}"
    )


def benchmark_metadata_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    training_regime = canonical_training_regime(get(cfg, "benchmark.training_regime", "oracle"))
    robust_sigmas = effective_robust_sigmas(training_regime, robust_sigmas_from_cfg(cfg))
    feature_set = _feature_set_from_cfg(cfg)
    objective_name = _objective_name_from_cfg(cfg)
    sigma_true = float(get(cfg, "data.sigma_true", 0.0))
    sigma_bar = float(get(cfg, "data.sigma_bar", 0.0))
    lambda_cost = float(get(cfg, "data.lam_cost", 0.0))
    return {
        "seed": int(get(cfg, "data.seed", 0)),
        "campaign_id": benchmark_campaign_id_from_cfg(cfg),
        "campaign_role": benchmark_campaign_role_from_cfg(cfg),
        "training_regime": training_regime,
        "run_mode": benchmark_run_mode_from_cfg(cfg),
        "is_benchmark_eligible": benchmark_is_eligible_from_cfg(cfg),
        "bs_method": "bs_delta",
        "deep_hedge_method": deep_hedge_method_name(training_regime),
        "robust_sigmas": list(robust_sigmas),
        "feature_set": feature_set,
        "objective_name": objective_name,
        "sigma_true": sigma_true,
        "sigma_bar": sigma_bar,
        "lambda_cost": lambda_cost,
        "scenario_id": benchmark_scenario_id(
            training_regime=training_regime,
            feature_set=feature_set,
            objective_name=objective_name,
            sigma_true=sigma_true,
            sigma_bar=sigma_bar,
            lambda_cost=lambda_cost,
            robust_sigmas=robust_sigmas,
        ),
    }


def prepare_benchmark_run(cfg: Mapping[str, Any], root_run_dir: str | Path) -> BenchmarkRunContext:
    created_at = _utc_now()
    run_id = benchmark_run_id(cfg)
    repro_dir = benchmark_run_dir(root_run_dir, run_id)
    metadata = benchmark_metadata_from_cfg(cfg)
    if repro_dir.exists():
        for child in repro_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    repro_dir.mkdir(parents=True, exist_ok=True)

    config_hash = _config_hash(cfg)
    write_run_config(run_cfg_path(repro_dir), cfg)
    write_json_file(
        seed_info_path(repro_dir),
        _seed_info_payload(cfg=cfg, benchmark_run_dir=repro_dir),
        sort_keys=True,
    )
    write_json_file(
        run_meta_path(repro_dir),
        {
            "run_id": run_id,
            "created_at_utc": created_at,
            "status": "running",
            "root_run_dir": str(Path(root_run_dir)),
            "benchmark_run_dir": str(repro_dir),
            "config_hash": config_hash,
            **metadata,
            "reproduce_command": f"GBM_CFG={run_cfg_path(repro_dir)} python -m src.train_deephedge_gbm",
        },
        sort_keys=True,
    )
    return BenchmarkRunContext(
        run_id=run_id,
        benchmark_run_dir=repro_dir,
        created_at_utc=created_at,
        config_hash=config_hash,
    )


def finalize_benchmark_run(
    *,
    cfg: Mapping[str, Any],
    root_run_dir: str | Path,
    context: BenchmarkRunContext,
    metrics_bs: Mapping[str, Any],
    metrics_nn: Mapping[str, Any],
    pl_bs: np.ndarray,
    pl_nn: np.ndarray,
    turnover_bs: np.ndarray,
    turnover_nn: np.ndarray,
    train_log: list[Mapping[str, Any]],
) -> None:
    repro_dir = context.benchmark_run_dir
    metadata = benchmark_metadata_from_cfg(cfg)
    np.save(pl_bs_array_path(repro_dir), np.asarray(pl_bs))
    np.save(pl_nn_array_path(repro_dir), np.asarray(pl_nn))
    np.save(turnover_bs_array_path(repro_dir), np.asarray(turnover_bs))
    np.save(turnover_nn_array_path(repro_dir), np.asarray(turnover_nn))
    plot_train_val_curves(train_log, str(train_curve_path(repro_dir)))

    metrics_summary = {
        "bs": dict(metrics_bs),
        "nn": dict(metrics_nn),
        "turnover_bs": _turnover_summary(turnover_bs),
        "turnover_nn": _turnover_summary(turnover_nn),
    }
    write_json_file(metrics_summary_path(repro_dir), metrics_summary, sort_keys=True)

    meta = {
        "run_id": context.run_id,
        "created_at_utc": context.created_at_utc,
        "status": "completed",
        "root_run_dir": str(Path(root_run_dir)),
        "benchmark_run_dir": str(repro_dir),
        "config_hash": context.config_hash,
        **metadata,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "artifacts": {
            "run_cfg": run_cfg_path(repro_dir).name,
            "seed_info": seed_info_path(repro_dir).name,
            "run_meta": run_meta_path(repro_dir).name,
            "feature_norm": feature_norm_path(repro_dir).name,
            "train_log": train_log_path(repro_dir).name,
            "best_state": best_state_path(repro_dir).name,
            "last_state": last_state_path(repro_dir).name,
            "metrics_bs": metrics_bs_path(repro_dir).name,
            "metrics_nn": metrics_nn_path(repro_dir).name,
            "metrics_summary": metrics_summary_path(repro_dir).name,
            "pl_bs": pl_bs_array_path(repro_dir).name,
            "pl_nn": pl_nn_array_path(repro_dir).name,
            "turnover_bs": turnover_bs_array_path(repro_dir).name,
            "turnover_nn": turnover_nn_array_path(repro_dir).name,
            "arrays_debug": arrays_debug_path(repro_dir).name,
            "hist_plot": hist_plot_path(repro_dir).name,
            "tail_plot": tail_plot_path(repro_dir).name,
            "train_curve": train_curve_path(repro_dir).name,
        },
        "reproduce_command": f"GBM_CFG={run_cfg_path(repro_dir)} python -m src.train_deephedge_gbm",
    }
    write_json_file(run_meta_path(repro_dir), meta, sort_keys=True)
    rebuild_benchmark_manifest(root_run_dir)
    rebuild_benchmark_summary_rows(root_run_dir)


def fail_benchmark_run(
    *,
    cfg: Mapping[str, Any],
    root_run_dir: str | Path,
    context: BenchmarkRunContext,
    error: BaseException,
) -> None:
    repro_dir = context.benchmark_run_dir
    metadata = benchmark_metadata_from_cfg(cfg)
    existing: dict[str, Any] = {}
    meta_path = run_meta_path(repro_dir)
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    payload = {
        "run_id": context.run_id,
        "created_at_utc": existing.get("created_at_utc", context.created_at_utc),
        "status": "failed",
        "root_run_dir": str(Path(root_run_dir)),
        "benchmark_run_dir": str(repro_dir),
        "config_hash": existing.get("config_hash", context.config_hash),
        **metadata,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "reproduce_command": f"GBM_CFG={run_cfg_path(repro_dir)} python -m src.train_deephedge_gbm",
    }
    write_json_file(meta_path, payload, sort_keys=True)
    rebuild_benchmark_manifest(root_run_dir)
    rebuild_benchmark_summary_rows(root_run_dir)


def rebuild_benchmark_manifest(root_run_dir: str | Path) -> None:
    runs_dir = benchmark_runs_dir(root_run_dir)
    meta_paths = sorted(runs_dir.glob("*/run_meta.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in meta_paths]
    rows = [row for row in rows if _meta_is_benchmark_eligible(row)]
    benchmark_spec = _load_benchmark_spec(root_run_dir)
    if benchmark_spec is not None:
        rows = [row for row in rows if _meta_matches_benchmark_spec(row, benchmark_spec)]
    rows.sort(key=lambda row: str(row.get("run_id", "")))

    payload = {
        "generated_at_utc": _utc_now(),
        "num_runs": len(rows),
        "runs": rows,
    }
    write_json_file(manifest_runs_json_path(root_run_dir), payload, sort_keys=True)

    csv_rows = [_manifest_csv_row(row) for row in rows]
    header = (
        "scenario_id",
        "run_id",
        "status",
        "created_at_utc",
        "seed",
        "campaign_id",
        "campaign_role",
        "run_mode",
        "is_benchmark_eligible",
        "training_regime",
        "bs_method",
        "deep_hedge_method",
        "robust_sigmas",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "benchmark_run_dir",
        "root_run_dir",
        "config_hash",
        "git_commit",
        "reproduce_command",
    )
    write_csv_rows(manifest_runs_csv_path(root_run_dir), header, csv_rows)


def rebuild_benchmark_summary_rows(root_run_dir: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    benchmark_spec = _load_benchmark_spec(root_run_dir)
    for meta_path in sorted(benchmark_runs_dir(root_run_dir).glob("*/run_meta.json")):
        metrics_path = metrics_summary_path(meta_path.parent)
        if not metrics_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            continue
        if not _meta_is_benchmark_eligible(meta):
            continue
        if benchmark_spec is not None and not _meta_matches_benchmark_spec(meta, benchmark_spec):
            continue
        metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.extend(_summary_rows_for_run(meta, metrics_summary))

    rows.sort(key=lambda row: (str(row.get("run_id", "")), str(row.get("method", ""))))
    payload = {
        "generated_at_utc": _utc_now(),
        "num_rows": len(rows),
        "rows": rows,
    }
    write_json_file(summary_rows_json_path(root_run_dir), payload, sort_keys=True)

    header = (
        "scenario_id",
        "run_id",
        "created_at_utc",
        "status",
        "campaign_id",
        "campaign_role",
        "run_mode",
        "is_benchmark_eligible",
        "method",
        "training_regime",
        "deep_hedge_method",
        "seed",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "robust_sigmas",
        "mean_PL",
        "std_PL",
        "entropic",
        "VaR_loss_0.95",
        "ES_loss_0.95",
        "VaR_loss_0.99",
        "ES_loss_0.99",
        "mean_turnover",
        "max_turnover",
        "total_turnover",
        "benchmark_run_dir",
        "root_run_dir",
        "config_hash",
        "git_commit",
        "reproduce_command",
    )
    write_csv_rows(summary_rows_csv_path(root_run_dir), header, rows)


def benchmark_run_id(cfg: Mapping[str, Any]) -> str:
    seed = int(get(cfg, "data.seed", 0))
    feature_set = str(get(cfg, "features.feature_set", "B")).strip().lower()
    objective = _objective_slug(cfg)
    training = _training_slug(cfg)
    run_mode = benchmark_run_mode_from_cfg(cfg)
    mode_slug = "" if run_mode == "benchmark" else f"__mode_{run_mode}"
    sigma_true = _float_slug(float(get(cfg, "data.sigma_true", 0.0)))
    sigma_bar = _float_slug(float(get(cfg, "data.sigma_bar", 0.0)))
    lam_cost = _float_slug(float(get(cfg, "data.lam_cost", 0.0)))
    return (
        f"seed_{seed:04d}"
        f"{mode_slug}"
        f"__train_{training}"
        f"__feat_{feature_set}"
        f"__obj_{objective}"
        f"__sigtrue_{sigma_true}"
        f"__sigbar_{sigma_bar}"
        f"__lam_{lam_cost}"
    )


def _seed_info_payload(*, cfg: Mapping[str, Any], benchmark_run_dir: Path) -> dict[str, Any]:
    seed = int(get(cfg, "data.seed", 0))
    return {
        "data_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_initial_seed": int(torch.initial_seed()),
        "reproduce_command": f"GBM_CFG={run_cfg_path(benchmark_run_dir)} python -m src.train_deephedge_gbm",
    }


def _objective_slug(cfg: Mapping[str, Any]) -> str:
    name = str(get(cfg, "objective.name", "cvar")).strip().lower().replace("-", "_")
    if name == "cvar":
        return f"cvar_a{_float_slug(float(get(cfg, 'objective.alpha', 0.95)))}"
    if name == "entropic":
        return f"entropic_g{_float_slug(float(get(cfg, 'objective.gamma', 1.0)))}"
    return f"mean_variance_l{_float_slug(float(get(cfg, 'objective.lambda_mv', 1.0)))}"


def _training_slug(cfg: Mapping[str, Any]) -> str:
    training_regime = canonical_training_regime(get(cfg, "benchmark.training_regime", "oracle"))
    if training_regime != "robust":
        return training_regime
    sigma_slug = "-".join(_float_slug(value) for value in effective_robust_sigmas(training_regime, robust_sigmas_from_cfg(cfg)))
    return f"robust_s{sigma_slug}"


def _turnover_summary(turnover: np.ndarray) -> dict[str, float]:
    arr = np.asarray(turnover, dtype=float)
    return {
        "mean_turnover": float(arr.mean()),
        "max_turnover": float(arr.max()),
        "total_turnover": float(arr.sum()),
    }


def _manifest_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    training_regime = _meta_training_regime(row)
    run_mode = _meta_run_mode(row)
    return {
        "scenario_id": _meta_scenario_id(row),
        "run_id": row.get("run_id", ""),
        "status": row.get("status", ""),
        "created_at_utc": row.get("created_at_utc", ""),
        "seed": row.get("seed", ""),
        "campaign_id": _meta_campaign_id(row),
        "campaign_role": _meta_campaign_role(row),
        "run_mode": run_mode,
        "is_benchmark_eligible": _meta_is_benchmark_eligible(row),
        "training_regime": training_regime,
        "bs_method": "bs_delta",
        "deep_hedge_method": row.get("deep_hedge_method", deep_hedge_method_name(training_regime)),
        "robust_sigmas": _meta_robust_sigmas(row),
        "feature_set": row.get("feature_set", ""),
        "objective_name": row.get("objective_name", ""),
        "sigma_true": row.get("sigma_true", ""),
        "sigma_bar": row.get("sigma_bar", ""),
        "lambda_cost": row.get("lambda_cost", ""),
        "benchmark_run_dir": row.get("benchmark_run_dir", ""),
        "root_run_dir": row.get("root_run_dir", ""),
        "config_hash": row.get("config_hash", ""),
        "git_commit": row.get("git_commit", ""),
        "reproduce_command": row.get("reproduce_command", ""),
    }


def _robust_sigmas_csv(values: Any) -> str:
    if isinstance(values, (list, tuple)):
        return "|".join(_float_slug(float(value)) for value in values)
    if values in ("", None):
        return ""
    return str(values)


def _summary_rows_for_run(meta: Mapping[str, Any], metrics_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    training_regime = _meta_training_regime(meta)
    deep_hedge_method = str(meta.get("deep_hedge_method", deep_hedge_method_name(training_regime)))
    common = {
        "scenario_id": _meta_scenario_id(meta),
        "run_id": meta.get("run_id", ""),
        "created_at_utc": meta.get("created_at_utc", ""),
        "status": meta.get("status", ""),
        "campaign_id": _meta_campaign_id(meta),
        "campaign_role": _meta_campaign_role(meta),
        "run_mode": _meta_run_mode(meta),
        "is_benchmark_eligible": _meta_is_benchmark_eligible(meta),
        "training_regime": training_regime,
        "deep_hedge_method": deep_hedge_method,
        "seed": meta.get("seed", ""),
        "feature_set": meta.get("feature_set", ""),
        "objective_name": meta.get("objective_name", ""),
        "sigma_true": meta.get("sigma_true", ""),
        "sigma_bar": meta.get("sigma_bar", ""),
        "lambda_cost": meta.get("lambda_cost", ""),
        "robust_sigmas": _meta_robust_sigmas(meta),
        "benchmark_run_dir": meta.get("benchmark_run_dir", ""),
        "root_run_dir": meta.get("root_run_dir", ""),
        "config_hash": meta.get("config_hash", ""),
        "git_commit": meta.get("git_commit", ""),
        "reproduce_command": meta.get("reproduce_command", ""),
    }
    method_specs = (
        ("bs_delta", "bs", "turnover_bs"),
        (deep_hedge_method, "nn", "turnover_nn"),
    )
    for method_name, metrics_key, turnover_key in method_specs:
        metrics = dict(metrics_summary.get(metrics_key, {}))
        turnover = dict(metrics_summary.get(turnover_key, {}))
        rows.append(
            {
                **common,
                "method": method_name,
                "mean_PL": metrics.get("mean_PL", ""),
                "std_PL": metrics.get("std_PL", ""),
                "entropic": metrics.get("entropic", ""),
                "VaR_loss_0.95": metrics.get("VaR_loss_0.95", ""),
                "ES_loss_0.95": metrics.get("ES_loss_0.95", ""),
                "VaR_loss_0.99": metrics.get("VaR_loss_0.99", ""),
                "ES_loss_0.99": metrics.get("ES_loss_0.99", ""),
                "mean_turnover": turnover.get("mean_turnover", ""),
                "max_turnover": turnover.get("max_turnover", ""),
                "total_turnover": turnover.get("total_turnover", ""),
            }
        )
    return rows


def _meta_training_regime(meta: Mapping[str, Any]) -> str:
    raw = meta.get("training_regime", "")
    if raw not in ("", None):
        return canonical_training_regime(str(raw))
    method = str(meta.get("deep_hedge_method", "")).strip().lower()
    if method.startswith("deep_hedge_"):
        return canonical_training_regime(method.removeprefix("deep_hedge_"))
    return "oracle"


def _meta_campaign_id(meta: Mapping[str, Any]) -> str:
    raw = str(meta.get("campaign_id", "main_benchmark")).strip()
    return raw or "main_benchmark"


def _meta_campaign_role(meta: Mapping[str, Any]) -> str:
    raw = str(meta.get("campaign_role", "main")).strip().lower()
    return raw or "main"


def _meta_run_mode(meta: Mapping[str, Any]) -> str:
    raw = meta.get("run_mode", "")
    if raw in ("", None):
        return "debug"
    return canonical_run_mode(str(raw))


def _meta_is_benchmark_eligible(meta: Mapping[str, Any]) -> bool:
    raw = meta.get("is_benchmark_eligible", None)
    if raw in ("", None):
        return _meta_run_mode(meta) == "benchmark"
    return _coerce_bool(raw)


def _meta_robust_sigmas(meta: Mapping[str, Any]) -> str:
    return _robust_sigmas_csv(effective_robust_sigmas(_meta_training_regime(meta), _coerce_robust_sigmas(meta.get("robust_sigmas", []))))


def _meta_scenario_id(meta: Mapping[str, Any]) -> str:
    raw = str(meta.get("scenario_id", "")).strip()
    if raw:
        return raw
    return benchmark_scenario_id(
        training_regime=_meta_training_regime(meta),
        feature_set=str(meta.get("feature_set", "")).strip(),
        objective_name=str(meta.get("objective_name", "")).strip(),
        sigma_true=float(meta.get("sigma_true", 0.0)),
        sigma_bar=float(meta.get("sigma_bar", 0.0)),
        lambda_cost=float(meta.get("lambda_cost", 0.0)),
        robust_sigmas=_coerce_robust_sigmas(meta.get("robust_sigmas", [])),
    )


def _load_benchmark_spec(root_run_dir: str | Path) -> dict[str, Any] | None:
    path = benchmark_spec_json_path(root_run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_matches_benchmark_spec(meta: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    campaign_id = str(spec.get("campaign_id", "")).strip()
    if campaign_id and _meta_campaign_id(meta) != campaign_id:
        return False
    campaign_role = str(spec.get("campaign_role", "")).strip().lower()
    if campaign_role and _meta_campaign_role(meta) != campaign_role:
        return False
    training_regime = _meta_training_regime(meta)
    if training_regime not in {canonical_training_regime(value) for value in spec.get("training_regimes", [])}:
        return False
    feature_sets = [str(value).strip() for value in spec.get("feature_sets", []) if str(value).strip()]
    if feature_sets:
        if str(meta.get("feature_set", "")).strip() not in set(feature_sets):
            return False
    elif str(meta.get("feature_set", "")).strip() != str(spec.get("feature_set", "")).strip():
        return False
    objective_names = [str(value).strip().lower() for value in spec.get("objective_names", []) if str(value).strip()]
    if objective_names:
        if str(meta.get("objective_name", "")).strip().lower() not in set(objective_names):
            return False
    elif str(meta.get("objective_name", "")).strip().lower() != str(spec.get("objective_name", "")).strip().lower():
        return False
    if not math.isclose(float(meta.get("sigma_true", 0.0)), float(spec.get("sigma_true", 0.0)), rel_tol=0.0, abs_tol=1e-12):
        return False
    scenario_pairs = spec.get("scenario_pairs", [])
    if scenario_pairs:
        allowed_pairs = {(float(item["sigma_bar"]), float(item["lambda_cost"])) for item in scenario_pairs}
        if (float(meta.get("sigma_bar", 0.0)), float(meta.get("lambda_cost", 0.0))) not in allowed_pairs:
            return False
    else:
        if float(meta.get("sigma_bar", 0.0)) not in {float(value) for value in spec.get("sigma_bars", [])}:
            return False
        if float(meta.get("lambda_cost", 0.0)) not in {float(value) for value in spec.get("lambda_costs", [])}:
            return False
    if int(meta.get("seed", -1)) not in {int(value) for value in spec.get("seeds", [])}:
        return False
    expected_sigmas = _coerce_robust_sigmas(spec.get("robust_sigmas", []))
    return _meta_robust_sigmas(meta) == _robust_sigmas_csv(effective_robust_sigmas(training_regime, expected_sigmas))


def _config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _float_slug(value: float) -> str:
    return format(float(value), ".12g")


def _feature_set_from_cfg(cfg: Mapping[str, Any]) -> str:
    return str(get(cfg, "features.feature_set", "B")).strip().upper()


def _objective_name_from_cfg(cfg: Mapping[str, Any]) -> str:
    return str(get(cfg, "objective.name", "cvar")).strip().lower().replace("-", "_")


def _coerce_robust_sigmas(raw: Any) -> tuple[float, ...]:
    if raw in ("", None):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(float(value) for value in raw)
    return tuple(float(token) for token in str(raw).replace("|", ",").split(",") if token.strip())


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"cannot interpret {raw!r} as bool")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
