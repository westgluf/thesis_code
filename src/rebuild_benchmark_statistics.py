from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import stats

from src.benchmark_repro import (
    benchmark_scenario_id,
    canonical_run_mode,
    canonical_training_regime,
    deep_hedge_method_name,
)
from src.config import load_yaml
from src.logging_utils import write_csv_rows
from src.paths import (
    aggregated_by_method_path,
    benchmark_spec_json_path,
    get_run_dir,
    paired_comparisons_path,
    scenario_summary_path,
    seed_level_metrics_path,
    summary_rows_csv_path,
    win_summary_path,
)


METRICS: tuple[str, ...] = (
    "mean_PL",
    "std_PL",
    "VaR_loss_0.95",
    "ES_loss_0.95",
    "VaR_loss_0.99",
    "ES_loss_0.99",
    "mean_turnover",
    "max_turnover",
    "total_turnover",
)
HIGHER_IS_BETTER = {"mean_PL"}
_SEED_KEY = "seed"
_BOOTSTRAP_RESAMPLES = 10_000
_ALLOWED_METHODS = {"bs_delta", "deep_hedge_oracle", "deep_hedge_robust"}
_SCENARIO_METADATA_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "campaign_id",
    "campaign_role",
    "training_regime",
    "feature_set",
    "objective_name",
    "sigma_true",
    "sigma_bar",
    "lambda_cost",
    "robust_sigmas",
)


def rebuild_benchmark_statistics(
    root_run_dir: str | Path,
    *,
    bootstrap_resamples: int = _BOOTSTRAP_RESAMPLES,
) -> dict[str, Path]:
    run_dir = Path(root_run_dir)
    rows = _load_summary_rows(summary_rows_csv_path(run_dir))
    deduped_rows = _deduplicate_seed_rows(rows)
    _audit_benchmark_rows(rows, deduped_rows, run_dir=run_dir)

    aggregated_rows = _build_aggregated_rows(deduped_rows)
    paired_rows = _build_paired_rows(deduped_rows, bootstrap_resamples=bootstrap_resamples)
    win_rows = _build_win_rows(paired_rows)
    scenario_rows = _build_scenario_summary_rows(aggregated_rows)

    write_csv_rows(seed_level_metrics_path(run_dir), _seed_level_header(), deduped_rows)
    write_csv_rows(aggregated_by_method_path(run_dir), _aggregated_header(), aggregated_rows)
    write_csv_rows(paired_comparisons_path(run_dir), _paired_header(), paired_rows)
    write_csv_rows(win_summary_path(run_dir), _win_header(), win_rows)
    write_csv_rows(scenario_summary_path(run_dir), _scenario_summary_header(), scenario_rows)

    return {
        "seed_level_metrics": seed_level_metrics_path(run_dir),
        "aggregated_by_method": aggregated_by_method_path(run_dir),
        "paired_comparisons": paired_comparisons_path(run_dir),
        "win_summary": win_summary_path(run_dir),
        "scenario_summary": scenario_summary_path(run_dir),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rebuild_benchmark_statistics")
    parser.add_argument(
        "--config",
        default=None,
        help="Base YAML config path. Defaults to GBM_CFG or configs/gbm_es95.yaml.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Root run directory. Defaults to the out_dir from the chosen config.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=_BOOTSTRAP_RESAMPLES,
        help="Number of paired bootstrap resamples for paired_comparisons.csv.",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(run_dir_arg=args.run_dir, cfg_path_arg=args.config)
    outputs = rebuild_benchmark_statistics(run_dir, bootstrap_resamples=int(args.bootstrap_resamples))
    print(f"Rebuilt benchmark statistics from: {summary_rows_csv_path(run_dir)}")
    print(f"seed_level_metrics: {outputs['seed_level_metrics']}")
    print(f"aggregated_by_method: {outputs['aggregated_by_method']}")
    print(f"paired_comparisons: {outputs['paired_comparisons']}")
    print(f"win_summary: {outputs['win_summary']}")
    print(f"scenario_summary: {outputs['scenario_summary']}")


def _resolve_run_dir(*, run_dir_arg: str | None, cfg_path_arg: str | None) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    cfg_path = cfg_path_arg or os.environ.get("GBM_CFG", "configs/gbm_es95.yaml")
    cfg = load_yaml(cfg_path)
    return get_run_dir(cfg)


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"summary rows not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _normalize_summary_row(raw)
            if row["is_benchmark_eligible"]:
                rows.append(row)
    return rows


def _normalize_summary_row(raw: Mapping[str, str]) -> dict[str, Any]:
    training_regime = _infer_training_regime(raw)
    run_mode = canonical_run_mode(str(raw.get("run_mode", "debug") or "debug"))
    robust_sigmas = _normalize_robust_sigmas(raw.get("robust_sigmas", ""), training_regime=training_regime)
    row = dict(raw)
    row["training_regime"] = training_regime
    row["run_mode"] = run_mode
    row["is_benchmark_eligible"] = _parse_bool(raw.get("is_benchmark_eligible", ""), default=run_mode == "benchmark")
    row["campaign_id"] = str(raw.get("campaign_id", "main_benchmark")).strip() or "main_benchmark"
    row["campaign_role"] = str(raw.get("campaign_role", "main")).strip().lower() or "main"
    row["method"] = str(raw.get("method", "")).strip()
    row["feature_set"] = str(raw.get("feature_set", "")).strip()
    row["objective_name"] = str(raw.get("objective_name", "")).strip().lower()
    row["robust_sigmas"] = robust_sigmas
    row[_SEED_KEY] = _parse_int(raw.get(_SEED_KEY, ""))
    row["sigma_true"] = _parse_float(raw.get("sigma_true", ""))
    row["sigma_bar"] = _parse_float(raw.get("sigma_bar", ""))
    row["lambda_cost"] = _parse_float(raw.get("lambda_cost", ""))
    for metric in METRICS:
        row[metric] = _parse_float(raw.get(metric, ""))
    row["scenario_id"] = _normalize_scenario_id(raw.get("scenario_id", ""), row)
    return row


def _deduplicate_seed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        normalized = dict(row)
        key = (
            normalized["scenario_id"],
            normalized.get("method", ""),
            normalized.get(_SEED_KEY, None),
        )
        current = latest_by_key.get(key)
        created_at = str(normalized.get("created_at_utc", ""))
        run_id = str(normalized.get("run_id", ""))
        if current is None:
            latest_by_key[key] = normalized
            continue
        current_key = (str(current.get("created_at_utc", "")), str(current.get("run_id", "")))
        if (created_at, run_id) >= current_key:
            latest_by_key[key] = normalized
    return sorted(
        latest_by_key.values(),
        key=lambda row: (str(row["scenario_id"]), str(row.get("method", "")), int(row.get(_SEED_KEY) or -1)),
    )


def _build_aggregated_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = _scenario_method_key(row)
        grouped.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        base = _scenario_metadata(group_rows[0])
        method = str(group_rows[0].get("method", ""))
        n_seeds = len(group_rows)
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            sample_stats = _sample_summary(values)
            output_rows.append(
                {
                    **base,
                    "method": method,
                    "metric": metric,
                    "better_direction": _better_direction(metric),
                    "n_seeds": n_seeds,
                    "mean": sample_stats["mean"],
                    "sd": sample_stats["sd"],
                    "se": sample_stats["se"],
                    "ci95_lo": sample_stats["ci95_lo"],
                    "ci95_hi": sample_stats["ci95_hi"],
                }
            )
    return output_rows


def _build_paired_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = _scenario_key(row)
        by_scenario.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key, scenario_rows in sorted(by_scenario.items(), key=lambda item: item[0]):
        base = _scenario_metadata(scenario_rows[0])
        training_regime = str(base["training_regime"])
        dh_method = deep_hedge_method_name(training_regime)
        bs_rows = {
            int(row[_SEED_KEY]): row
            for row in scenario_rows
            if row.get("method") == "bs_delta" and row.get(_SEED_KEY) is not None
        }
        dh_rows = {
            int(row[_SEED_KEY]): row
            for row in scenario_rows
            if row.get("method") == dh_method and row.get(_SEED_KEY) is not None
        }
        common_seeds = sorted(set(bs_rows) & set(dh_rows))
        if not common_seeds:
            continue

        for metric in METRICS:
            bs_values = np.asarray([float(bs_rows[seed][metric]) for seed in common_seeds], dtype=float)
            dh_values = np.asarray([float(dh_rows[seed][metric]) for seed in common_seeds], dtype=float)
            diffs = dh_values - bs_values
            sample_stats = _sample_summary(diffs)
            p_value = _paired_ttest_pvalue(diffs)
            boot_lo, boot_hi = _bootstrap_ci_mean(
                diffs,
                n_resamples=bootstrap_resamples,
                seed=_bootstrap_seed(base["scenario_id"], metric),
            )
            output_rows.append(
                {
                    **base,
                    "bs_method": "bs_delta",
                    "deep_hedge_method": dh_method,
                    "metric": metric,
                    "better_direction": _better_direction(metric),
                    "n_pairs": len(common_seeds),
                    "seed_list": "|".join(str(seed) for seed in common_seeds),
                    "mean_bs": float(np.mean(bs_values)),
                    "mean_dh": float(np.mean(dh_values)),
                    "mean_diff": sample_stats["mean"],
                    "sd_diff": sample_stats["sd"],
                    "se_diff": sample_stats["se"],
                    "ci95_lo": sample_stats["ci95_lo"],
                    "ci95_hi": sample_stats["ci95_hi"],
                    "p_value": p_value,
                    "bootstrap_ci95_lo": boot_lo,
                    "bootstrap_ci95_hi": boot_hi,
                }
            )
    return output_rows


def _build_win_rows(paired_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(paired_rows)
    by_scenario_metric = {(str(row["scenario_id"]), str(row["metric"])): row for row in rows}
    scenario_rows = {str(row["scenario_id"]): row for row in rows}

    output_rows: list[dict[str, Any]] = []
    for scenario_id, example in sorted(scenario_rows.items(), key=lambda item: item[0]):
        es95 = by_scenario_metric.get((scenario_id, "ES_loss_0.95"))
        es99 = by_scenario_metric.get((scenario_id, "ES_loss_0.99"))
        if es95 is None or es99 is None:
            continue

        primary_verdict = _verdict_from_pair(es95, metric="ES_loss_0.95")
        secondary_verdict = _verdict_from_pair(es99, metric="ES_loss_0.99")
        output_rows.append(
            {
                "scenario_id": scenario_id,
                "campaign_id": example["campaign_id"],
                "campaign_role": example["campaign_role"],
                "training_regime": example["training_regime"],
                "feature_set": example["feature_set"],
                "objective_name": example["objective_name"],
                "sigma_true": example["sigma_true"],
                "sigma_bar": example["sigma_bar"],
                "lambda_cost": example["lambda_cost"],
                "robust_sigmas": example["robust_sigmas"],
                "bs_method": example["bs_method"],
                "deep_hedge_method": example["deep_hedge_method"],
                "n_pairs": es95["n_pairs"],
                "primary_metric": "ES_loss_0.95",
                "primary_mean_diff": es95["mean_diff"],
                "primary_ci95_lo": es95["ci95_lo"],
                "primary_ci95_hi": es95["ci95_hi"],
                "primary_p_value": es95["p_value"],
                "primary_bootstrap_ci95_lo": es95["bootstrap_ci95_lo"],
                "primary_bootstrap_ci95_hi": es95["bootstrap_ci95_hi"],
                "primary_verdict": primary_verdict,
                "secondary_metric": "ES_loss_0.99",
                "secondary_mean_diff": es99["mean_diff"],
                "secondary_ci95_lo": es99["ci95_lo"],
                "secondary_ci95_hi": es99["ci95_hi"],
                "secondary_p_value": es99["p_value"],
                "secondary_bootstrap_ci95_lo": es99["bootstrap_ci95_lo"],
                "secondary_bootstrap_ci95_hi": es99["bootstrap_ci95_hi"],
                "secondary_verdict": secondary_verdict,
                "es99_agrees_with_primary": secondary_verdict == primary_verdict,
                "deep_hedge_wins_primary": primary_verdict == "deep_hedge_win",
            }
        )
    return output_rows


def _build_scenario_summary_rows(aggregated_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in aggregated_rows:
        key = (str(row["scenario_id"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for _, rows in sorted(grouped.items(), key=lambda item: item[0]):
        example = rows[0]
        out: dict[str, Any] = {
            **_scenario_metadata(example),
            "method": example["method"],
            "n_seeds": example["n_seeds"],
        }
        by_metric = {str(row["metric"]): row for row in rows}
        for metric in METRICS:
            metric_row = by_metric.get(metric, {})
            for stat_name in ("mean", "sd", "se", "ci95_lo", "ci95_hi"):
                out[f"{metric}_{stat_name}"] = metric_row.get(stat_name, "")
        output_rows.append(out)
    return output_rows


def _sample_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.shape[0])
    mean = float(np.mean(arr)) if n > 0 else math.nan
    if n < 2:
        return {
            "mean": mean,
            "sd": math.nan,
            "se": math.nan,
            "ci95_lo": math.nan,
            "ci95_hi": math.nan,
        }
    sd = float(np.std(arr, ddof=1))
    se = float(sd / math.sqrt(n))
    if math.isfinite(se):
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_half = t_crit * se
        ci95_lo = float(mean - ci_half)
        ci95_hi = float(mean + ci_half)
    else:
        ci95_lo = math.nan
        ci95_hi = math.nan
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_lo": ci95_lo,
        "ci95_hi": ci95_hi,
    }


def _paired_ttest_pvalue(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.shape[0])
    if n < 2:
        return math.nan
    if np.allclose(arr, arr[0]):
        return 1.0 if math.isclose(float(arr[0]), 0.0, abs_tol=1e-12) else 0.0
    with np.errstate(all="ignore"):
        return float(stats.ttest_1samp(arr, popmean=0.0, alternative="two-sided").pvalue)


def _bootstrap_ci_mean(diffs: np.ndarray, *, n_resamples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.shape[0])
    if n == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_resamples), dtype=float)
    for idx in range(int(n_resamples)):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(np.mean(arr[sample_idx]))
    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _bootstrap_seed(scenario_id: str, metric: str) -> int:
    payload = f"{scenario_id}::{metric}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _audit_benchmark_rows(
    rows: list[Mapping[str, Any]],
    deduped_rows: list[Mapping[str, Any]],
    *,
    run_dir: Path,
) -> None:
    _audit_row_integrity(rows)
    _audit_unique_seed_keys(deduped_rows)
    spec = _load_benchmark_spec(run_dir)
    if rows and spec is not None and _parse_bool(spec.get("claims_grid_experiment", False), default=False):
        _audit_grid_coverage(deduped_rows, spec)


def _audit_row_integrity(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        for field in ("scenario_id", "campaign_id", "campaign_role", "method", "training_regime", "feature_set", "objective_name"):
            if str(row.get(field, "")).strip() == "":
                raise AssertionError(f"benchmark summary row missing required field {field!r}: {row}")
        if row.get(_SEED_KEY) is None:
            raise AssertionError(f"benchmark summary row missing seed: {row}")
        if row.get("run_mode") != "benchmark" or not bool(row.get("is_benchmark_eligible")):
            raise AssertionError(f"benchmark aggregate contains non-benchmark row: {row}")
        method = str(row.get("method", ""))
        if method not in _ALLOWED_METHODS:
            raise AssertionError(f"unknown method label {method!r} in benchmark row")
        training_regime = str(row.get("training_regime", ""))
        expected_dh = deep_hedge_method_name(training_regime)
        if method not in {"bs_delta", expected_dh}:
            raise AssertionError(
                "training_regime and method labeling disagree "
                f"(training_regime={training_regime!r}, method={method!r})"
            )
        robust_sigmas = str(row.get("robust_sigmas", "") or "")
        if training_regime == "oracle" and robust_sigmas:
            raise AssertionError(f"oracle row unexpectedly carries robust_sigmas={robust_sigmas!r}")
        if training_regime == "robust" and not robust_sigmas:
            raise AssertionError("robust row is missing robust_sigmas metadata")
        for field in ("sigma_true", "sigma_bar", "lambda_cost"):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value):
                raise AssertionError(f"benchmark row has non-finite {field}: {row}")


def _audit_unique_seed_keys(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (
            str(row["scenario_id"]),
            str(row["method"]),
            int(row[_SEED_KEY]),
        )
        if key in seen:
            raise AssertionError(f"duplicate (scenario_id, method, seed) row survived deduplication: {key}")
        seen.add(key)


def _load_benchmark_spec(run_dir: Path) -> dict[str, Any] | None:
    path = benchmark_spec_json_path(run_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _audit_grid_coverage(rows: list[Mapping[str, Any]], spec: Mapping[str, Any]) -> None:
    campaign_id = str(spec.get("campaign_id", "")).strip()
    campaign_role = str(spec.get("campaign_role", "")).strip().lower()
    feature_sets = [str(value).strip() for value in spec.get("feature_sets", []) if str(value).strip()]
    if not feature_sets:
        feature_set = str(spec.get("feature_set", "")).strip()
        feature_sets = [feature_set] if feature_set else []
    objective_names = [str(value).strip().lower() for value in spec.get("objective_names", []) if str(value).strip()]
    if not objective_names:
        objective_name = str(spec.get("objective_name", "")).strip().lower()
        objective_names = [objective_name] if objective_name else []
    sigma_true = float(spec.get("sigma_true", math.nan))
    robust_sigmas = tuple(float(value) for value in spec.get("robust_sigmas", []))
    scenario_pairs = [(float(item["sigma_bar"]), float(item["lambda_cost"])) for item in spec.get("scenario_pairs", [])]
    sigma_bars = [float(value) for value in spec.get("sigma_bars", [])]
    lambda_costs = [float(value) for value in spec.get("lambda_costs", [])]
    seeds = [int(value) for value in spec.get("seeds", [])]
    training_regimes = [canonical_training_regime(value) for value in spec.get("training_regimes", [])]

    if not rows:
        raise AssertionError("benchmark spec claims a grid experiment but no benchmark-eligible rows were found")

    for row in rows:
        if campaign_id and str(row["campaign_id"]).strip() != campaign_id:
            raise AssertionError(f"benchmark row campaign_id={row['campaign_id']!r} does not match spec={campaign_id!r}")
        if campaign_role and str(row["campaign_role"]).strip().lower() != campaign_role:
            raise AssertionError(f"benchmark row campaign_role={row['campaign_role']!r} does not match spec={campaign_role!r}")
        if str(row["feature_set"]).strip() not in set(feature_sets):
            raise AssertionError(f"benchmark row feature_set={row['feature_set']!r} is outside declared grid")
        if str(row["objective_name"]).strip().lower() not in set(objective_names):
            raise AssertionError(
                f"benchmark row objective_name={row['objective_name']!r} is outside declared grid"
            )
        if not math.isclose(float(row["sigma_true"]), sigma_true, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"benchmark row sigma_true={row['sigma_true']!r} does not match spec={sigma_true!r}"
            )
        pair = (float(row["sigma_bar"]), float(row["lambda_cost"]))
        if scenario_pairs:
            if pair not in set(scenario_pairs):
                raise AssertionError(
                    "benchmark row (sigma_bar, lambda_cost)="
                    f"{pair!r} is outside declared scenario_pairs"
                )
        else:
            if float(row["sigma_bar"]) not in sigma_bars:
                raise AssertionError(f"benchmark row sigma_bar={row['sigma_bar']!r} is outside declared grid")
            if float(row["lambda_cost"]) not in lambda_costs:
                raise AssertionError(f"benchmark row lambda_cost={row['lambda_cost']!r} is outside declared grid")
        if int(row["seed"]) not in seeds:
            raise AssertionError(f"benchmark row seed={row['seed']!r} is outside declared grid")
        if str(row["training_regime"]) not in training_regimes:
            raise AssertionError(
                f"benchmark row training_regime={row['training_regime']!r} is outside declared grid"
            )

    observed = {(str(row["scenario_id"]), str(row["method"]), int(row["seed"])) for row in rows}
    missing: list[str] = []
    for training_regime in training_regimes:
        scenario_sigmas = robust_sigmas if training_regime == "robust" else ()
        pair_iter = scenario_pairs if scenario_pairs else [(sigma_bar, lambda_cost) for sigma_bar in sigma_bars for lambda_cost in lambda_costs]
        for feature_set in feature_sets:
            for objective_name in objective_names:
                for sigma_bar, lambda_cost in pair_iter:
                    scenario_id = benchmark_scenario_id(
                        training_regime=training_regime,
                        feature_set=feature_set,
                        objective_name=objective_name,
                        sigma_true=sigma_true,
                        sigma_bar=sigma_bar,
                        lambda_cost=lambda_cost,
                        robust_sigmas=scenario_sigmas,
                    )
                    for seed in seeds:
                        for method in ("bs_delta", deep_hedge_method_name(training_regime)):
                            key = (scenario_id, method, seed)
                            if key not in observed:
                                missing.append(
                                    f"scenario_id={scenario_id}, method={method}, seed={seed}"
                                )
    if missing:
        sample = "; ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8} more)"
        raise AssertionError(
            "benchmark grid is incomplete for the declared scenario dimensions: "
            f"{sample}{extra}"
        )


def _scenario_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in _SCENARIO_METADATA_FIELDS}


def _normalize_scenario_id(raw: str, row: Mapping[str, Any]) -> str:
    expected = benchmark_scenario_id(
        training_regime=str(row.get("training_regime", "")),
        feature_set=str(row.get("feature_set", "")).strip(),
        objective_name=str(row.get("objective_name", "")).strip().lower(),
        sigma_true=float(row.get("sigma_true", math.nan)),
        sigma_bar=float(row.get("sigma_bar", math.nan)),
        lambda_cost=float(row.get("lambda_cost", math.nan)),
        robust_sigmas=_parse_robust_sigmas(str(row.get("robust_sigmas", "") or "")),
    )
    text = str(raw).strip()
    if text and text != expected:
        raise AssertionError(f"scenario_id mismatch: file has {text!r}, expected {expected!r}")
    return expected


def _scenario_key(row: Mapping[str, Any]) -> str:
    return str(row["scenario_id"])


def _scenario_method_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["scenario_id"]), str(row.get("method", "")))


def _better_direction(metric: str) -> str:
    return "higher" if metric in HIGHER_IS_BETTER else "lower"


def _verdict_from_pair(row: Mapping[str, Any], *, metric: str) -> str:
    lower_is_better = metric not in HIGHER_IS_BETTER
    mean_diff = float(row.get("mean_diff", math.nan))
    ci95_lo = float(row.get("ci95_lo", math.nan))
    ci95_hi = float(row.get("ci95_hi", math.nan))
    if not (math.isfinite(mean_diff) and math.isfinite(ci95_lo) and math.isfinite(ci95_hi)):
        return "no_clear_difference"
    if lower_is_better:
        if mean_diff < 0.0 and ci95_hi < 0.0:
            return "deep_hedge_win"
        if mean_diff > 0.0 and ci95_lo > 0.0:
            return "bs_win"
        return "no_clear_difference"
    if mean_diff > 0.0 and ci95_lo > 0.0:
        return "deep_hedge_win"
    if mean_diff < 0.0 and ci95_hi < 0.0:
        return "bs_win"
    return "no_clear_difference"


def _infer_training_regime(raw: Mapping[str, Any]) -> str:
    text = str(raw.get("training_regime", "")).strip()
    if text:
        return canonical_training_regime(text)
    method = str(raw.get("method", "")).strip().lower()
    if method.startswith("deep_hedge_"):
        return canonical_training_regime(method.removeprefix("deep_hedge_"))
    return "oracle"


def _normalize_robust_sigmas(raw: Any, *, training_regime: str) -> str:
    values = _parse_robust_sigmas(raw)
    if canonical_training_regime(training_regime) != "robust":
        return ""
    return "|".join(_fmt_float(value) for value in values)


def _parse_robust_sigmas(raw: Any) -> tuple[float, ...]:
    text = str(raw).replace("|", ",")
    if raw in ("", None) or text.strip() == "":
        return ()
    return tuple(float(token) for token in text.split(",") if token.strip())


def _parse_float(raw: str) -> float:
    text = str(raw).strip()
    if text == "":
        return math.nan
    return float(text)


def _parse_int(raw: str) -> int | None:
    text = str(raw).strip()
    if text == "":
        return None
    return int(text)


def _parse_bool(raw: Any, *, default: bool) -> bool:
    text = str(raw).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"cannot interpret {raw!r} as bool")


def _fmt_float(value: Any) -> str:
    val = float(value)
    if math.isnan(val):
        return "nan"
    return format(val, ".12g")


def _seed_level_header() -> tuple[str, ...]:
    return (
        "scenario_id",
        "campaign_id",
        "campaign_role",
        "run_id",
        "created_at_utc",
        "status",
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


def _aggregated_header() -> tuple[str, ...]:
    return (
        "scenario_id",
        "campaign_id",
        "campaign_role",
        "training_regime",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "robust_sigmas",
        "method",
        "metric",
        "better_direction",
        "n_seeds",
        "mean",
        "sd",
        "se",
        "ci95_lo",
        "ci95_hi",
    )


def _paired_header() -> tuple[str, ...]:
    return (
        "scenario_id",
        "campaign_id",
        "campaign_role",
        "training_regime",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "robust_sigmas",
        "bs_method",
        "deep_hedge_method",
        "metric",
        "better_direction",
        "n_pairs",
        "seed_list",
        "mean_bs",
        "mean_dh",
        "mean_diff",
        "sd_diff",
        "se_diff",
        "ci95_lo",
        "ci95_hi",
        "p_value",
        "bootstrap_ci95_lo",
        "bootstrap_ci95_hi",
    )


def _win_header() -> tuple[str, ...]:
    return (
        "scenario_id",
        "campaign_id",
        "campaign_role",
        "training_regime",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "robust_sigmas",
        "bs_method",
        "deep_hedge_method",
        "n_pairs",
        "primary_metric",
        "primary_mean_diff",
        "primary_ci95_lo",
        "primary_ci95_hi",
        "primary_p_value",
        "primary_bootstrap_ci95_lo",
        "primary_bootstrap_ci95_hi",
        "primary_verdict",
        "secondary_metric",
        "secondary_mean_diff",
        "secondary_ci95_lo",
        "secondary_ci95_hi",
        "secondary_p_value",
        "secondary_bootstrap_ci95_lo",
        "secondary_bootstrap_ci95_hi",
        "secondary_verdict",
        "es99_agrees_with_primary",
        "deep_hedge_wins_primary",
    )


def _scenario_summary_header() -> tuple[str, ...]:
    dynamic_fields = []
    for metric in METRICS:
        for stat_name in ("mean", "sd", "se", "ci95_lo", "ci95_hi"):
            dynamic_fields.append(f"{metric}_{stat_name}")
    return (
        "scenario_id",
        "campaign_id",
        "campaign_role",
        "training_regime",
        "feature_set",
        "objective_name",
        "sigma_true",
        "sigma_bar",
        "lambda_cost",
        "robust_sigmas",
        "method",
        "n_seeds",
        *dynamic_fields,
    )


if __name__ == "__main__":
    main()
