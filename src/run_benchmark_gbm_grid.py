from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.benchmark_repro import benchmark_run_id, canonical_training_regime
from src.config import get, load_yaml
from src.logging_utils import write_json_file
from src.paths import benchmark_spec_json_path, get_run_dir, manifest_runs_csv_path, summary_rows_csv_path
from src.rebuild_benchmark_statistics import rebuild_benchmark_statistics
from src.train_deephedge_gbm import run_from_cfg


ObjectiveSpec = dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config or os.environ.get("GBM_CFG", "configs/gbm_es95.yaml"))
    base_cfg = load_yaml(cfg_path)
    if args.out_dir:
        base_cfg["out_dir"] = str(args.out_dir)

    scenario_spec = _scenario_spec_from_args(args, base_cfg)
    run_cfgs = _build_run_cfgs(
        base_cfg,
        scenario_spec,
        quick=bool(args.quick),
        smoke_subset=bool(args.smoke_subset),
        campaign_id=str(args.campaign_id).strip(),
        campaign_role=str(args.campaign_role).strip().lower(),
    )

    if args.max_runs is not None:
        run_cfgs = run_cfgs[: max(int(args.max_runs), 0)]

    total_runs = len(run_cfgs)
    run_dir = get_run_dir(base_cfg)
    _assert_campaign_root_is_safe(
        run_dir=run_dir,
        scenario_spec=scenario_spec,
        total_runs=total_runs,
        quick=bool(args.quick),
        smoke_subset=bool(args.smoke_subset),
        max_runs=args.max_runs,
        campaign_id=str(args.campaign_id).strip(),
        campaign_role=str(args.campaign_role).strip().lower(),
        sigma_true=float(get(base_cfg, "data.sigma_true", 0.0)),
        robust_sigmas=[float(value) for value in get(base_cfg, "benchmark.robust_sigmas", [])],
    )

    if not (args.quick or args.smoke_subset):
        _write_benchmark_spec(
            run_dir=run_dir,
            scenario_spec=scenario_spec,
            total_runs=total_runs,
            quick=bool(args.quick),
            smoke_subset=bool(args.smoke_subset),
            max_runs=args.max_runs,
            campaign_id=str(args.campaign_id).strip(),
            campaign_role=str(args.campaign_role).strip().lower(),
            sigma_true=float(get(base_cfg, "data.sigma_true", 0.0)),
            robust_sigmas=[float(value) for value in get(base_cfg, "benchmark.robust_sigmas", [])],
        )

    print(f"Grid runner config: {cfg_path}")
    print(f"Output root: {run_dir}")
    print(f"Campaign: {args.campaign_id} ({args.campaign_role})")
    print(f"Planned runs: {total_runs}")

    failures: list[dict[str, Any]] = []
    for run_idx, cfg in enumerate(run_cfgs, start=1):
        run_id = benchmark_run_id(cfg)
        regime = get(cfg, "benchmark.training_regime", "oracle")
        sigma_bar = float(get(cfg, "data.sigma_bar", 0.0))
        lam_cost = float(get(cfg, "data.lam_cost", 0.0))
        seed = int(get(cfg, "data.seed", 0))
        feature_set = str(get(cfg, "features.feature_set", "B")).strip()
        objective_name = str(get(cfg, "objective.name", "cvar")).strip().lower()
        print(
            f"[{run_idx}/{total_runs}] run_id={run_id} "
            f"regime={regime} feature_set={feature_set} objective={objective_name} "
            f"sigma_bar={sigma_bar:.6g} lambda_cost={lam_cost:.6g} seed={seed}"
        )
        try:
            run_from_cfg(cfg)
        except Exception as exc:
            failures.append(
                {
                    "run_id": run_id,
                    "training_regime": regime,
                    "feature_set": feature_set,
                    "objective_name": objective_name,
                    "sigma_bar": sigma_bar,
                    "lambda_cost": lam_cost,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"FAILED: {run_id}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                break

    rebuild_benchmark_statistics(run_dir)
    print(f"Completed runs: {total_runs - len(failures)} / {total_runs}")
    print(f"Manifest: {manifest_runs_csv_path(run_dir)}")
    print(f"Summary rows: {summary_rows_csv_path(run_dir)}")
    if failures:
        print("Failed scenarios:")
        for failure in failures:
            print(json.dumps(failure, sort_keys=True))
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.run_benchmark_gbm_grid")
    parser.add_argument("--config", default=None, help="Base YAML config path. Defaults to GBM_CFG or configs/gbm_es95.yaml.")
    parser.add_argument("--out-dir", default=None, help="Output root for this campaign.")
    parser.add_argument("--campaign-id", default="main_benchmark", help="Stable campaign identifier.")
    parser.add_argument("--campaign-role", default="main", help="Campaign role label, e.g. main or supplementary.")
    parser.add_argument("--sigma-bars", default="0.10,0.15,0.20,0.25,0.30", help="Comma-separated sigma_bar values.")
    parser.add_argument("--lambda-costs", default="0,1e-4,5e-4,1e-3", help="Comma-separated lambda_cost values.")
    parser.add_argument(
        "--scenario-pairs",
        default=None,
        help="Optional comma-separated sigma_bar@lambda_cost pairs. Overrides the full sigma/lambda cartesian product.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--training-regimes",
        default="oracle,robust",
        help="Comma-separated deep hedging training regimes.",
    )
    parser.add_argument(
        "--feature-sets",
        default=None,
        help="Comma-separated feature sets. Defaults to the base config feature set.",
    )
    parser.add_argument(
        "--objective-specs",
        default=None,
        help="Comma-separated objective specs such as cvar:a0.95 or entropic:g1.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on the number of planned runs.")
    parser.add_argument(
        "--smoke-subset",
        action="store_true",
        help="Use a minimal scenario slice: sigma_bar=0.20, lambda_cost=0, seed=0, both training regimes.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce train/test sizes and epochs for a fast smoke-sized run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed run instead of continuing through the campaign.",
    )
    return parser


def _scenario_spec_from_args(args: argparse.Namespace, base_cfg: Mapping[str, Any]) -> dict[str, Any]:
    default_feature_set = str(get(base_cfg, "features.feature_set", "B")).strip().upper()
    default_objective = _default_objective_spec(base_cfg)
    if args.smoke_subset:
        return {
            "scenario_pairs": [(0.20, 0.0)],
            "sigma_bars": [0.20],
            "lambda_costs": [0.0],
            "seeds": [0],
            "training_regimes": ["oracle", "robust"],
            "feature_sets": [default_feature_set],
            "objective_specs": [default_objective],
        }

    scenario_pairs = _parse_scenario_pairs(args.scenario_pairs) if args.scenario_pairs else None
    sigma_bars = sorted({pair[0] for pair in scenario_pairs}, key=float) if scenario_pairs else _parse_float_csv(args.sigma_bars)
    lambda_costs = sorted({pair[1] for pair in scenario_pairs}, key=float) if scenario_pairs else _parse_float_csv(args.lambda_costs)
    return {
        "scenario_pairs": scenario_pairs,
        "sigma_bars": sigma_bars,
        "lambda_costs": lambda_costs,
        "seeds": _parse_int_csv(args.seeds),
        "training_regimes": [canonical_training_regime(name) for name in _parse_str_csv(args.training_regimes)],
        "feature_sets": _parse_feature_sets(args.feature_sets, default_feature_set),
        "objective_specs": _parse_objective_specs(args.objective_specs, default_objective),
    }


def _build_run_cfgs(
    base_cfg: dict[str, Any],
    scenario_spec: dict[str, Any],
    *,
    quick: bool,
    smoke_subset: bool,
    campaign_id: str,
    campaign_role: str,
) -> list[dict[str, Any]]:
    run_mode = "smoke" if (quick or smoke_subset) else "benchmark"
    is_benchmark_eligible = run_mode == "benchmark"
    pair_iter = scenario_spec["scenario_pairs"] or [
        (sigma_bar, lambda_cost)
        for sigma_bar in scenario_spec["sigma_bars"]
        for lambda_cost in scenario_spec["lambda_costs"]
    ]
    cfgs: list[dict[str, Any]] = []
    for feature_set in scenario_spec["feature_sets"]:
        for objective_spec in scenario_spec["objective_specs"]:
            for sigma_bar, lambda_cost in pair_iter:
                for seed in scenario_spec["seeds"]:
                    for training_regime in scenario_spec["training_regimes"]:
                        cfg = copy.deepcopy(base_cfg)
                        _set_dotted(cfg, "features.feature_set", str(feature_set).strip().upper())
                        _apply_objective_spec(cfg, objective_spec)
                        _set_dotted(cfg, "data.sigma_bar", float(sigma_bar))
                        _set_dotted(cfg, "data.lam_cost", float(lambda_cost))
                        _set_dotted(cfg, "data.seed", int(seed))
                        _set_dotted(cfg, "benchmark.training_regime", canonical_training_regime(training_regime))
                        _set_dotted(cfg, "benchmark.run_mode", run_mode)
                        _set_dotted(cfg, "benchmark.is_benchmark_eligible", is_benchmark_eligible)
                        _set_dotted(cfg, "benchmark.campaign_id", campaign_id)
                        _set_dotted(cfg, "benchmark.campaign_role", campaign_role)
                        if quick:
                            _apply_quick_overrides(cfg)
                        cfgs.append(cfg)
    return cfgs


def _apply_objective_spec(cfg: dict[str, Any], objective_spec: ObjectiveSpec) -> None:
    name = str(objective_spec["name"]).strip().lower()
    _set_dotted(cfg, "objective.name", name)
    if name == "cvar":
        _set_dotted(cfg, "objective.alpha", float(objective_spec["alpha"]))
    elif name == "entropic":
        _set_dotted(cfg, "objective.gamma", float(objective_spec["gamma"]))
    elif name == "mean_variance":
        _set_dotted(cfg, "objective.lambda_mv", float(objective_spec["lambda_mv"]))


def _apply_quick_overrides(cfg: dict[str, Any]) -> None:
    n_train = min(int(get(cfg, "data.N_train", 5000)), 256)
    n_val = min(int(get(cfg, "data.N_val", 1000)), 64)
    n_test = min(int(get(cfg, "data.N_test", 2000)), 128)
    _set_dotted(cfg, "data.N_train", n_train)
    _set_dotted(cfg, "data.N_val", n_val)
    _set_dotted(cfg, "data.N_test", n_test)
    _set_dotted(cfg, "train.epochs", min(int(get(cfg, "train.epochs", 60)), 2))
    _set_dotted(cfg, "train.patience", min(int(get(cfg, "train.patience", 10)), 2))
    _set_dotted(cfg, "train.batch_size", min(int(get(cfg, "train.batch_size", 2048)), n_train))


def _assert_campaign_root_is_safe(
    *,
    run_dir: Path,
    scenario_spec: Mapping[str, Any],
    total_runs: int,
    quick: bool,
    smoke_subset: bool,
    max_runs: int | None,
    campaign_id: str,
    campaign_role: str,
    sigma_true: float,
    robust_sigmas: list[float],
) -> None:
    spec_path = benchmark_spec_json_path(run_dir)
    if not spec_path.exists():
        return
    existing = json.loads(spec_path.read_text(encoding="utf-8"))
    planned = _benchmark_spec_payload(
        scenario_spec=scenario_spec,
        total_runs=total_runs,
        quick=quick,
        smoke_subset=smoke_subset,
        max_runs=max_runs,
        campaign_id=campaign_id,
        campaign_role=campaign_role,
        sigma_true=sigma_true,
        robust_sigmas=robust_sigmas,
    )
    if existing != planned:
        raise RuntimeError(
            "refusing to reuse a benchmark root with a different benchmark_spec.json; "
            f"existing={spec_path}. Use a different --out-dir to keep campaigns isolated."
        )


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _parse_float_csv(raw: str) -> list[float]:
    return [float(token) for token in _parse_str_csv(raw)]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(token) for token in _parse_str_csv(raw)]


def _parse_str_csv(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _parse_feature_sets(raw: str | None, default_feature_set: str) -> list[str]:
    if raw is None:
        return [default_feature_set]
    return [token.strip().upper() for token in _parse_str_csv(raw)]


def _parse_scenario_pairs(raw: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for token in _parse_str_csv(raw):
        if "@" not in token:
            raise ValueError(f"scenario pair {token!r} must have the form sigma_bar@lambda_cost")
        sigma_text, lambda_text = token.split("@", 1)
        pairs.append((float(sigma_text), float(lambda_text)))
    return pairs


def _default_objective_spec(cfg: Mapping[str, Any]) -> ObjectiveSpec:
    name = str(get(cfg, "objective.name", "cvar")).strip().lower()
    if name == "cvar":
        alpha = float(get(cfg, "objective.alpha", 0.95))
        return {"name": "cvar", "alpha": alpha, "label": f"cvar_a{_float_slug(alpha)}"}
    if name == "entropic":
        gamma = float(get(cfg, "objective.gamma", 1.0))
        return {"name": "entropic", "gamma": gamma, "label": f"entropic_g{_float_slug(gamma)}"}
    lambda_mv = float(get(cfg, "objective.lambda_mv", 1.0))
    return {"name": "mean_variance", "lambda_mv": lambda_mv, "label": f"mean_variance_l{_float_slug(lambda_mv)}"}


def _parse_objective_specs(raw: str | None, default_objective: ObjectiveSpec) -> list[ObjectiveSpec]:
    if raw is None:
        return [default_objective]
    return [_parse_objective_spec(token) for token in _parse_str_csv(raw)]


def _parse_objective_spec(token: str) -> ObjectiveSpec:
    text = token.strip().lower().replace("-", "_")
    name, _, param = text.partition(":")
    if name == "cvar":
        alpha = _parse_objective_param(param, key_aliases=("a", "alpha"), default=0.95)
        return {"name": "cvar", "alpha": alpha, "label": f"cvar_a{_float_slug(alpha)}"}
    if name == "entropic":
        gamma = _parse_objective_param(param, key_aliases=("g", "gamma"), default=1.0)
        return {"name": "entropic", "gamma": gamma, "label": f"entropic_g{_float_slug(gamma)}"}
    if name == "mean_variance":
        lambda_mv = _parse_objective_param(param, key_aliases=("l", "lambda_mv"), default=1.0)
        return {"name": "mean_variance", "lambda_mv": lambda_mv, "label": f"mean_variance_l{_float_slug(lambda_mv)}"}
    raise ValueError(f"unknown objective spec {token!r}")


def _parse_objective_param(raw: str, *, key_aliases: tuple[str, ...], default: float) -> float:
    text = str(raw).strip().lower()
    if not text:
        return float(default)
    for alias in key_aliases:
        if text.startswith(alias + "="):
            return float(text.split("=", 1)[1])
        if text.startswith(alias):
            return float(text[len(alias) :])
    return float(text)


def _write_benchmark_spec(
    *,
    run_dir: Path,
    scenario_spec: Mapping[str, Any],
    total_runs: int,
    quick: bool,
    smoke_subset: bool,
    max_runs: int | None,
    campaign_id: str,
    campaign_role: str,
    sigma_true: float,
    robust_sigmas: list[float],
) -> None:
    payload = _benchmark_spec_payload(
        scenario_spec=scenario_spec,
        total_runs=total_runs,
        quick=quick,
        smoke_subset=smoke_subset,
        max_runs=max_runs,
        campaign_id=campaign_id,
        campaign_role=campaign_role,
        sigma_true=sigma_true,
        robust_sigmas=robust_sigmas,
    )
    write_json_file(benchmark_spec_json_path(run_dir), payload, sort_keys=True)


def _benchmark_spec_payload(
    *,
    scenario_spec: Mapping[str, Any],
    total_runs: int,
    quick: bool,
    smoke_subset: bool,
    max_runs: int | None,
    campaign_id: str,
    campaign_role: str,
    sigma_true: float,
    robust_sigmas: list[float],
) -> dict[str, Any]:
    claims_grid_experiment = not quick and not smoke_subset and max_runs is None
    objective_names = sorted({str(spec["name"]).strip().lower() for spec in scenario_spec["objective_specs"]})
    payload = {
        "benchmark_name": "section_6_2_gbm",
        "campaign_id": campaign_id,
        "campaign_role": campaign_role,
        "feature_set": str(scenario_spec["feature_sets"][0]).strip(),
        "feature_sets": [str(value).strip() for value in scenario_spec["feature_sets"]],
        "objective_name": objective_names[0],
        "objective_names": objective_names,
        "objective_specs": [dict(spec) for spec in scenario_spec["objective_specs"]],
        "sigma_true": float(sigma_true),
        "robust_sigmas": [float(value) for value in robust_sigmas],
        "sigma_bars": [float(value) for value in scenario_spec["sigma_bars"]],
        "lambda_costs": [float(value) for value in scenario_spec["lambda_costs"]],
        "scenario_pairs": [
            {"sigma_bar": float(sigma_bar), "lambda_cost": float(lambda_cost)}
            for sigma_bar, lambda_cost in (scenario_spec["scenario_pairs"] or [])
        ],
        "seeds": [int(value) for value in scenario_spec["seeds"]],
        "training_regimes": [canonical_training_regime(value) for value in scenario_spec["training_regimes"]],
        "planned_runs": int(total_runs),
        "run_mode": "smoke" if (quick or smoke_subset) else "benchmark",
        "is_benchmark_eligible": not (quick or smoke_subset),
        "claims_grid_experiment": claims_grid_experiment,
    }
    return payload


def _float_slug(value: float) -> str:
    return format(float(value), ".12g")


if __name__ == "__main__":
    raise SystemExit(main())
