"""Eval-only benchmark runner for extra sigma_bar values.

Reuses trained checkpoints from a prior grid run (e.g. sigma_bar=0.20)
to evaluate BS-delta at additional sigma_bar values without retraining.

Usage:
    python -m src.run_benchmark_eval_only \
        --config configs/gbm_benchmark.yaml \
        --sigma-bars 0.10,0.15,0.25,0.30 \
        --source-sigma-bar 0.20 \
        --lambda-costs 0,1e-4,5e-4,1e-3 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --training-regimes oracle,robust \
        --campaign-id main_benchmark \
        --campaign-role main
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.benchmark_repro import (
    benchmark_run_id,
    canonical_training_regime,
    finalize_benchmark_run,
    prepare_benchmark_run,
    robust_sigmas_from_cfg,
)
from src.bs import bs_call_price_discounted
from src.config import get, load_yaml
from src.costs_and_pl import pl_paths_proportional_costs, turnover_paths
from src.deep_hedging_model import MLPHedge
from src.eval import save_eval_artifacts
from src.hedge_core import compute_pl_torch, rollout_strategy
from src.logging_utils import write_run_config
from src.metrics import summary_metrics
from src.paths import (
    benchmark_run_dir,
    best_state_path,
    feature_norm_path,
    get_run_dir,
    run_cfg_path,
)
from src.rebuild_benchmark_statistics import rebuild_benchmark_statistics
from src.strategies_delta import bs_delta_strategy_paths
from src.world_gbm import (
    canonical_feature_set,
    make_gbm_dataset,
    make_gbm_robust_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config or os.environ.get("GBM_CFG", "configs/gbm_es95.yaml"))
    base_cfg = load_yaml(cfg_path)

    sigma_bars = _parse_float_csv(args.sigma_bars)
    source_sigma_bar = float(args.source_sigma_bar)
    lambda_costs = _parse_float_csv(args.lambda_costs)
    seeds = _parse_int_csv(args.seeds)
    training_regimes = [canonical_training_regime(r) for r in _parse_str_csv(args.training_regimes)]
    campaign_id = str(args.campaign_id).strip()
    campaign_role = str(args.campaign_role).strip().lower()

    run_dir = get_run_dir(base_cfg)

    eval_configs = []
    for seed in seeds:
        for lam_cost in lambda_costs:
            for regime in training_regimes:
                for sigma_bar in sigma_bars:
                    eval_configs.append({
                        "seed": seed,
                        "lambda_cost": lam_cost,
                        "training_regime": regime,
                        "sigma_bar": sigma_bar,
                    })

    total = len(eval_configs)
    print(f"Eval-only runner: {total} evaluations planned")
    print(f"Source sigma_bar: {source_sigma_bar}")
    print(f"Target sigma_bars: {sigma_bars}")
    print(f"Output root: {run_dir}")

    failures: list[dict[str, Any]] = []
    for idx, ec in enumerate(eval_configs, start=1):
        print(
            f"[{idx}/{total}] eval-only "
            f"regime={ec['training_regime']} sigma_bar={ec['sigma_bar']:.6g} "
            f"lambda_cost={ec['lambda_cost']:.6g} seed={ec['seed']}"
        )
        try:
            _run_eval_only(
                base_cfg=base_cfg,
                run_dir=run_dir,
                source_sigma_bar=source_sigma_bar,
                target_sigma_bar=ec["sigma_bar"],
                seed=ec["seed"],
                lambda_cost=ec["lambda_cost"],
                training_regime=ec["training_regime"],
                campaign_id=campaign_id,
                campaign_role=campaign_role,
            )
        except Exception as exc:
            failures.append({
                "seed": ec["seed"],
                "sigma_bar": ec["sigma_bar"],
                "lambda_cost": ec["lambda_cost"],
                "training_regime": ec["training_regime"],
                "error": str(exc),
            })
            print(f"  FAILED: {exc}")
            if args.fail_fast:
                break

    rebuild_benchmark_statistics(run_dir)

    print(f"\nCompleted: {total - len(failures)} / {total}")
    if failures:
        print("Failed evaluations:")
        for f in failures:
            print(json.dumps(f, sort_keys=True))
        return 2
    return 0


def _run_eval_only(
    *,
    base_cfg: dict[str, Any],
    run_dir: Path,
    source_sigma_bar: float,
    target_sigma_bar: float,
    seed: int,
    lambda_cost: float,
    training_regime: str,
    campaign_id: str,
    campaign_role: str,
) -> None:
    # Build source config to find checkpoint directory
    source_cfg = _build_cfg(
        base_cfg,
        sigma_bar=source_sigma_bar,
        seed=seed,
        lambda_cost=lambda_cost,
        training_regime=training_regime,
        campaign_id=campaign_id,
        campaign_role=campaign_role,
    )
    source_run_id = benchmark_run_id(source_cfg)
    source_dir = benchmark_run_dir(run_dir, source_run_id)

    checkpoint_path = best_state_path(source_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Run training grid at sigma_bar={source_sigma_bar} first."
        )

    # Build target config for the new sigma_bar evaluation
    target_cfg = _build_cfg(
        base_cfg,
        sigma_bar=target_sigma_bar,
        seed=seed,
        lambda_cost=lambda_cost,
        training_regime=training_regime,
        campaign_id=campaign_id,
        campaign_role=campaign_role,
    )

    device = str(get(base_cfg, "device", "cpu"))
    S0 = float(get(base_cfg, "data.S0", 1.0))
    T = float(get(base_cfg, "data.T", 1.0))
    n = int(get(base_cfg, "data.n", 50))
    sigma_true = float(get(base_cfg, "data.sigma_true", 0.2))
    K_cfg = get(base_cfg, "data.K", None)
    K = float(S0 if K_cfg is None else K_cfg)
    N_train = int(get(base_cfg, "data.N_train", 5000))
    N_val = int(get(base_cfg, "data.N_val", 1000))
    N_test = int(get(base_cfg, "data.N_test", 2000))
    feature_set = canonical_feature_set(get(base_cfg, "features.feature_set", "B"))
    sigma_in_cfg = get(base_cfg, "features.sigma_in", sigma_true)
    sigma_in = float(sigma_true if sigma_in_cfg is None else sigma_in_cfg)
    hidden = int(get(base_cfg, "model.hidden", 128))
    depth = int(get(base_cfg, "model.depth", 4))
    robust_sigmas = robust_sigmas_from_cfg(base_cfg)

    # Seed everything identically to training run
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate the same data as the training run
    reference_data = make_gbm_dataset(
        S0=S0, sigma_true=sigma_true, T=T, n=n, K=K,
        N_train=N_train, N_val=N_val, N_test=N_test,
        seed=seed, feature_set=feature_set, sigma_in=sigma_in,
    )

    if training_regime == "robust":
        data_np = make_gbm_robust_dataset(
            S0=S0, sigma_true=sigma_true, robust_sigmas=robust_sigmas,
            T=T, n=n, K=K, N_train=N_train, N_val=N_val, N_test=N_test,
            seed=seed, feature_set=feature_set, sigma_in_eval=sigma_in,
        )
    else:
        data_np = reference_data

    t_grid = data_np["t_grid"]
    S_te = data_np["S_te"]
    Z_te = data_np["Z_te"]
    F_te = data_np["F_te"]

    p0 = float(bs_call_price_discounted(0.0, S0, K, sigma_true, T))

    # Compute BS deltas at the TARGET sigma_bar
    deltas_bs_test = bs_delta_strategy_paths(t_grid, S_te, K, target_sigma_bar, T)
    pl_bs = pl_paths_proportional_costs(S_te, deltas_bs_test, Z_te, p0, lambda_cost)
    turnover_bs = turnover_paths(deltas_bs_test)

    # Load checkpoint and run NN forward pass
    model = MLPHedge(in_dim=int(data_np["feature_dim"]) + 1, hidden=hidden, depth=depth).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    S_te_t = torch.tensor(S_te, device=device)
    Z_te_t = torch.tensor(Z_te, device=device)
    F_te_t = torch.tensor(F_te, device=device)

    with torch.no_grad():
        deltas_te = rollout_strategy(model, F_te_t)
        pl_te = compute_pl_torch(S_te_t, deltas_te, Z_te_t, p0, lambda_cost).cpu().numpy()
    deltas_te_np = deltas_te.cpu().numpy()
    turnover_nn = turnover_paths(deltas_te_np)

    alpha_list = (0.95, 0.99)

    # Prepare benchmark run directory for the target config
    benchmark_context = prepare_benchmark_run(target_cfg, run_dir)
    repro_dir = benchmark_context.benchmark_run_dir

    # Skip heavy artifacts for eval-only runs:
    # - arrays_debug.npz (~79 MB each) — not needed for statistics
    # - checkpoint copies (~2 MB each) — source checkpoint is sufficient
    # Instead, save only what rebuild_benchmark_statistics needs:
    #   run_meta.json and metrics_summary.json

    # Write minimal feature_norm (small, needed for completeness)
    source_norm = feature_norm_path(source_dir)
    target_norm = feature_norm_path(repro_dir)
    if source_norm.exists():
        import shutil
        shutil.copy2(source_norm, target_norm)

    # Write dummy checkpoint placeholders (empty files, just for path existence)
    best_state_path(repro_dir).touch()
    (Path(repro_dir) / "last_state.pt").touch()

    # Write a dummy train_log (no training was done)
    from src.logging_utils import write_train_log
    from src.paths import train_log_path
    write_train_log(train_log_path(repro_dir), [{"epoch": 0, "train_loss": 0.0, "val_loss": 0.0, "lr": 0.0, "w": 0.0}])

    # Save eval artifacts WITHOUT arrays_debug (pass None to skip the ~79 MB npz)
    m_bs, m_nn = save_eval_artifacts(
        run_dir=repro_dir,
        pl_bs=pl_bs,
        pl_nn=pl_te,
        label_bs="BS-delta",
        label_nn="Deep hedging",
        alpha_list=alpha_list,
        lam_entropic=1.0,
        arrays_debug=None,
    )

    # Finalize — this writes metrics_summary.json, run_meta.json,
    # pl/turnover .npy files, and updates manifest/summary_rows
    finalize_benchmark_run(
        cfg=target_cfg,
        root_run_dir=run_dir,
        context=benchmark_context,
        metrics_bs=m_bs,
        metrics_nn=m_nn,
        pl_bs=pl_bs,
        pl_nn=pl_te,
        turnover_bs=turnover_bs,
        turnover_nn=turnover_nn,
        train_log=[{"epoch": 0, "train_loss": 0.0, "val_loss": 0.0, "lr": 0.0, "w": 0.0}],
    )

    print(f"  BS-delta(sigma_bar={target_sigma_bar}): ES95={m_bs.get('ES_loss_0.95', 'N/A'):.6f}")
    print(f"  DH (reused checkpoint): ES95={m_nn.get('ES_loss_0.95', 'N/A'):.6f}")


def _build_cfg(
    base_cfg: dict[str, Any],
    *,
    sigma_bar: float,
    seed: int,
    lambda_cost: float,
    training_regime: str,
    campaign_id: str,
    campaign_role: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    _set_dotted(cfg, "data.sigma_bar", sigma_bar)
    _set_dotted(cfg, "data.seed", seed)
    _set_dotted(cfg, "data.lam_cost", lambda_cost)
    _set_dotted(cfg, "benchmark.training_regime", canonical_training_regime(training_regime))
    _set_dotted(cfg, "benchmark.run_mode", "benchmark")
    _set_dotted(cfg, "benchmark.is_benchmark_eligible", True)
    _set_dotted(cfg, "benchmark.campaign_id", campaign_id)
    _set_dotted(cfg, "benchmark.campaign_role", campaign_role)
    return cfg


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.run_benchmark_eval_only")
    parser.add_argument("--config", default=None, help="Base YAML config path.")
    parser.add_argument("--sigma-bars", required=True, help="Comma-separated target sigma_bar values to evaluate.")
    parser.add_argument("--source-sigma-bar", default="0.20", help="Source sigma_bar where checkpoints exist.")
    parser.add_argument("--lambda-costs", default="0,1e-4,5e-4,1e-3", help="Comma-separated lambda_cost values.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds.")
    parser.add_argument("--training-regimes", default="oracle,robust", help="Comma-separated training regimes.")
    parser.add_argument("--campaign-id", default="main_benchmark", help="Campaign identifier.")
    parser.add_argument("--campaign-role", default="main", help="Campaign role.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    return parser


def _parse_float_csv(raw: str) -> list[float]:
    return [float(t.strip()) for t in raw.split(",") if t.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(t.strip()) for t in raw.split(",") if t.strip()]


def _parse_str_csv(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
