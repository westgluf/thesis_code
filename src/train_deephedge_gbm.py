from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange

from src.benchmark_repro import (
    canonical_training_regime,
    deep_hedge_method_name,
    fail_benchmark_run,
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
from src.logging_utils import write_run_config, write_train_log
from src.metrics import summary_metrics
from src.objectives import build_objective, canonical_objective_name
from src.paths import best_state_path, get_run_dir, last_state_path, run_cfg_path, train_log_path
from src.strategies_delta import bs_delta_strategy_paths
from src.train_loop import train_loop
from src.world_gbm import (
    canonical_feature_set,
    make_gbm_dataset,
    make_gbm_robust_dataset,
    save_feature_norm_json,
)


@dataclass(frozen=True)
class ExperimentRunResult:
    run_id: str
    run_dir: Path
    benchmark_run_dir: Path
    training_regime: str
    deep_hedge_method: str
    metrics_bs: dict[str, float]
    metrics_nn: dict[str, float]


def _save_training_artifacts(
    run_dir: str | Path,
    best_state: dict[str, torch.Tensor],
    last_state: dict[str, torch.Tensor],
    train_log,
) -> None:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, best_state_path(out_dir))
    torch.save(last_state, last_state_path(out_dir))
    write_train_log(train_log_path(out_dir), train_log)


def run_from_cfg(cfg: Mapping[str, Any]) -> ExperimentRunResult:
    device = str(get(cfg, "device", "cpu"))
    seed = int(get(cfg, "data.seed", 1234))
    _seed_everything(seed)

    S0 = float(get(cfg, "data.S0", 1.0))
    T = float(get(cfg, "data.T", 1.0))
    n = int(get(cfg, "data.n", 50))
    sigma_true = float(get(cfg, "data.sigma_true", 0.2))
    sigma_bar = float(get(cfg, "data.sigma_bar", 0.2))
    lam_cost = float(get(cfg, "data.lam_cost", 0.0))
    feature_set = canonical_feature_set(get(cfg, "features.feature_set", "B"))
    sigma_in_cfg = get(cfg, "features.sigma_in", sigma_true)
    sigma_in = float(sigma_true if sigma_in_cfg is None else sigma_in_cfg)
    training_regime = canonical_training_regime(get(cfg, "benchmark.training_regime", "oracle"))
    robust_sigmas = robust_sigmas_from_cfg(cfg)

    k_cfg = get(cfg, "data.K", None)
    K = float(S0 if k_cfg is None else k_cfg)

    N_train = int(get(cfg, "data.N_train", 5000))
    N_val = int(get(cfg, "data.N_val", 1000))
    N_test = int(get(cfg, "data.N_test", 2000))

    objective_name = canonical_objective_name(get(cfg, "objective.name", "cvar"))
    alpha_es = float(get(cfg, "objective.alpha", 0.95))
    gamma_entropic = float(get(cfg, "objective.gamma", 1.0))
    lambda_mv = float(get(cfg, "objective.lambda_mv", 1.0))
    objective_w0 = float(get(cfg, "objective.w0", 0.0))

    epochs = int(get(cfg, "train.epochs", 60))
    batch_size = int(get(cfg, "train.batch_size", 2048))
    lr = float(get(cfg, "train.lr", 3e-4))
    wd = float(get(cfg, "train.weight_decay", 0.0))
    patience = int(get(cfg, "train.patience", 10))

    hidden = int(get(cfg, "model.hidden", 128))
    depth = int(get(cfg, "model.depth", 4))

    run_dir = get_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_cfg_path(run_dir), cfg)
    benchmark_context = prepare_benchmark_run(cfg, run_dir)
    try:
        reference_data = make_gbm_dataset(
            S0=S0,
            sigma_true=sigma_true,
            T=T,
            n=n,
            K=K,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            feature_set=feature_set,
            sigma_in=sigma_in,
        )
        data_np = _select_training_dataset(
            feature_set=feature_set,
            sigma_in=sigma_in,
            reference_data=reference_data,
            S0=S0,
            sigma_true=sigma_true,
            T=T,
            n=n,
            K=K,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            training_regime=training_regime,
            robust_sigmas=robust_sigmas,
        )

        save_feature_norm_json(data_np["feature_norm"], run_dir)
        save_feature_norm_json(data_np["feature_norm"], benchmark_context.benchmark_run_dir)

        t_grid = data_np["t_grid"]
        S_tr, Z_tr, F_tr = data_np["S_tr"], data_np["Z_tr"], data_np["F_tr"]
        S_va, Z_va, F_va = data_np["S_va"], data_np["Z_va"], data_np["F_va"]
        S_te, Z_te, F_te = data_np["S_te"], data_np["Z_te"], data_np["F_te"]

        p0_true_mc = float(np.mean(reference_data["Z_tr"]))

        deltas_bs_test = bs_delta_strategy_paths(t_grid, S_te, K, sigma_bar, T)
        p0_bs = bs_call_price_discounted(0.0, S0, K, sigma_bar, T)
        pl_bs = pl_paths_proportional_costs(S_te, deltas_bs_test, Z_te, p0_true_mc, lam_cost)
        turnover_bs = turnover_paths(deltas_bs_test)

        S_tr_t = torch.tensor(S_tr, device=device)
        Z_tr_t = torch.tensor(Z_tr, device=device)
        F_tr_t = torch.tensor(F_tr, device=device)
        S_va_t = torch.tensor(S_va, device=device)
        Z_va_t = torch.tensor(Z_va, device=device)
        F_va_t = torch.tensor(F_va, device=device)
        S_te_t = torch.tensor(S_te, device=device)
        Z_te_t = torch.tensor(Z_te, device=device)
        F_te_t = torch.tensor(F_te, device=device)

        model = MLPHedge(in_dim=int(data_np["feature_dim"]) + 1, hidden=hidden, depth=depth).to(device)
        objective_fn = build_objective(
            name=objective_name,
            alpha=alpha_es,
            gamma=gamma_entropic,
            lambda_mv=lambda_mv,
            w0=objective_w0,
        ).to(device)
        optimizer = Adam(list(model.parameters()) + list(objective_fn.parameters()), lr=lr, weight_decay=wd)

        train_data = {
            "F_tr": F_tr_t,
            "S_tr": S_tr_t,
            "Z_tr": Z_tr_t,
            "F_va": F_va_t,
            "S_va": S_va_t,
            "Z_va": Z_va_t,
            "p0_true_mc": p0_true_mc,
            "lam_cost": lam_cost,
        }

        best_state, last_state, train_log = train_loop(
            model=model,
            optimizer=optimizer,
            objective_fn=objective_fn,
            data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            trange=trange,
        )
        _save_training_artifacts(run_dir, best_state, last_state, train_log)
        _save_training_artifacts(benchmark_context.benchmark_run_dir, best_state, last_state, train_log)

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            deltas_te = rollout_strategy(model, F_te_t)
            pl_te = compute_pl_torch(S_te_t, deltas_te, Z_te_t, p0_true_mc, lam_cost).cpu().numpy()
        deltas_te_np = deltas_te.cpu().numpy()
        turnover_nn = turnover_paths(deltas_te_np)

        alpha_list = (0.95, 0.99)
        arrays = {
            "S_test": S_te,
            "Z_test": Z_te,
            "deltas_nn": deltas_te_np,
            "deltas_bs": deltas_bs_test,
            "pl_nn": pl_te,
            "pl_bs": pl_bs,
            "turnover_nn": turnover_nn,
            "turnover_bs": turnover_bs,
        }
        m_bs = summary_metrics(pl_bs, alpha_list=alpha_list, lam_entropic=1.0)
        m_nn = summary_metrics(pl_te, alpha_list=alpha_list, lam_entropic=1.0)

        m_bs, m_nn = save_eval_artifacts(
            run_dir=run_dir,
            pl_bs=pl_bs,
            pl_nn=pl_te,
            label_bs="BS-delta",
            label_nn="Deep hedging",
            alpha_list=alpha_list,
            lam_entropic=1.0,
            arrays_debug=arrays,
        )
        save_eval_artifacts(
            run_dir=benchmark_context.benchmark_run_dir,
            pl_bs=pl_bs,
            pl_nn=pl_te,
            label_bs="BS-delta",
            label_nn="Deep hedging",
            alpha_list=alpha_list,
            lam_entropic=1.0,
            arrays_debug=arrays,
        )
        finalize_benchmark_run(
            cfg=cfg,
            root_run_dir=run_dir,
            context=benchmark_context,
            metrics_bs=m_bs,
            metrics_nn=m_nn,
            pl_bs=pl_bs,
            pl_nn=pl_te,
            turnover_bs=turnover_bs,
            turnover_nn=turnover_nn,
            train_log=train_log,
        )

        print(f"Saved results to: {run_dir}")
        print("BS-delta:", m_bs)
        print("Deep hedging:", m_nn)
        print("Note: p0 used = MC estimate from true-vol train set; BS price also available:", p0_bs)

        return ExperimentRunResult(
            run_id=benchmark_context.run_id,
            run_dir=run_dir,
            benchmark_run_dir=benchmark_context.benchmark_run_dir,
            training_regime=training_regime,
            deep_hedge_method=deep_hedge_method_name(training_regime),
            metrics_bs=dict(m_bs),
            metrics_nn=dict(m_nn),
        )
    except Exception as exc:
        fail_benchmark_run(cfg=cfg, root_run_dir=run_dir, context=benchmark_context, error=exc)
        raise


def main() -> None:
    cfg_path = os.environ.get("GBM_CFG", "configs/gbm_es95.yaml")
    cfg = load_yaml(cfg_path)
    run_from_cfg(cfg)


def _select_training_dataset(
    *,
    feature_set: str,
    sigma_in: float,
    reference_data: dict[str, Any],
    S0: float,
    sigma_true: float,
    T: float,
    n: int,
    K: float,
    N_train: int,
    N_val: int,
    N_test: int,
    seed: int,
    training_regime: str,
    robust_sigmas: tuple[float, ...],
) -> dict[str, Any]:
    if training_regime == "oracle":
        return reference_data
    return make_gbm_robust_dataset(
        S0=S0,
        sigma_true=sigma_true,
        robust_sigmas=robust_sigmas,
        T=T,
        n=n,
        K=K,
        N_train=N_train,
        N_val=N_val,
        N_test=N_test,
        seed=seed,
        feature_set=feature_set,
        sigma_in_eval=sigma_in,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


if __name__ == "__main__":
    main()
