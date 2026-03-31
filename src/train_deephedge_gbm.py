from __future__ import annotations

import os

import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange

from src.bs import bs_call_price_discounted
from src.config import get, load_yaml
from src.costs_and_pl import pl_paths_proportional_costs
from src.deep_hedging_model import MLPHedge
from src.eval import save_eval_artifacts
from src.hedge_core import compute_pl_torch, rollout_strategy
from src.logging_utils import write_run_config, write_train_log
from src.objectives import CVaRObjective
from src.paths import best_state_path, get_run_dir, last_state_path, run_cfg_path, train_log_path
from src.strategies_delta import bs_delta_strategy_paths
from src.train_loop import train_loop
from src.world_gbm import make_gbm_dataset, save_feature_norm_json


def _save_training_artifacts(
    run_dir,
    best_state: dict[str, torch.Tensor],
    last_state: dict[str, torch.Tensor],
    train_log,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, best_state_path(run_dir))
    torch.save(last_state, last_state_path(run_dir))
    write_train_log(train_log_path(run_dir), train_log)


def main() -> None:
    cfg_path = os.environ.get("GBM_CFG", "configs/gbm_es95.yaml")
    cfg = load_yaml(cfg_path)

    device = str(get(cfg, "device", "cpu"))
    seed = int(get(cfg, "data.seed", 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)

    S0 = float(get(cfg, "data.S0", 1.0))
    T = float(get(cfg, "data.T", 1.0))
    n = int(get(cfg, "data.n", 50))
    sigma_true = float(get(cfg, "data.sigma_true", 0.2))
    sigma_bar = float(get(cfg, "data.sigma_bar", 0.2))
    lam_cost = float(get(cfg, "data.lam_cost", 0.0))

    k_cfg = get(cfg, "data.K", None)
    K = float(S0 if k_cfg is None else k_cfg)

    N_train = int(get(cfg, "data.N_train", 5000))
    N_val = int(get(cfg, "data.N_val", 1000))
    N_test = int(get(cfg, "data.N_test", 2000))

    alpha_es = float(get(cfg, "objective.alpha", 0.95))

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

    data_np = make_gbm_dataset(
        S0=S0,
        sigma_true=sigma_true,
        T=T,
        n=n,
        K=K,
        N_train=N_train,
        N_val=N_val,
        N_test=N_test,
        seed=seed,
    )

    save_feature_norm_json(data_np["feature_norm"], run_dir)

    t_grid = data_np["t_grid"]
    S_tr, Z_tr, F_tr = data_np["S_tr"], data_np["Z_tr"], data_np["F_tr"]
    S_va, Z_va, F_va = data_np["S_va"], data_np["Z_va"], data_np["F_va"]
    S_te, Z_te, F_te = data_np["S_te"], data_np["Z_te"], data_np["F_te"]

    p0_true_mc = float(np.mean(np.maximum(S_tr[:, -1] - K, 0.0)))

    deltas_bs_test = bs_delta_strategy_paths(t_grid, S_te, K, sigma_bar, T)
    p0_bs = bs_call_price_discounted(0.0, S0, K, sigma_bar, T)
    pl_bs = pl_paths_proportional_costs(S_te, deltas_bs_test, Z_te, p0_true_mc, lam_cost)

    S_tr_t = torch.tensor(S_tr, device=device)
    Z_tr_t = torch.tensor(Z_tr, device=device)
    F_tr_t = torch.tensor(F_tr, device=device)

    S_va_t = torch.tensor(S_va, device=device)
    Z_va_t = torch.tensor(Z_va, device=device)
    F_va_t = torch.tensor(F_va, device=device)

    S_te_t = torch.tensor(S_te, device=device)
    Z_te_t = torch.tensor(Z_te, device=device)
    F_te_t = torch.tensor(F_te, device=device)

    model = MLPHedge(in_dim=4, hidden=hidden, depth=depth).to(device)
    objective_fn = CVaRObjective(alpha=alpha_es).to(device)
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

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        deltas_te = rollout_strategy(model, F_te_t)
        pl_te = compute_pl_torch(S_te_t, deltas_te, Z_te_t, p0_true_mc, lam_cost).cpu().numpy()

    alpha_list = (0.95, 0.99)
    arrays = {
        "S_test": S_te,
        "Z_test": Z_te,
        "deltas_nn": deltas_te.cpu().numpy() if hasattr(deltas_te, "cpu") else np.asarray(deltas_te),
        "pl_nn": pl_te,
        "pl_bs": pl_bs,
    }

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

    print(f"Saved results to: {run_dir}")
    print("BS-delta:", m_bs)
    print("Deep hedging:", m_nn)
    print("Note: p0 used = MC estimate from train set; BS price also available:", p0_bs)


if __name__ == "__main__":
    main()
