#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p src tools configs results/archive archive results/gbm_deephedge results/gbm_baseline

if [ ! -f src/__init__.py ]; then
  printf "" > src/__init__.py
fi

cat > src/config.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import os

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from e


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError("Config root must be a dict")
    return obj


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
PY

cat > configs/gbm_es95.yaml <<'YAML'
out_dir: results/gbm_deephedge

data:
  S0: 1.0
  T: 1.0
  n: 50
  sigma_true: 0.2
  sigma_bar: 0.2
  lam_cost: 0.0
  N_train: 5000
  N_val: 1000
  N_test: 2000
  seed: 1234
  K: null

objective:
  alpha: 0.95
  w0: 0.0

model:
  hidden: 128
  depth: 4

train:
  epochs: 60
  batch_size: 2048
  lr: 3e-4
  weight_decay: 0.0
  patience: 10
  grad_clip: 1.0

delta_clip:
  enabled: true
YAML

cat > src/world_gbm.py <<'PY'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import json
import os
import numpy as np

from src.models_gbm import simulate_gbm_discounted_paths
from src.payoff import payoff_call


def make_features(t_grid: np.ndarray, S_paths: np.ndarray) -> np.ndarray:
    N, n_plus_1 = S_paths.shape
    n = n_plus_1 - 1
    feats = np.zeros((N, n, 4), dtype=np.float32)
    for k in range(n):
        t = t_grid[k]
        tau = t_grid[-1] - t
        feats[:, k, 0] = t
        feats[:, k, 1] = np.log(S_paths[:, k])
        feats[:, k, 2] = tau
    return feats


def fit_feature_norm(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = F.reshape(-1, F.shape[-1]).astype(np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-8
    mu[3] = 0.0
    sd[3] = 1.0
    return mu, sd


def apply_feature_norm(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    G = ((F - mu) / sd).astype(np.float32)
    G[..., 3] = F[..., 3].astype(np.float32)
    return G


def save_feature_norm_json(feature_norm: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feature_norm.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_norm, f, indent=2)


def make_gbm_dataset(
    S0: float,
    sigma_true: float,
    T: float,
    n: int,
    K: float,
    N_train: int,
    N_val: int,
    N_test: int,
    seed: int,
) -> Dict[str, Any]:
    t_grid, S_paths = simulate_gbm_discounted_paths(S0, sigma_true, T, n, N_train + N_val + N_test, seed)

    idx = np.arange(S_paths.shape[0])
    train_idx = idx[:N_train]
    val_idx = idx[N_train:N_train + N_val]
    test_idx = idx[N_train + N_val:]

    def pack(split_idx: np.ndarray):
        S = S_paths[split_idx].astype(np.float32)
        ST = S[:, -1]
        Z = payoff_call(ST, K).astype(np.float32)
        F = make_features(t_grid, S)
        return S, Z, F

    S_tr, Z_tr, F_tr = pack(train_idx)
    S_va, Z_va, F_va = pack(val_idx)
    S_te, Z_te, F_te = pack(test_idx)

    mu, sd = fit_feature_norm(F_tr)
    F_tr = apply_feature_norm(F_tr, mu, sd)
    F_va = apply_feature_norm(F_va, mu, sd)
    F_te = apply_feature_norm(F_te, mu, sd)

    feature_norm = {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
    }

    return {
        "t_grid": t_grid.astype(np.float32),
        "S_tr": S_tr,
        "Z_tr": Z_tr,
        "F_tr": F_tr,
        "S_va": S_va,
        "Z_va": Z_va,
        "F_va": F_va,
        "S_te": S_te,
        "Z_te": Z_te,
        "F_te": F_te,
        "feature_norm": feature_norm,
    }
PY

cat > src/objectives.py <<'PY'
from __future__ import annotations
import torch


def cvar_loss_from_pl(pl: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    loss = -pl
    a = float(alpha)
    return w + torch.relu(loss - w).mean() / (1.0 - a)
PY

cat > src/hedge_core.py <<'PY'
from __future__ import annotations
import torch


def compute_pl_torch(S: torch.Tensor, deltas: torch.Tensor, Z: torch.Tensor, p0: float, lam: float) -> torch.Tensor:
    dS = S[:, 1:] - S[:, :-1]
    gains = (deltas * dS).sum(dim=1)

    delta_prev = torch.cat([torch.zeros((S.shape[0], 1), device=S.device), deltas[:, :-1]], dim=1)
    trade = deltas - delta_prev

    costs = (lam * S[:, :-1] * trade.abs()).sum(dim=1)
    close_cost = lam * S[:, -1] * deltas[:, -1].abs()
    costs = costs + close_cost

    pl = -Z + p0 + gains - costs
    return pl


def rollout_strategy(model, feats_base: torch.Tensor, clip: bool = True) -> torch.Tensor:
    N, n, _ = feats_base.shape
    deltas = []
    delta_prev = torch.zeros((N, 1), device=feats_base.device)
    for k in range(n):
        x = feats_base[:, k, :].clone()
        x[:, 3:4] = delta_prev
        delta_k = model(x)
        if clip:
            delta_k = torch.tanh(delta_k)
        deltas.append(delta_k)
        delta_prev = delta_k
    return torch.cat(deltas, dim=1)
PY

cat > src/eval.py <<'PY'
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional
import os
import numpy as np

from src.metrics import summary_metrics
from src.plots import plot_hist, plot_es_var_bars


def save_eval_artifacts(
    out_dir: str,
    pl_bs: np.ndarray,
    pl_nn: np.ndarray,
    label_bs: str,
    label_nn: str,
    alpha_list: Iterable[float] = (0.95, 0.99),
    lam_entropic: float = 1.0,
    arrays_debug: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)

    alpha_list = tuple(alpha_list)
    m_bs = summary_metrics(pl_bs, alpha_list=alpha_list, lam_entropic=lam_entropic)
    m_nn = summary_metrics(pl_nn, alpha_list=alpha_list, lam_entropic=lam_entropic)

    import json
    with open(os.path.join(out_dir, "metrics_bs.json"), "w", encoding="utf-8") as f:
        json.dump(m_bs, f, indent=2)
    with open(os.path.join(out_dir, "metrics_nn.json"), "w", encoding="utf-8") as f:
        json.dump(m_nn, f, indent=2)

    plot_hist(pl_bs, pl_nn, label_bs, label_nn, os.path.join(out_dir, "hist_pl_bs_vs_nn.png"))
    plot_es_var_bars(m_bs, m_nn, alpha_list, os.path.join(out_dir, "tail_metrics_bs_vs_nn.png"), title="GBM: BS-delta vs Deep hedging")

    if arrays_debug is not None:
        try:
            np.savez(os.path.join(out_dir, "arrays_debug.npz"), **arrays_debug)
        except Exception as e:
            print("Warning: could not save arrays_debug.npz:", e)

    return m_bs, m_nn
PY

cat > src/train_loop.py <<'PY'
from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple, Optional
import os
import torch


def train_loop(
    model,
    objective: Callable[[torch.Tensor], torch.Tensor],
    rollout_fn: Callable,
    pl_fn: Callable,
    S_tr_t: torch.Tensor,
    Z_tr_t: torch.Tensor,
    F_tr_t: torch.Tensor,
    S_va_t: torch.Tensor,
    Z_va_t: torch.Tensor,
    F_va_t: torch.Tensor,
    p0_true_mc: float,
    lam_cost: float,
    opt: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    patience: int,
    grad_clip: float,
    out_dir: str,
    w_value_fn: Optional[Callable[[], float]] = None,
    trange=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    if trange is None:
        from tqdm import trange as _trange
        trange = _trange

    best_val = float("inf")
    best_state = None
    bad = 0

    train_log: List[Dict[str, float]] = []
    Ntr = F_tr_t.shape[0]

    for ep in trange(int(epochs), desc="Training"):
        model.train()
        perm = torch.randperm(Ntr, device=F_tr_t.device)
        total_loss = 0.0
        nb = 0

        for start in range(0, Ntr, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            F_b = F_tr_t[idx]
            S_b = S_tr_t[idx]
            Z_b = Z_tr_t[idx]

            opt.zero_grad()
            deltas = rollout_fn(model, F_b)
            pl = pl_fn(S_b, deltas, Z_b, p0_true_mc, lam_cost)

            loss = objective(pl)
            loss = loss + (1e-4 * (deltas ** 2).mean()) + (0.0 * ((deltas[:, 1:] - deltas[:, :-1]) ** 2).mean())

            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()

            total_loss += float(loss.detach().cpu())
            nb += 1

        model.eval()
        with torch.no_grad():
            deltas_va = rollout_fn(model, F_va_t)
            pl_va = pl_fn(S_va_t, deltas_va, Z_va_t, p0_true_mc, lam_cost)
            val_loss = float(objective(pl_va).detach().cpu().item())

        train_loss_epoch = float(total_loss) / float(max(nb, 1))
        lr_now = float(opt.param_groups[0].get("lr", 0.0))
        w_now = float(w_value_fn()) if w_value_fn is not None else float("nan")

        train_log.append({
            "epoch": float(ep),
            "train_loss": train_loss_epoch,
            "val_loss": float(val_loss),
            "lr": lr_now,
            "w": w_now,
        })

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = last_state

    os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(best_state, os.path.join(out_dir, "best_state.pt"))
        torch.save(last_state, os.path.join(out_dir, "last_state.pt"))
    except Exception as e:
        print("Warning: could not save checkpoints:", e)

    try:
        import csv
        with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "w"])
            writer.writeheader()
            for row in train_log:
                writer.writerow(row)
    except Exception as e:
        print("Warning: could not save train_log.csv:", e)

    return best_state, last_state, train_log
PY

cat > src/train_deephedge_gbm.py <<'PY'
from __future__ import annotations
import os
import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange

from src.config import load_yaml, get
from src.world_gbm import make_gbm_dataset, save_feature_norm_json
from src.bs import bs_call_price_discounted
from src.strategies_delta import bs_delta_strategy_paths
from src.costs_and_pl import pl_paths_proportional_costs
from src.deep_hedging_model import MLPHedge

from src.objectives import cvar_loss_from_pl
from src.hedge_core import compute_pl_torch, rollout_strategy
from src.train_loop import train_loop
from src.eval import save_eval_artifacts


def main() -> None:
    cfg_path = os.environ.get("GBM_CFG", "configs/gbm_es95.yaml")
    cfg = load_yaml(cfg_path)

    out_dir = str(get(cfg, "out_dir", "results/gbm_deephedge"))
    os.makedirs(out_dir, exist_ok=True)

    device = "cpu"
    seed = int(get(cfg, "data.seed", 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)

    S0 = float(get(cfg, "data.S0", 1.0))
    T = float(get(cfg, "data.T", 1.0))
    n = int(get(cfg, "data.n", 50))
    sigma_true = float(get(cfg, "data.sigma_true", 0.2))
    sigma_bar = float(get(cfg, "data.sigma_bar", 0.2))
    lam_cost = float(get(cfg, "data.lam_cost", 0.0))

    N_train = int(get(cfg, "data.N_train", 5000))
    N_val = int(get(cfg, "data.N_val", 1000))
    N_test = int(get(cfg, "data.N_test", 2000))

    K_cfg = get(cfg, "data.K", None)
    K = float(S0 if K_cfg is None else K_cfg)

    alpha_es = float(get(cfg, "objective.alpha", 0.95))
    w0 = float(get(cfg, "objective.w0", 0.0))

    hidden = int(get(cfg, "model.hidden", 128))
    depth = int(get(cfg, "model.depth", 4))

    epochs = int(get(cfg, "train.epochs", 60))
    batch_size = int(get(cfg, "train.batch_size", 2048))
    lr = float(get(cfg, "train.lr", 3e-4))
    weight_decay = float(get(cfg, "train.weight_decay", 0.0))
    patience = int(get(cfg, "train.patience", 10))
    grad_clip = float(get(cfg, "train.grad_clip", 1.0))

    delta_clip = bool(get(cfg, "delta_clip.enabled", True))

    data = make_gbm_dataset(
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
    t_grid = data["t_grid"]
    S_tr, Z_tr, F_tr = data["S_tr"], data["Z_tr"], data["F_tr"]
    S_va, Z_va, F_va = data["S_va"], data["Z_va"], data["F_va"]
    S_te, Z_te, F_te = data["S_te"], data["Z_te"], data["F_te"]

    save_feature_norm_json(data["feature_norm"], out_dir)

    p0_true_mc = float(np.mean(Z_tr))
    p0_bs = bs_call_price_discounted(0.0, S0, K, sigma_bar, T)

    deltas_bs_test = bs_delta_strategy_paths(t_grid, S_te, K, sigma_bar, T)
    PL_bs = pl_paths_proportional_costs(S_te, deltas_bs_test, Z_te, p0_true_mc, lam_cost)

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
    w_es = torch.nn.Parameter(torch.tensor(float(w0), device=device))
    opt = Adam(list(model.parameters()) + [w_es], lr=lr, weight_decay=weight_decay)

    def objective(pl: torch.Tensor) -> torch.Tensor:
        return cvar_loss_from_pl(pl, w_es, alpha=alpha_es)

    def w_value() -> float:
        return float(w_es.detach().cpu().item())

    def rollout_fn(m, F):
        return rollout_strategy(m, F, clip=delta_clip)

    best_state, last_state, train_log = train_loop(
        model=model,
        objective=objective,
        rollout_fn=rollout_fn,
        pl_fn=compute_pl_torch,
        S_tr_t=S_tr_t,
        Z_tr_t=Z_tr_t,
        F_tr_t=F_tr_t,
        S_va_t=S_va_t,
        Z_va_t=Z_va_t,
        F_va_t=F_va_t,
        p0_true_mc=p0_true_mc,
        lam_cost=lam_cost,
        opt=opt,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        grad_clip=grad_clip,
        out_dir=out_dir,
        w_value_fn=w_value,
        trange=trange,
    )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        deltas_te = rollout_fn(model, F_te_t)
        pl_te = compute_pl_torch(S_te_t, deltas_te, Z_te_t, p0_true_mc, lam_cost).cpu().numpy()

    arrays = dict(
        S_test=S_te,
        Z_test=Z_te,
        deltas_nn=deltas_te.detach().cpu().numpy(),
        pl_nn=pl_te,
        pl_bs=PL_bs,
    )

    m_bs, m_nn = save_eval_artifacts(
        out_dir=out_dir,
        pl_bs=PL_bs,
        pl_nn=pl_te,
        label_bs="BS-delta",
        label_nn="Deep hedging",
        alpha_list=(0.95, 0.99),
        lam_entropic=1.0,
        arrays_debug=arrays,
    )

    print(f"Saved results to: {out_dir}")
    print("BS-delta:", m_bs)
    print("Deep hedging:", m_nn)
    print("Note: p0 used = MC estimate from train set; BS price also available:", p0_bs)


if __name__ == "__main__":
    main()
PY

python -m py_compile src/*.py tools/*.py

