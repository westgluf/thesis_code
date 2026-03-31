from __future__ import annotations
from typing import Any, Dict

from pathlib import Path
import numpy as np

from src.logging_utils import write_json_file
from src.models_gbm import simulate_gbm_discounted_paths
from src.paths import feature_norm_path
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


def save_feature_norm_json(feature_norm: Dict[str, Any], run_dir: str | Path) -> None:
    write_json_file(feature_norm_path(run_dir), feature_norm)


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
