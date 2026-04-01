from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np

from src.logging_utils import write_json_file
from src.models_gbm import simulate_gbm_discounted_paths
from src.paths import feature_norm_path
from src.payoff import payoff_call


FeatureSet = Literal["B", "C", "D"]

_FEATURE_NAMES: dict[FeatureSet, tuple[str, ...]] = {
    "B": ("tau", "log_moneyness"),
    "C": ("tau", "log_moneyness", "r_last", "rv_run"),
    "D": ("tau", "log_moneyness", "sigma_in"),
}

_NORM_EPS = 1e-8


def canonical_feature_set(name: str | None) -> FeatureSet:
    raw = "B" if name is None else str(name).strip().upper()
    if raw not in _FEATURE_NAMES:
        valid = ", ".join(sorted(_FEATURE_NAMES))
        raise ValueError(f"unknown feature_set {name!r}; expected one of: {valid}")
    return raw


def feature_names(feature_set: str | None) -> tuple[str, ...]:
    return _FEATURE_NAMES[canonical_feature_set(feature_set)]


def feature_dim(feature_set: str | None) -> int:
    return len(feature_names(feature_set))


def policy_input_dim(feature_set: str | None) -> int:
    return feature_dim(feature_set) + 1


def make_features(
    t_grid: np.ndarray,
    S_paths: np.ndarray,
    *,
    K: float,
    feature_set: str | None = "B",
    sigma_in: float | np.ndarray | None = None,
) -> np.ndarray:
    feature_set_norm = canonical_feature_set(feature_set)
    N, n_plus_1 = S_paths.shape
    n = n_plus_1 - 1
    if n <= 0:
        raise ValueError("S_paths must contain at least one hedge step")

    T = float(t_grid[-1])
    if T <= 0.0:
        raise ValueError("t_grid must end at a positive maturity")
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")

    tau = ((T - t_grid[:-1]) / T).astype(np.float32)
    tau_grid = np.broadcast_to(tau, (N, n))
    log_moneyness = np.log(S_paths[:, :-1] / float(K)).astype(np.float32)

    if feature_set_norm == "B":
        feats = np.stack([tau_grid, log_moneyness], axis=-1)
        return feats.astype(np.float32)

    if feature_set_norm == "C":
        log_returns = np.log(S_paths[:, 1:] / S_paths[:, :-1]).astype(np.float32)

        r_last = np.zeros((N, n), dtype=np.float32)
        if n > 1:
            r_last[:, 1:] = log_returns[:, :-1]

        rv_run = np.zeros((N, n), dtype=np.float32)
        sq_returns = np.square(log_returns[:, :-1]).astype(np.float32, copy=False)
        for k in range(1, n):
            elapsed = float(t_grid[k] - t_grid[0])
            rv_run[:, k] = np.sqrt(np.sum(sq_returns[:, :k], axis=1) / max(elapsed, _NORM_EPS))

        feats = np.stack([tau_grid, log_moneyness, r_last, rv_run], axis=-1)
        return feats.astype(np.float32)

    sigma_feat = _broadcast_sigma_feature(sigma_in=sigma_in, sigma_default=None, N=N, n=n)
    feats = np.stack([tau_grid, log_moneyness, sigma_feat], axis=-1)
    return feats.astype(np.float32)


def fit_feature_norm(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = F.reshape(-1, F.shape[-1]).astype(np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + _NORM_EPS
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_feature_norm(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((F - mu) / sd).astype(np.float32)


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
    feature_set: str | None = "B",
    sigma_in: float | np.ndarray | None = None,
) -> Dict[str, Any]:
    feature_set_norm = canonical_feature_set(feature_set)
    t_grid, S_paths = simulate_gbm_discounted_paths(S0, sigma_true, T, n, N_train + N_val + N_test, seed)

    idx = np.arange(S_paths.shape[0])
    train_idx = idx[:N_train]
    val_idx = idx[N_train:N_train + N_val]
    test_idx = idx[N_train + N_val:]

    def pack(split_idx: np.ndarray):
        S = S_paths[split_idx].astype(np.float32)
        ST = S[:, -1]
        Z = payoff_call(ST, K).astype(np.float32)
        sigma_in_split = _slice_sigma_input(sigma_in=sigma_in, split_idx=split_idx, sigma_default=sigma_true)
        F = make_features(t_grid, S, K=K, feature_set=feature_set_norm, sigma_in=sigma_in_split)
        return S, Z, F

    S_tr, Z_tr, F_tr = pack(train_idx)
    S_va, Z_va, F_va = pack(val_idx)
    S_te, Z_te, F_te = pack(test_idx)

    mu, sd = fit_feature_norm(F_tr)
    F_tr = apply_feature_norm(F_tr, mu, sd)
    F_va = apply_feature_norm(F_va, mu, sd)
    F_te = apply_feature_norm(F_te, mu, sd)

    feature_norm = {
        "feature_set": feature_set_norm,
        "feature_names": list(feature_names(feature_set_norm)),
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
        "feature_set": feature_set_norm,
        "feature_names": list(feature_names(feature_set_norm)),
        "feature_dim": feature_dim(feature_set_norm),
        "feature_norm": feature_norm,
    }


def make_gbm_robust_dataset(
    S0: float,
    sigma_true: float,
    robust_sigmas: Sequence[float],
    T: float,
    n: int,
    K: float,
    N_train: int,
    N_val: int,
    N_test: int,
    seed: int,
    feature_set: str | None = "B",
    sigma_in_eval: float | np.ndarray | None = None,
) -> Dict[str, Any]:
    feature_set_norm = canonical_feature_set(feature_set)
    sigma_values = tuple(float(value) for value in robust_sigmas)
    if not sigma_values:
        raise ValueError("robust_sigmas must contain at least one volatility")

    reference_dataset = make_gbm_dataset(
        S0=S0,
        sigma_true=sigma_true,
        T=T,
        n=n,
        K=K,
        N_train=N_train,
        N_val=N_val,
        N_test=N_test,
        seed=seed,
        feature_set=feature_set_norm,
        sigma_in=sigma_true if feature_set_norm == "D" and sigma_in_eval is None else sigma_in_eval,
    )
    t_grid = reference_dataset["t_grid"]
    S_te = reference_dataset["S_te"].astype(np.float32)
    Z_te = reference_dataset["Z_te"].astype(np.float32)

    S_tr, Z_tr, F_tr_raw = _make_mixture_split(
        split_name="train",
        S0=S0,
        robust_sigmas=sigma_values,
        T=T,
        n=n,
        K=K,
        total_count=N_train,
        seed=seed,
        feature_set=feature_set_norm,
    )
    S_va, Z_va, F_va_raw = _make_mixture_split(
        split_name="val",
        S0=S0,
        robust_sigmas=sigma_values,
        T=T,
        n=n,
        K=K,
        total_count=N_val,
        seed=seed,
        feature_set=feature_set_norm,
    )

    sigma_eval = sigma_true if sigma_in_eval is None else sigma_in_eval
    F_te_raw = make_features(
        t_grid,
        S_te,
        K=K,
        feature_set=feature_set_norm,
        sigma_in=sigma_eval if feature_set_norm == "D" else None,
    )

    mu, sd = fit_feature_norm(F_tr_raw)
    F_tr = apply_feature_norm(F_tr_raw, mu, sd)
    F_va = apply_feature_norm(F_va_raw, mu, sd)
    F_te = apply_feature_norm(F_te_raw, mu, sd)

    feature_norm = {
        "feature_set": feature_set_norm,
        "feature_names": list(feature_names(feature_set_norm)),
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
        "feature_set": feature_set_norm,
        "feature_names": list(feature_names(feature_set_norm)),
        "feature_dim": feature_dim(feature_set_norm),
        "feature_norm": feature_norm,
        "reference_Z_tr": reference_dataset["Z_tr"],
        "robust_sigmas": list(sigma_values),
    }


def _slice_sigma_input(
    *,
    sigma_in: float | np.ndarray | None,
    split_idx: np.ndarray,
    sigma_default: float,
) -> float | np.ndarray:
    if sigma_in is None:
        return float(sigma_default)

    sigma_arr = np.asarray(sigma_in, dtype=np.float32)
    if sigma_arr.ndim == 0:
        return float(sigma_arr)
    if sigma_arr.ndim == 1:
        return sigma_arr[split_idx]
    if sigma_arr.ndim == 2:
        return sigma_arr[split_idx]
    raise ValueError(f"sigma_in must be scalar, 1-D, or 2-D; got shape {sigma_arr.shape}")


def _make_mixture_split(
    *,
    split_name: str,
    S0: float,
    robust_sigmas: Sequence[float],
    T: float,
    n: int,
    K: float,
    total_count: int,
    seed: int,
    feature_set: FeatureSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = _balanced_counts(total_count, len(robust_sigmas))
    S_parts: list[np.ndarray] = []
    Z_parts: list[np.ndarray] = []
    F_parts: list[np.ndarray] = []
    t_grid_ref: np.ndarray | None = None

    for sigma_idx, (sigma_value, count) in enumerate(zip(robust_sigmas, counts)):
        if count <= 0:
            continue
        split_seed = _mixture_seed(seed=seed, split_name=split_name, sigma_idx=sigma_idx)
        t_grid, S_paths = simulate_gbm_discounted_paths(S0, float(sigma_value), T, n, int(count), split_seed)
        if t_grid_ref is None:
            t_grid_ref = t_grid.astype(np.float32)
        sigma_in = float(sigma_value) if feature_set == "D" else None
        S_split = S_paths.astype(np.float32)
        Z_split = payoff_call(S_split[:, -1], K).astype(np.float32)
        F_split = make_features(t_grid, S_split, K=K, feature_set=feature_set, sigma_in=sigma_in)
        S_parts.append(S_split)
        Z_parts.append(Z_split)
        F_parts.append(F_split)

    if not S_parts or t_grid_ref is None:
        raise ValueError(f"{split_name} split requires a positive number of paths")

    S = np.concatenate(S_parts, axis=0).astype(np.float32)
    Z = np.concatenate(Z_parts, axis=0).astype(np.float32)
    F = np.concatenate(F_parts, axis=0).astype(np.float32)

    rng = np.random.default_rng(_shuffle_seed(seed=seed, split_name=split_name))
    order = rng.permutation(S.shape[0])
    return S[order], Z[order], F[order]


def _balanced_counts(total: int, num_buckets: int) -> list[int]:
    base = int(total) // int(num_buckets)
    rem = int(total) % int(num_buckets)
    counts = [base] * int(num_buckets)
    for idx in range(rem):
        counts[idx] += 1
    return counts


def _mixture_seed(*, seed: int, split_name: str, sigma_idx: int) -> int:
    split_offsets = {"train": 10_000, "val": 20_000}
    if split_name not in split_offsets:
        raise ValueError(f"unsupported split_name {split_name!r}")
    return int(seed) + split_offsets[split_name] + (1_000 * int(sigma_idx))


def _shuffle_seed(*, seed: int, split_name: str) -> int:
    split_offsets = {"train": 30_000, "val": 40_000}
    if split_name not in split_offsets:
        raise ValueError(f"unsupported split_name {split_name!r}")
    return int(seed) + split_offsets[split_name]


def _broadcast_sigma_feature(
    *,
    sigma_in: float | np.ndarray | None,
    sigma_default: float | None,
    N: int,
    n: int,
) -> np.ndarray:
    sigma_value = sigma_default if sigma_in is None else sigma_in
    if sigma_value is None:
        raise ValueError("sigma_in must be provided for feature set D")

    sigma_arr = np.asarray(sigma_value, dtype=np.float32)
    if sigma_arr.ndim == 0:
        return np.full((N, n), float(sigma_arr), dtype=np.float32)
    if sigma_arr.shape == (N,):
        return np.broadcast_to(sigma_arr[:, None], (N, n)).astype(np.float32)
    if sigma_arr.shape == (N, n):
        return sigma_arr.astype(np.float32)
    raise ValueError(f"sigma_in could not be broadcast to shape {(N, n)}; got {sigma_arr.shape}")
