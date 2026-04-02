import numpy as np
from .bs import bs_call_delta


def bs_delta_strategy_paths(
    t_grid: np.ndarray,
    S_paths: np.ndarray,
    K: float,
    sigma_bar: float,
    T: float,
) -> np.ndarray:
    """
    Returns deltas of shape (N, n) for holding on [t_k, t_{k+1})
    """
    N, n_plus_1 = S_paths.shape
    n = n_plus_1 - 1
    t_col = t_grid[:n]  # shape (n,)
    deltas = bs_call_delta(t_col[np.newaxis, :], S_paths[:, :n], K, sigma_bar, T)
    return np.asarray(deltas, dtype=np.float64)
