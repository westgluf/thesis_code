import numpy as np

def pl_paths_proportional_costs(
    S_paths: np.ndarray,
    deltas: np.ndarray,
    Z: np.ndarray,
    p0: float,
    lam: float,
) -> np.ndarray:
    """
    S_paths: (N, n+1)
    deltas:  (N, n)   delta_k held on [t_k, t_{k+1})
    Z:       (N,)     payoff at maturity (discounted)
    Proportional costs: c_k = lam * S_k * |delta_k - delta_{k-1}|
    with delta_-1 = 0 and delta_n = 0 (close at maturity).
    """
    N, n_plus_1 = S_paths.shape
    n = n_plus_1 - 1

    dS = S_paths[:, 1:] - S_paths[:, :-1]          # (N, n)
    gains = np.sum(deltas * dS, axis=1)            # (N,)

    delta_prev = np.concatenate([np.zeros((N, 1)), deltas[:, :-1]], axis=1)  # delta_{k-1}
    trade = deltas - delta_prev                                        # (N, n)
    costs = np.sum(lam * S_paths[:, :-1] * np.abs(trade), axis=1)      # pay costs at t_k

    close_cost = lam * S_paths[:, -1] * np.abs(deltas[:, -1])          # close to 0 at T
    costs = costs + close_cost

    PL = -Z + p0 + gains - costs
    return PL


def turnover_paths(deltas: np.ndarray) -> np.ndarray:
    """
    Unweighted turnover per path:
      |delta_0| + sum_k |delta_k - delta_{k-1}| + |delta_{n-1}|
    matching the open/adjust/close structure used in proportional cost accounting.
    """
    N, n = deltas.shape
    delta_prev = np.concatenate([np.zeros((N, 1), dtype=deltas.dtype), deltas[:, :-1]], axis=1)
    trade = np.abs(deltas - delta_prev)
    close = np.abs(deltas[:, -1])
    return trade.sum(axis=1) + close
