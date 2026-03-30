import numpy as np

def simulate_gbm_discounted_paths(
    S0: float,
    sigma: float,
    T: float,
    n: int,
    N: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discounted GBM: dS_t = sigma S_t dW_t
    Returns:
      t_grid: shape (n+1,)
      S_paths: shape (N, n+1)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = np.diff(t)  # shape (n,)

    Z = rng.standard_normal(size=(N, n))
    increments = (-0.5 * sigma**2 * dt)[None, :] + (sigma * np.sqrt(dt))[None, :] * Z
    logS = np.cumsum(increments, axis=1)
    logS = np.concatenate([np.zeros((N, 1)), logS], axis=1)

    S = S0 * np.exp(logS)
    return t, S
