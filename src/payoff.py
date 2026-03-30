import numpy as np

def payoff_call(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)

def payoff_put(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0.0)
