import numpy as np
from scipy.stats import norm

def bs_call_price_discounted(t: float, s: float, K: float, sigma: float, T: float) -> float:
    tau = max(T - t, 1e-12)
    d1 = (np.log(s / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return float(s * norm.cdf(d1) - K * norm.cdf(d2))

def bs_call_delta(t: float, s: float, K: float, sigma: float, T: float) -> float:
    tau = max(T - t, 1e-12)
    d1 = (np.log(s / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return float(norm.cdf(d1))
