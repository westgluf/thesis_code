import numpy as np

def var_alpha(x: np.ndarray, alpha: float) -> float:
    return float(np.quantile(x, alpha))

def es_alpha(loss: np.ndarray, alpha: float) -> float:
    """
    Expected shortfall of LOSS at level alpha, i.e. average of worst (1-alpha) tail.
    loss = -PL typically.
    """
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    if tail.size == 0:
        return float(q)
    return float(tail.mean())

def entropic_risk(pl: np.ndarray, lam: float) -> float:
    """
    Entropic risk of PL: (1/lam) log E[exp(-lam PL)]
    """
    m = np.max(-lam * pl)
    return float((np.log(np.mean(np.exp(-lam * pl - m))) + m) / lam)

def summary_metrics(pl: np.ndarray, alpha_list=(0.95, 0.99), lam_entropic=1.0) -> dict:
    loss = -pl
    out = {
        "mean_PL": float(np.mean(pl)),
        "std_PL": float(np.std(pl, ddof=1)),
        "entropic": entropic_risk(pl, lam_entropic),
    }
    for a in alpha_list:
        out[f"VaR_loss_{a:.2f}"] = float(np.quantile(loss, a))
        out[f"ES_loss_{a:.2f}"] = es_alpha(loss, a)
    return out
