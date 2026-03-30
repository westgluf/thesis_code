import os
import json
import numpy as np

from src.metrics import summary_metrics
from src.plots import plot_hist, plot_es_var_bars

def save_eval_artifacts(
    out_dir: str,
    pl_bs,
    pl_nn,
    label_bs: str,
    label_nn: str,
    alpha_list=(0.95, 0.99),
    lam_entropic: float = 1.0,
    arrays_debug: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    m_bs = summary_metrics(pl_bs, alpha_list=alpha_list, lam_entropic=lam_entropic)
    m_nn = summary_metrics(pl_nn, alpha_list=alpha_list, lam_entropic=lam_entropic)

    with open(os.path.join(out_dir, "metrics_bs.json"), "w") as f:
        json.dump(m_bs, f, indent=2)
    with open(os.path.join(out_dir, "metrics_nn.json"), "w") as f:
        json.dump(m_nn, f, indent=2)

    plot_hist(pl_bs, pl_nn, label_bs, label_nn, os.path.join(out_dir, "hist_pl_bs_vs_nn.png"))
    plot_es_var_bars(m_bs, m_nn, alpha_list, os.path.join(out_dir, "tail_metrics_bs_vs_nn.png"), title="GBM: BS-delta vs Deep hedging")

    if arrays_debug is not None:
        try:
            out_npz = os.path.join(out_dir, "arrays_debug.npz")
            payload = {}
            for k, v in arrays_debug.items():
                if hasattr(v, "detach"):
                    payload[k] = v.detach().cpu().numpy()
                else:
                    payload[k] = np.asarray(v)
            np.savez(out_npz, **payload)
        except Exception as e:
            print("Warning: could not save arrays_debug.npz:", e)

    return m_bs, m_nn
