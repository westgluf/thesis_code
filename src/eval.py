from __future__ import annotations

from pathlib import Path

import numpy as np

from src.logging_utils import write_json_file
from src.metrics import summary_metrics
from src.paths import arrays_debug_path, hist_plot_path, metrics_bs_path, metrics_nn_path, tail_plot_path
from src.plots import plot_hist, plot_es_var_bars


def save_eval_artifacts(
    run_dir: str | Path,
    pl_bs,
    pl_nn,
    label_bs: str,
    label_nn: str,
    alpha_list=(0.95, 0.99),
    lam_entropic: float = 1.0,
    arrays_debug: dict | None = None,
):
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_bs = summary_metrics(pl_bs, alpha_list=alpha_list, lam_entropic=lam_entropic)
    m_nn = summary_metrics(pl_nn, alpha_list=alpha_list, lam_entropic=lam_entropic)

    write_json_file(metrics_bs_path(out_dir), m_bs)
    write_json_file(metrics_nn_path(out_dir), m_nn)

    if arrays_debug is not None:
        payload = {}
        for key, value in arrays_debug.items():
            if hasattr(value, "detach"):
                payload[key] = value.detach().cpu().numpy()
            else:
                payload[key] = np.asarray(value)
        np.savez(arrays_debug_path(out_dir), **payload)

    plot_hist(pl_bs, pl_nn, label_bs, label_nn, hist_plot_path(out_dir))
    plot_es_var_bars(
        m_bs,
        m_nn,
        alpha_list,
        tail_plot_path(out_dir),
        title="GBM: BS-delta vs Deep hedging",
    )

    return m_bs, m_nn
