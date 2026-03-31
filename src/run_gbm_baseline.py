from __future__ import annotations

import numpy as np

from src.bs import bs_call_price_discounted
from src.costs_and_pl import pl_paths_proportional_costs
from src.logging_utils import write_json_file
from src.metrics import summary_metrics
from src.models_gbm import simulate_gbm_discounted_paths
from src.paths import (
    baseline_hist_plot_path,
    baseline_metrics_bsprice_path,
    baseline_metrics_mcprice_path,
    baseline_tail_plot_path,
    get_baseline_dir,
)
from src.payoff import payoff_call
from src.plots import plot_es_var_bars, plot_hist
from src.strategies_delta import bs_delta_strategy_paths


def main() -> None:
    S0 = 1.0
    T = 1.0
    n = 50
    sigma_true = 0.2
    sigma_bar = 0.2
    lam_cost = 0.0

    N_train = 5000
    N_val = 1000
    N_test = 2000
    seed = 1234

    K = S0

    t_grid, S_paths = simulate_gbm_discounted_paths(S0, sigma_true, T, n, N_train + N_val + N_test, seed)

    idx = np.arange(S_paths.shape[0])
    test_idx = idx[N_train + N_val :]

    S_test = S_paths[test_idx]
    ST_test = S_test[:, -1]
    Z_test = payoff_call(ST_test, K)

    p0_true_mc = float(np.mean(Z_test))
    p0_bs = bs_call_price_discounted(0.0, S0, K, sigma_bar, T)

    deltas_bs = bs_delta_strategy_paths(t_grid, S_test, K, sigma_bar, T)

    pl_bs_mcprice = pl_paths_proportional_costs(S_test, deltas_bs, Z_test, p0_true_mc, lam_cost)
    pl_bs_bsprice = pl_paths_proportional_costs(S_test, deltas_bs, Z_test, p0_bs, lam_cost)

    alpha_list = (0.95, 0.99)
    m_bs_mc = summary_metrics(pl_bs_mcprice, alpha_list=alpha_list, lam_entropic=1.0)
    m_bs_bs = summary_metrics(pl_bs_bsprice, alpha_list=alpha_list, lam_entropic=1.0)

    run_dir = get_baseline_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json_file(baseline_metrics_mcprice_path(run_dir), m_bs_mc)
    write_json_file(baseline_metrics_bsprice_path(run_dir), m_bs_bs)

    plot_hist(
        pl_bs_mcprice,
        pl_bs_bsprice,
        "BS-delta (p0=MC)",
        "BS-delta (p0=BS)",
        baseline_hist_plot_path(run_dir),
    )
    plot_es_var_bars(
        m_bs_mc,
        m_bs_bs,
        alpha_list,
        baseline_tail_plot_path(run_dir),
        title="GBM: BS-delta tail metrics (two p0 choices)",
    )

    print("Saved results to:", run_dir)
    print("BS-delta with p0=MC price:", m_bs_mc)
    print("BS-delta with p0=BS price:", m_bs_bs)


if __name__ == "__main__":
    main()
