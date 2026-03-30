import os
import json
import numpy as np

from src.models_gbm import simulate_gbm_discounted_paths
from src.payoff import payoff_call
from src.bs import bs_call_price_discounted
from src.strategies_delta import bs_delta_strategy_paths
from src.costs_and_pl import pl_paths_proportional_costs
from src.metrics import summary_metrics
from src.plots import plot_hist, plot_es_var_bars

def main():
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
    train_idx = idx[:N_train]
    val_idx = idx[N_train:N_train+N_val]
    test_idx = idx[N_train+N_val:]

    S_test = S_paths[test_idx]
    ST_test = S_test[:, -1]
    Z_test = payoff_call(ST_test, K)

    p0_true_mc = float(np.mean(Z_test))
    p0_bs = bs_call_price_discounted(0.0, S0, K, sigma_bar, T)

    deltas_bs = bs_delta_strategy_paths(t_grid, S_test, K, sigma_bar, T)

    PL_bs_mcprice = pl_paths_proportional_costs(S_test, deltas_bs, Z_test, p0_true_mc, lam_cost)
    PL_bs_bsprice = pl_paths_proportional_costs(S_test, deltas_bs, Z_test, p0_bs, lam_cost)

    alpha_list = (0.95, 0.99)
    m_bs_mc = summary_metrics(PL_bs_mcprice, alpha_list=alpha_list, lam_entropic=1.0)
    m_bs_bs = summary_metrics(PL_bs_bsprice, alpha_list=alpha_list, lam_entropic=1.0)

    outdir = "results/gbm_baseline"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metrics_bs_mcprice.json"), "w") as f:
        json.dump(m_bs_mc, f, indent=2)
    with open(os.path.join(outdir, "metrics_bs_bsprice.json"), "w") as f:
        json.dump(m_bs_bs, f, indent=2)

    plot_hist(PL_bs_mcprice, PL_bs_bsprice, "BS-delta (p0=MC)", "BS-delta (p0=BS)", os.path.join(outdir, "hist_pl_bs_mc_vs_bs.png"))
    plot_es_var_bars(m_bs_mc, m_bs_bs, alpha_list, os.path.join(outdir, "tail_metrics_bs_mc_vs_bs.png"), title="GBM: BS-delta tail metrics (two p0 choices)")

    print("Saved results to:", outdir)
    print("BS-delta with p0=MC price:", m_bs_mc)
    print("BS-delta with p0=BS price:", m_bs_bs)

if __name__ == "__main__":
    main()
