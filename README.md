# thesis_code

GBM deep hedging benchmark for MMath dissertation Section 6.2.
Compares Black-Scholes delta hedging against neural-network deep hedging
for a European ATM call under proportional transaction costs and volatility
misspecification.

## Quick start

    ./tools/clean.sh
    ./tools/compile.sh
    ./tools/smoke.sh
    ./tools/guard.sh

## Full benchmark

    python -m src.run_benchmark_gbm_grid \
      --config configs/gbm_benchmark.yaml \
      --sigma-bars 0.10,0.15,0.20,0.25,0.30 \
      --lambda-costs 0,1e-4,5e-4,1e-3 \
      --seeds 0,1,2,3,4,5,6,7,8,9 \
      --training-regimes oracle,robust

    python -m src.rebuild_benchmark_statistics \
      --config configs/gbm_benchmark.yaml

## Project structure

    src/
      models_gbm.py                  — GBM path simulation
      world_gbm.py                   — dataset construction, feature engineering, normalization
      bs.py                          — Black-Scholes price and delta (vectorized)
      strategies_delta.py            — BS delta strategy on a grid
      costs_and_pl.py                — P&L and turnover computation (NumPy)
      hedge_core.py                  — P&L and rollout (PyTorch, used in training)
      deep_hedging_model.py          — MLP hedge policy
      objectives.py                  — CVaR/ES, entropic, mean-variance objectives
      train_loop.py                  — training loop (no I/O)
      train_deephedge_gbm.py         — single-run orchestrator
      run_benchmark_gbm_grid.py      — multi-seed benchmark grid runner
      run_benchmark_eval_only.py     — eval-only runner (reuses trained checkpoints)
      rebuild_benchmark_statistics.py — aggregation, paired tests, win summary
      run_gbm_baseline.py            — BS-delta baseline runner
      eval.py                        — per-run metrics and plots
      metrics.py                     — scalar risk metrics
      plots.py                       — plotting utilities
      benchmark_repro.py             — reproducibility metadata and manifest
      paths.py                       — canonical path helpers
      logging_utils.py               — JSON/CSV writing
      config.py                      — YAML config loader
      payoff.py                      — option payoff functions
      tools_cli.py                   — clean/compile/smoke/guard CLI
      ablation_regularization.py     — regularization ablation study

    configs/
      gbm_es95.yaml                  — smoke/CI config (small data, fast)
      gbm_benchmark.yaml             — full benchmark config (50K train, 100K test)

    tools/
      clean.sh, compile.sh, smoke.sh, guard.sh

## Outputs

Smoke run writes to `results/gbm_deephedge/`:

    metrics_bs.json, metrics_nn.json, train_log.csv,
    best_state.pt, last_state.pt, feature_norm.json,
    arrays_debug.npz, run_cfg.json, plots

Full benchmark writes to `results/gbm_deephedge/benchmark_6_2/`:

    runs/         — per-scenario artifacts
    aggregate/    — seed_level_metrics.csv, paired_comparisons.csv,
                    win_summary.csv, scenario_summary.csv

## Guard

Guard compares current NN metrics against the archived baseline.
Lower is better: `std_PL`, `ES_loss_0.95`, `VaR_loss_0.95`, `ES_loss_0.99`, `VaR_loss_0.99`.
If any metric worsens, guard fails.

## Config

Edit `configs/gbm_es95.yaml` for smoke runs.
Edit `configs/gbm_benchmark.yaml` for the full benchmark.
Do not change hedging math without re-running guard.
