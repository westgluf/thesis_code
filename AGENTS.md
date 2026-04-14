# AGENTS.md — Working rules for Codex

## Hard rules

1. Run `./tools/clean.sh`, `./tools/compile.sh`, `./tools/smoke.sh`, `./tools/guard.sh` before declaring success.
2. If guard worsens, rollback immediately.
3. Do not change hedging math (P&L, ES/VaR, objectives, features) unless explicitly requested.
4. Avoid circular imports. Shared math goes in `src/hedge_core.py`.
5. No shell-breaking comments in zsh snippets.
6. Keep code clean, modular, and package-safe. Path and artifact names belong in shared helpers (`src/paths.py`) rather than inline strings.

## Architecture

- `src/train_loop.py` — strict keyword-only API, no file I/O, returns `(best_state, last_state, train_log)`.
- `src/objectives.py` — modular objective (CVaR, entropic, mean-variance), created once in runner.
- `src/world_gbm.py` — dataset generation + split + normalization + `feature_norm.json`.
- `src/hedge_core.py` — rollout + PL computation (shared by train_loop + runner).
- `src/eval.py` — metrics + plots + `arrays_debug.npz`.
- `src/train_deephedge_gbm.py` — thin runner: load config, build data, build model, call `train_loop`, call eval.
- `src/paths.py` — all artifact paths.
- `src/logging_utils.py` — deterministic JSON/CSV writing.

## Output contract

Runner must create in `results/gbm_deephedge`:
- metrics files, plots, optional `arrays_debug.npz`
- checkpoints `best_state.pt`, `last_state.pt`
- `train_log.csv` with header: `epoch, train_loss, val_loss, lr, w`
- `run_cfg.json`

## Guard metrics (lower is better)

`std_PL`, `ES_loss_0.95`, `VaR_loss_0.95`, `ES_loss_0.99`, `VaR_loss_0.99`

## Premium convention (two regimes)

The premium `p_0` used when computing terminal P&L follows two
conventions depending on the data-generating model:

1. **GBM benchmark (Section 6.2)**:
   `p_0 = bs_call_price_discounted(0, S_0, K, sigma_true, T)` —
   the Black-Scholes analytical price under the true (and assumed)
   volatility. Exact because the model is complete and has closed
   form. Implemented in `src/run_benchmark_gbm_grid.py` and related
   GBM scripts.

2. **Rough Bergomi (Section 6.3, Section 7)**:
   `p_0 = float(compute_payoff(S_train, K, "call").mean())` —
   a Monte Carlo estimate of `E_true[Z]` on the training paths. Used
   because rough Bergomi has no closed-form European-call price.
   Implemented in `run_unified_baseline.py`, `run_section_6_3_baseline.py`,
   `diagnostic_controls.py`, `h2_grid_extension.py`, and other rough-
   Bergomi experiments.

Both conventions are consistent with thesis Definition 4.16. The
thesis footnote in Section 6.1 makes the two-regime distinction
explicit to the reader.

Never compute the MC premium on test paths — always on an independent
training set — to avoid information leakage from test to premium.
