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

## Premium convention

`p0 = bs_call_price_discounted(0, S0, K, sigma_true, T)` — BS analytical price, not MC estimate.
