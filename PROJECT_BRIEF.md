# Project Brief — thesis_code (Section 6.2)

## Goal
Maintain a clean, reproducible deep hedging experiment under GBM for thesis section 6.2.
Primary constraint: **do not worsen risk metrics** (guarded regression).

## Non-negotiables
- Every code change must pass:
  - `./tools/clean.sh`
  - `./tools/compile.sh`
  - `./tools/smoke.sh`
  - `./tools/guard.sh`
- If guard metrics worsen, revert the change (rollback).
- No shell-breaking comments in command snippets (avoid lines starting with `#` in copy-paste blocks for zsh sessions).
- Keep training math unchanged unless explicitly requested.

## Current baseline artifacts (must be produced by smoke)
- `results/gbm_deephedge/metrics_bs.json`
- `results/gbm_deephedge/metrics_nn.json`
- `results/gbm_deephedge/train_log.csv`
- `results/gbm_deephedge/best_state.pt`
- `results/gbm_deephedge/last_state.pt`
- `results/gbm_deephedge/feature_norm.json`
- `results/gbm_deephedge/arrays_debug.npz` (optional but helpful)
- `results/gbm_deephedge/run_cfg.json`

## Guard definition (lower is better)
Compare current run vs latest baseline in `results/archive/gbm_baseline_metrics_*.json` for:
- std_PL
- ES_loss_0.95
- VaR_loss_0.95
- ES_loss_0.99
- VaR_loss_0.99

Guard must not worsen any of the above.

## Design principles
- `train_loop` is a strict keyword-only API:
  - no file I/O inside training loop
  - no "extra kwargs"
  - returns: `best_state`, `last_state`, `train_log`
- Objective is modular (`src/objectives.py`) and created once in the runner.
- Dataset generation/splitting/normalization is modular (`src/world_gbm.py`), also responsible for `feature_norm.json`.
- Evaluation & artifact writing is modular (`src/eval.py`).
- Runner `src/train_deephedge_gbm.py` is thin:
  - load config
  - build data
  - build model+optimizer+objective
  - call train_loop
  - call eval/artifact saving

## Definition of "done" for refactors
- No behavior change in hedging math (same objective, same P&L computation, same features).
- Same outputs exist in `results/gbm_deephedge`.
- Guard passes.

## How to work
- Make changes in small, reviewable commits.
- After each commit: run `./tools/smoke.sh` and `./tools/guard.sh`.
- Prefer editing whole files over fragile regex patching.
