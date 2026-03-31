# AGENTS.md — Working rules for Codex

## Hard rules
1) Run `./tools/clean.sh`, `./tools/compile.sh`, `./tools/smoke.sh`, `./tools/guard.sh` before declaring success.
2) Always run guard for experiment 6.2 work. If guard worsens, rollback immediately.
3) Do not add shell snippets with comment-only lines that can be pasted into zsh and fail.
4) Avoid circular imports. Shared math goes into a core module (e.g. `src/hedge_core.py`).
5) Do not change experiment 6.2 hedging math in structural tasks. Dataset generation, PL math, ES/VaR formulas, and baseline evaluation must stay untouched unless the user explicitly requests model changes.
6) Never worsen guard metrics without rollback.
7) Keep code clean, modular, and package-safe. Path and artifact names belong in shared helpers rather than inline strings.

## Preferred architecture
- `src/world_gbm.py` — dataset generation + split + feature normalization + save `feature_norm.json`.
- `src/objectives.py` — objective modules (CVaR/ES threshold param).
- `src/hedge_core.py` — rollout + PL computation (shared by train_loop + runner).
- `src/train_loop.py` — strict training loop (no file I/O).
- `src/paths.py` — results root, run directory, and artifact path helpers.
- `src/logging_utils.py` — deterministic `run_cfg.json` + `train_log.csv` writing.
- `src/eval.py` — metrics + plots + arrays_debug.npz.
- `src/train_deephedge_gbm.py` — thin runner reading config and orchestrating.

## Output contract
Runner must create in `results/gbm_deephedge`:
- metrics files, plots, optional arrays_debug
- checkpoints `best_state.pt`, `last_state.pt`
- `train_log.csv` with header: epoch, train_loss, val_loss, lr, w
- `run_cfg.json`

## Guard metrics
Lower is better:
std_PL, ES_loss_0.95, VaR_loss_0.95, ES_loss_0.99, VaR_loss_0.99
