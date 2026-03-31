# AGENTS.md — Working rules for Codex

## Hard rules
1) Run `./tools/clean.sh`, `./tools/compile.sh`, `./tools/smoke.sh`, `./tools/guard.sh` before declaring success.
2) If guard worsens, revert immediately.
3) Do not add shell snippets with comment-only lines that can be pasted into zsh and fail.
4) Avoid circular imports. Shared math goes into a core module (e.g. `src/hedge_core.py`).
5) Keep training math unchanged unless user explicitly requests model improvements.

## Preferred architecture
- `src/world_gbm.py` — dataset generation + split + feature normalization + save `feature_norm.json`.
- `src/objectives.py` — objective modules (CVaR/ES threshold param).
- `src/hedge_core.py` — rollout + PL computation (shared by train_loop + runner).
- `src/train_loop.py` — strict training loop (no file I/O).
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