# Agents Guidelines (thesis_code)

## Golden rule: never worsen metrics
Before and after any change:
1) run compile
2) run smoke
3) run guard
If guard fails — rollback immediately.

## Repro / determinism
- Keep a fixed default config (configs/gbm_es95.yaml).
- Keep seed controlled by config.
- Do not change math/logic without an experiment plan.

## Output contracts
`src/train_deephedge_gbm.py` must produce:
- results/gbm_deephedge/metrics_nn.json
- results/gbm_deephedge/metrics_bs.json
- results/gbm_deephedge/train_log.csv
- results/gbm_deephedge/best_state.pt
- results/gbm_deephedge/last_state.pt
- results/gbm_deephedge/feature_norm.json
- results/gbm_deephedge/arrays_debug.npz (optional but preferred)

## Structure targets (Section 6.2)
- objectives.py: objectives (CVaR/ES/entropic)
- world_gbm.py: data generation + split + feature normalization + save feature_norm.json
- eval.py: metrics + plots + arrays_debug.npz + run_cfg.json
- hedge_core.py: rollout + PL computation shared between training loop and runner
- train_loop.py: pure training loop (no circular imports), returns best/last/log

## Coding style
- No zsh-triggering comments in terminal scripts.
- Keep scripts in tools/ and make them executable.
- Prefer full-file edits over fragile regex patches.
- Always keep guard passing on main branch.
