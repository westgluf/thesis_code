AGENT RULES FOR THIS REPO (thesis_code_copy)

Goal
- Keep Section 6.2 codebase clean, reproducible, and regression-proof.
- Any change must preserve (or improve) the risk metrics enforced by Guard.

Golden rule
- Run Guard before and after changes. If Guard fails, revert immediately.

Mandatory commands
- Compile: ./tools/compile.sh
- Clean:   ./tools/clean.sh
- Smoke:   ./tools/smoke.sh
- Guard:   python tools/guard_train_gbm.py

Definition of “done”
- compile OK
- smoke OK
- guard PASS (no metric regressions vs baseline)

Change policy
- Prefer large, clean edits (replace whole files) over fragile regex/patches.
- Avoid circular imports. Shared logic must live in dedicated modules (e.g., hedge_core.py).
- Keep train_deephedge_gbm.py as a thin runner:
  - load config
  - build data via world_gbm
  - build model
  - call train_loop
  - call eval.save_eval_artifacts
  - save run_cfg.json

Reproducibility
- Always seed via config.
- Save artifacts to results/gbm_deephedge:
  - metrics_bs.json, metrics_nn.json
  - hist_pl_bs_vs_nn.png, tail_metrics_bs_vs_nn.png
  - arrays_debug.npz (optional)
  - feature_norm.json
  - best_state.pt, last_state.pt
  - train_log.csv

No-shell-footguns
- Do not print shell comments that start with # as commands.
- Provide runnable bash blocks only.

Style
- Keep functions small.
- Explicit names, no magic globals.
- Validate inputs (NaN/Inf checks for feature norms).
