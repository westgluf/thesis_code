# Experiment 6.2

This repo contains the GBM deep hedging experiment used for thesis section 6.2. The current task keeps the hedging math fixed and focuses on structure, reproducibility, and guard-railed execution.

## Purpose

- `src/run_gbm_baseline.py` runs the BS-delta baseline and archives the baseline guard metrics.
- `src/train_deephedge_gbm.py` trains the deep hedging model, writes checkpoints and logs, and saves evaluation artifacts.
- `src/tools_cli.py` backs the shell scripts and package entry points for clean, compile, smoke, and guard.

## How To Run

From the repo root:

```bash
./tools/clean.sh
./tools/compile.sh
./tools/smoke.sh
./tools/guard.sh
```

If the package is installed, the equivalent entry points are:

```bash
thesis-code-baseline-gbm
thesis-code-train-gbm
thesis-code-smoke
thesis-code-guard
```

## Expected Outputs

Training writes the experiment outputs into the run directory resolved from `out_dir` in `configs/gbm_es95.yaml`:

- `metrics_bs.json`
- `metrics_nn.json`
- `hist_pl_bs_vs_nn.png`
- `tail_metrics_bs_vs_nn.png`
- `arrays_debug.npz`
- `feature_norm.json`
- `best_state.pt`
- `last_state.pt`
- `train_log.csv`
- `run_cfg.json`

Baseline outputs are written to `results/gbm_baseline`, and the baseline guard reference is archived under `results/archive/gbm_baseline_metrics_*.json`.

## Safe Config Changes

- Change only config values in `configs/gbm_es95.yaml` when you want a new run configuration.
- The run directory is computed once from `out_dir` and then reused for every artifact in that run.
- Save-format and path logic live in `src/paths.py` and `src/logging_utils.py`; prefer extending those helpers instead of adding new file names inline.

## Reproducibility Notes

- The experiment seeds NumPy and PyTorch from config before training.
- `run_cfg.json` is written before training starts so each run records the exact config used.
- `train_log.csv` uses a fixed schema: `epoch,train_loss,val_loss,lr,w`.
- Guard compares the current deep hedging metrics against the latest archived baseline and must not allow any worse value for `std_PL`, `ES_loss_0.95`, `VaR_loss_0.95`, `ES_loss_0.99`, or `VaR_loss_0.99`.
