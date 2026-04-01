# Section 6.2 Reproducibility and Artifact-Saving Standard

- Objective:
  - Define a research-grade storage and metadata standard for the Section 6.2 GBM deep hedging benchmark.
  - Ensure every experiment can be reproduced exactly from saved artifacts without ambiguity about configuration, seeds, code version, or aggregation logic.
  - Extend the current single-run Section 6.2 output contract into a multi-scenario, multi-seed benchmark layout.

- Design principles:
  - Every run must be self-describing.
  - Every saved artifact must have a deterministic path and filename.
  - Aggregate tables and plots must be rebuildable from stored per-run outputs.
  - Raw pathwise arrays must be saved for any result used in the dissertation.
  - The saved config must reflect the effective config after defaults are resolved, not only the user-edited input file.
  - Reproducibility metadata must be written before training starts.

- `1. Directory structure`
  - Recommended benchmark root:
    - `results/gbm_deephedge/benchmark_6_2/`
  - Top-level layout:
    - `results/gbm_deephedge/benchmark_6_2/spec/`
    - `results/gbm_deephedge/benchmark_6_2/runs/`
    - `results/gbm_deephedge/benchmark_6_2/aggregate/`
    - `results/gbm_deephedge/benchmark_6_2/paper/`
    - `results/gbm_deephedge/benchmark_6_2/archive/`
  - `spec/` contents:
    - immutable copies of the benchmark protocol, feature specification, and objective specification used for the study
    - benchmark-wide config templates
    - scenario grid definitions
  - `runs/` contents:
    - one directory per scenario-run cell
    - grouping by seed first keeps paired comparisons easy to inspect
  - `aggregate/` contents:
    - merged seed-level tables
    - scenario summaries
    - pairwise tests
    - aggregate plots
    - manifest files
  - `paper/` contents:
    - camera-ready plots and LaTeX tables
    - a small paper manifest describing which outputs were used in the dissertation
  - `archive/` contents:
    - frozen snapshots of aggregate outputs used for submitted drafts or milestones
  - Recommended per-run layout:
    - `results/gbm_deephedge/benchmark_6_2/runs/seed_000/strategy_dh_oracle/objective_cvar95/lambda_1e-4/`
    - `results/gbm_deephedge/benchmark_6_2/runs/seed_000/strategy_bs/sigma_bar_0.20/lambda_1e-4/`
  - Recommended files inside each run directory:
    - `run_meta.json`
    - `run_cfg.json`
    - `dataset_meta.json`
    - `feature_norm.json`
    - `train_log.csv`
    - `best_state.pt`
    - `last_state.pt`
    - `metrics.json`
    - `arrays_debug.npz`
    - `pl_paths.npy`
    - `turnover_paths.npy`
    - `deltas_test.npy`
    - `curve_train_val_loss.png`
    - `hist_pl.png`
    - `tail_metrics.png`
    - `turnover_hist.png`

- `2. File naming conventions`
  - General rules:
    - Use lowercase snake case only.
    - Keep names semantic rather than timestamp-based inside a run directory.
    - Encode scenario identity in the directory path, not by inflating file names.
    - Use fixed filenames for files with unique meaning inside a run.
  - Run directory naming:
    - `seed_{seed:03d}`
    - `strategy_{strategy_name}`
    - `objective_{objective_name}`
    - `sigma_bar_{sigma_bar:.2f}` for BS runs only
    - `lambda_{lambda_cost:g}` or a standardized formatter such as `lambda_1e-4`
  - Fixed filenames within a run:
    - `run_meta.json`
    - `run_cfg.json`
    - `dataset_meta.json`
    - `feature_norm.json`
    - `metrics.json`
    - `train_log.csv`
    - `best_state.pt`
    - `last_state.pt`
    - `pl_paths.npy`
    - `turnover_paths.npy`
    - `deltas_test.npy`
    - `arrays_debug.npz`
    - `curve_train_val_loss.png`
    - `hist_pl.png`
    - `tail_metrics.png`
    - `turnover_hist.png`
  - Aggregate filenames:
    - `aggregate/manifest_runs.csv`
    - `aggregate/manifest_runs.json`
    - `aggregate/seed_level_metrics.csv`
    - `aggregate/seed_level_metrics.parquet`
    - `aggregate/scenario_summary.csv`
    - `aggregate/scenario_summary.tex`
    - `aggregate/pairwise_tests.csv`
    - `aggregate/pairwise_tests.tex`
    - `aggregate/plots/metric_vs_cost.png`
    - `aggregate/plots/tail_risk_vs_cost.png`
    - `aggregate/plots/turnover_vs_cost.png`
  - Frozen publication snapshots:
    - `archive/2026-03-31_draft_v1/`
    - `archive/2026-04-15_submission_v1/`

- `3. Minimal metadata to store in each run`
  - Required run identity fields:
    - `run_id`
    - `benchmark_id`
    - `created_at_utc`
    - `status`
    - `seed`
    - `strategy`
    - `training_regime`
    - `objective_name`
    - `objective_params`
  - Required market and contract fields:
    - `S0`
    - `K`
    - `T`
    - `n_steps`
    - `sigma_true`
    - `sigma_bar`
    - `lambda_cost`
    - `premium_mode`
    - `premium_value`
  - Required data fields:
    - `N_train`
    - `N_val`
    - `N_test`
    - `feature_set`
    - `feature_dim`
    - `normalization_source`
    - `dataset_hash`
  - Required model and optimization fields:
    - `model_name`
    - `model_dim_in`
    - `model_hidden`
    - `model_depth`
    - `optimizer_name`
    - `lr`
    - `weight_decay`
    - `batch_size`
    - `epochs_max`
    - `epochs_trained`
    - `patience`
    - `device`
    - `dtype`
  - Required reproducibility fields:
    - `python_version`
    - `platform`
    - `numpy_version`
    - `torch_version`
    - `git_commit`
    - `git_dirty`
    - `hostname`
    - `working_dir`
    - `config_hash`
    - `code_hash`
  - Required artifact pointers:
    - relative paths to `run_cfg.json`, `metrics.json`, checkpoints, arrays, and plots
  - Recommended completion fields:
    - `best_epoch`
    - `wall_time_sec`
    - `train_exit_reason`
    - `notes`

- `4. Aggregate manifest format`
  - Purpose:
    - provide one machine-readable index of all benchmark runs
    - avoid directory scanning as the primary source of truth
    - support exact reconstruction of aggregate tables from individual runs
  - Recommended manifest files:
    - CSV for easy inspection
    - JSON for typed nested metadata
  - One manifest row per run
  - Required manifest columns:
    - `run_id`
    - `benchmark_id`
    - `seed`
    - `strategy`
    - `training_regime`
    - `objective_name`
    - `feature_set`
    - `sigma_true`
    - `sigma_bar`
    - `lambda_cost`
    - `status`
    - `run_dir`
    - `metrics_path`
    - `config_path`
    - `pl_paths_path`
    - `turnover_paths_path`
    - `checkpoint_best_path`
    - `created_at_utc`
    - `git_commit`
    - `config_hash`
    - `dataset_hash`
  - Recommended manifest-level metadata:
    - `benchmark_id`
    - `spec_version`
    - `generated_at_utc`
    - `generated_by`
    - `num_runs`
    - `num_completed_runs`
    - `manifest_schema_version`

- `5. Ensuring exact reproducibility`
  - Configuration discipline:
    - Save the fully resolved effective config before training starts.
    - Include every implicit default in the saved config.
    - Never rely on unstored command-line defaults or environment-only settings.
  - Seed discipline:
    - Save the master seed and any derived seeds.
    - If separate generators are used, save:
      - `seed_data`
      - `seed_model_init`
      - `seed_batch_order`
      - `seed_eval`
    - Use deterministic seed derivation rules rather than ad hoc calls.
  - Code-version discipline:
    - Save the exact git commit hash.
    - Save whether the worktree was dirty.
    - Save a lightweight code hash over the relevant source files if dirty worktrees are allowed.
  - Environment discipline:
    - Save Python, NumPy, PyTorch, and OS versions.
    - Save CUDA version and deterministic flags when GPU is used.
    - Save device name and dtype.
  - Deterministic execution discipline:
    - Set NumPy and PyTorch seeds explicitly.
    - Use deterministic PyTorch settings when exact reruns matter.
    - Save whether deterministic algorithms were enabled.
    - Avoid nondeterministic data-loading order unless the loader seed is saved.
  - Dataset discipline:
    - Save dataset-generation config separately from training config.
    - Save hashes of test arrays, or save the arrays themselves.
    - For published results, keep raw `S_test`, `Z_test`, and feature normalization artifacts or a dataset archive hash.
  - Artifact discipline:
    - Write `run_meta.json` and `run_cfg.json` first.
    - Write large arrays and checkpoints atomically:
      - write temporary file
      - fsync if needed
      - rename into final path
    - Mark run status as `completed` only after all required artifacts exist.
  - Aggregation discipline:
    - Build aggregate tables only from manifest entries with `status = completed`.
    - Save the manifest snapshot used to produce every published table.
    - Save the aggregation code version or script hash.
  - Publication discipline:
    - Freeze a paper snapshot directory with the exact tables and figures cited in the thesis.
    - Store the manifest and aggregate summary used to produce that snapshot.

- `6. Suggested JSON and YAML schemas`
  - `run_cfg.json` example:
```json
{
  "benchmark_id": "gbm_benchmark_6_2_v1",
  "out_dir": "results/gbm_deephedge/benchmark_6_2/runs/seed_000/strategy_dh_oracle/objective_cvar95/lambda_1e-4",
  "data": {
    "S0": 1.0,
    "K": 1.0,
    "T": 1.0,
    "n": 50,
    "sigma_true": 0.2,
    "sigma_bar": null,
    "lambda_cost": 0.0001,
    "N_train": 50000,
    "N_val": 10000,
    "N_test": 100000,
    "seed": 0
  },
  "feature_set": {
    "name": "state_prev_hedge",
    "dim": 3
  },
  "objective": {
    "name": "cvar",
    "alpha": 0.95
  },
  "train": {
    "epochs": 60,
    "batch_size": 2048,
    "lr": 0.0003,
    "weight_decay": 0.0,
    "patience": 10
  },
  "model": {
    "name": "mlp_hedge",
    "hidden": 128,
    "depth": 4
  }
}
```
  - `run_meta.json` example:
```json
{
  "run_id": "seed000_dh_oracle_cvar95_lambda1e-4",
  "benchmark_id": "gbm_benchmark_6_2_v1",
  "created_at_utc": "2026-03-31T12:45:00Z",
  "status": "completed",
  "seed": 0,
  "strategy": "deep_hedging",
  "training_regime": "oracle",
  "objective_name": "cvar",
  "objective_params": {
    "alpha": 0.95
  },
  "feature_set": "state_prev_hedge",
  "feature_dim": 3,
  "git_commit": "abc123def456",
  "git_dirty": false,
  "python_version": "3.14.0",
  "numpy_version": "2.3.0",
  "torch_version": "2.8.0",
  "platform": "macOS-15-arm64",
  "device": "cpu",
  "dtype": "float32",
  "config_hash": "sha256:4d7f...",
  "dataset_hash": "sha256:9f01...",
  "code_hash": "sha256:889a...",
  "artifacts": {
    "run_cfg": "run_cfg.json",
    "train_log": "train_log.csv",
    "metrics": "metrics.json",
    "best_checkpoint": "best_state.pt",
    "last_checkpoint": "last_state.pt",
    "pl_paths": "pl_paths.npy",
    "turnover_paths": "turnover_paths.npy",
    "deltas_test": "deltas_test.npy",
    "arrays_debug": "arrays_debug.npz",
    "curve_train_val_loss": "curve_train_val_loss.png",
    "hist_pl": "hist_pl.png",
    "tail_metrics": "tail_metrics.png",
    "turnover_hist": "turnover_hist.png"
  }
}
```
  - `metrics.json` example:
```json
{
  "mean_PL": 0.00189,
  "std_PL": 0.01550,
  "VaR_loss_0.95": 0.01931,
  "ES_loss_0.95": 0.02351,
  "VaR_loss_0.99": 0.02567,
  "ES_loss_0.99": 0.03041,
  "mean_turnover": 2.482,
  "max_turnover": 5.337,
  "total_turnover": 248200.4,
  "N_test": 100000
}
```
  - Aggregate manifest JSON example:
```json
{
  "benchmark_id": "gbm_benchmark_6_2_v1",
  "manifest_schema_version": "1.0",
  "generated_at_utc": "2026-03-31T18:00:00Z",
  "num_runs": 200,
  "num_completed_runs": 200,
  "runs": [
    {
      "run_id": "seed000_bs_sigma0.20_lambda1e-4",
      "seed": 0,
      "strategy": "bs",
      "training_regime": "none",
      "objective_name": "none",
      "feature_set": "none",
      "sigma_true": 0.2,
      "sigma_bar": 0.2,
      "lambda_cost": 0.0001,
      "status": "completed",
      "run_dir": "results/gbm_deephedge/benchmark_6_2/runs/seed_000/strategy_bs/sigma_bar_0.20/lambda_1e-4",
      "metrics_path": "results/gbm_deephedge/benchmark_6_2/runs/seed_000/strategy_bs/sigma_bar_0.20/lambda_1e-4/metrics.json",
      "config_hash": "sha256:1abc...",
      "dataset_hash": "sha256:9f01...",
      "git_commit": "abc123def456"
    }
  ]
}
```
  - Benchmark YAML example:
```yaml
benchmark_id: "gbm_benchmark_6_2_v1"
out_root: "results/gbm_deephedge/benchmark_6_2"

data:
  S0: 1.0
  K: 1.0
  T: 1.0
  n: 50
  sigma_true: 0.20
  N_train: 50000
  N_val: 10000
  N_test: 100000

grid:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  sigma_bar: [0.15, 0.20, 0.25]
  lambda_cost: [0.0, 1.0e-4, 5.0e-4, 1.0e-3]
  training_regimes: ["oracle", "robust"]

feature_set:
  name: "state_prev_hedge"

objective:
  name: "cvar"
  alpha: 0.95

train:
  epochs: 60
  batch_size: 2048
  lr: 3.0e-4
  weight_decay: 0.0
  patience: 10

model:
  name: "mlp_hedge"
  hidden: 128
  depth: 4

reproducibility:
  deterministic_torch: true
  save_test_paths: true
  save_pl_arrays: true
  save_turnover_arrays: true
```

- `7. Python config structures`
  - Dataclass-style design:
```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    S0: float
    K: float
    T: float
    n: int
    sigma_true: float
    sigma_bar: float | None
    lambda_cost: float
    N_train: int
    N_val: int
    N_test: int
    seed: int


@dataclass(frozen=True)
class FeatureConfig:
    name: Literal["state_only", "state_prev_hedge", "state_prev_hedge_pathstats", "state_prev_hedge_sigma"]
    dim: int


@dataclass(frozen=True)
class ObjectiveConfig:
    name: Literal["cvar", "entropic", "mean_variance"]
    alpha: float | None = None
    gamma: float | None = None
    lambda_mv: float | None = None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    depth: int


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    out_root: str
    data: DataConfig
    feature_set: FeatureConfig
    objective: ObjectiveConfig
    train: TrainConfig
    model: ModelConfig
```
  - Pydantic-style design:
```python
from typing import Literal
from pydantic import BaseModel, Field


class DataConfigModel(BaseModel):
    S0: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    n: int = Field(..., ge=1)
    sigma_true: float = Field(..., gt=0)
    sigma_bar: float | None = Field(default=None, gt=0)
    lambda_cost: float = Field(..., ge=0)
    N_train: int = Field(..., ge=1)
    N_val: int = Field(..., ge=1)
    N_test: int = Field(..., ge=1)
    seed: int = Field(..., ge=0)


class FeatureConfigModel(BaseModel):
    name: Literal[
        "state_only",
        "state_prev_hedge",
        "state_prev_hedge_pathstats",
        "state_prev_hedge_sigma",
    ]
    dim: int = Field(..., ge=1)


class ObjectiveConfigModel(BaseModel):
    name: Literal["cvar", "entropic", "mean_variance"]
    alpha: float | None = None
    gamma: float | None = None
    lambda_mv: float | None = None


class TrainConfigModel(BaseModel):
    epochs: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    lr: float = Field(..., gt=0)
    weight_decay: float = Field(..., ge=0)
    patience: int = Field(..., ge=0)
    device: str = "cpu"
    dtype: str = "float32"


class ModelConfigModel(BaseModel):
    name: str
    hidden: int = Field(..., ge=1)
    depth: int = Field(..., ge=1)


class BenchmarkConfigModel(BaseModel):
    benchmark_id: str
    out_root: str
    data: DataConfigModel
    feature_set: FeatureConfigModel
    objective: ObjectiveConfigModel
    train: TrainConfigModel
    model: ModelConfigModel
```

- `8. Practical implementation notes for this repo`
  - Existing files that already align with this standard:
    - [src/paths.py](/Users/l/Desktop/thesis_code copy/src/paths.py)
    - [src/logging_utils.py](/Users/l/Desktop/thesis_code copy/src/logging_utils.py)
    - [README_6_2.md](/Users/l/Desktop/thesis_code copy/README_6_2.md)
  - Recommended next implementation steps:
    - extend `src/paths.py` with benchmark-root and per-run path helpers
    - extend `src/logging_utils.py` with `write_run_meta` and manifest writers
    - keep `run_cfg.json` as the effective resolved config
    - split current `arrays_debug.npz` into canonical arrays:
      - `pl_paths.npy`
      - `turnover_paths.npy`
      - `deltas_test.npy`
    - keep `arrays_debug.npz` as an optional convenience bundle, not the only raw-data artifact

- `9. Minimal standard for publication claims`
  - A result is publication-ready only if all of the following exist:
    - exact effective config
    - exact seed
    - exact git commit or code hash
    - final metrics
    - per-path P&L array
    - turnover array
    - model checkpoint for learned strategies
    - aggregate manifest entry
    - aggregate summary table or frozen paper snapshot referencing the run
