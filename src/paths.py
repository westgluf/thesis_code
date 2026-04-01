from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RESULTS_ROOT = Path("results")
ARCHIVE_DIR = RESULTS_ROOT / "archive"
DEFAULT_RUN_DIRNAME = "gbm_deephedge"
BASELINE_DIRNAME = "gbm_baseline"
BENCHMARK_DIRNAME = "benchmark_6_2"
BENCHMARK_RUNS_DIRNAME = "runs"
BENCHMARK_AGGREGATE_DIRNAME = "aggregate"


def get_run_dir(cfg: Mapping[str, Any]) -> Path:
    configured = cfg.get("out_dir") if isinstance(cfg, Mapping) else None
    return Path(configured) if configured else RESULTS_ROOT / DEFAULT_RUN_DIRNAME


def get_baseline_dir() -> Path:
    return RESULTS_ROOT / BASELINE_DIRNAME


def get_archive_dir() -> Path:
    return ARCHIVE_DIR


def feature_norm_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "feature_norm.json"


def metrics_bs_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "metrics_bs.json"


def metrics_nn_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "metrics_nn.json"


def hist_plot_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "hist_pl_bs_vs_nn.png"


def tail_plot_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "tail_metrics_bs_vs_nn.png"


def arrays_debug_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "arrays_debug.npz"


def best_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "best_state.pt"


def last_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "last_state.pt"


def train_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "train_log.csv"


def run_cfg_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run_cfg.json"


def benchmark_root_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / BENCHMARK_DIRNAME


def benchmark_runs_dir(run_dir: str | Path) -> Path:
    return benchmark_root_dir(run_dir) / BENCHMARK_RUNS_DIRNAME


def benchmark_aggregate_dir(run_dir: str | Path) -> Path:
    return benchmark_root_dir(run_dir) / BENCHMARK_AGGREGATE_DIRNAME


def benchmark_run_dir(run_dir: str | Path, run_id: str) -> Path:
    return benchmark_runs_dir(run_dir) / run_id


def benchmark_spec_json_path(run_dir: str | Path) -> Path:
    return benchmark_root_dir(run_dir) / "benchmark_spec.json"


def seed_info_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "seed_info.json"


def run_meta_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run_meta.json"


def metrics_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "metrics_summary.json"


def pl_bs_array_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "pl_bs.npy"


def pl_nn_array_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "pl_nn.npy"


def turnover_bs_array_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "turnover_bs.npy"


def turnover_nn_array_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "turnover_nn.npy"


def train_curve_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "curve_train_val_loss.png"


def manifest_runs_json_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "manifest_runs.json"


def manifest_runs_csv_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "manifest_runs.csv"


def summary_rows_json_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "summary_rows.json"


def summary_rows_csv_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "summary_rows.csv"


def seed_level_metrics_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "seed_level_metrics.csv"


def aggregated_by_method_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "aggregated_by_method.csv"


def paired_comparisons_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "paired_comparisons.csv"


def win_summary_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "win_summary.csv"


def scenario_summary_path(run_dir: str | Path) -> Path:
    return benchmark_aggregate_dir(run_dir) / "scenario_summary.csv"


def baseline_metrics_mcprice_path(run_dir: str | Path | None = None) -> Path:
    base_dir = Path(run_dir) if run_dir is not None else get_baseline_dir()
    return base_dir / "metrics_bs_mcprice.json"


def baseline_metrics_bsprice_path(run_dir: str | Path | None = None) -> Path:
    base_dir = Path(run_dir) if run_dir is not None else get_baseline_dir()
    return base_dir / "metrics_bs_bsprice.json"


def baseline_hist_plot_path(run_dir: str | Path | None = None) -> Path:
    base_dir = Path(run_dir) if run_dir is not None else get_baseline_dir()
    return base_dir / "hist_pl_bs_mc_vs_bs.png"


def baseline_tail_plot_path(run_dir: str | Path | None = None) -> Path:
    base_dir = Path(run_dir) if run_dir is not None else get_baseline_dir()
    return base_dir / "tail_metrics_bs_mc_vs_bs.png"


def new_baseline_archive_metrics_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_archive_dir() / f"gbm_baseline_metrics_{timestamp}.json"


def latest_baseline_archive_metrics_path() -> Path | None:
    candidates = sorted(get_archive_dir().glob("gbm_baseline_metrics_*.json"))
    return candidates[-1] if candidates else None
