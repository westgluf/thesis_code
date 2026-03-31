from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RESULTS_ROOT = Path("results")
ARCHIVE_DIR = RESULTS_ROOT / "archive"
DEFAULT_RUN_DIRNAME = "gbm_deephedge"
BASELINE_DIRNAME = "gbm_baseline"


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
