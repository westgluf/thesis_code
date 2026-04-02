"""Ablation study: effect of the L2 regularization penalty on hedge positions.

Runs three variants using the smoke-sized config (configs/gbm_es95.yaml):
  BASELINE  reg_delta_l2 = 1e-4  (current default)
  ZERO      reg_delta_l2 = 0.0   (no penalty)
  SMALLER   reg_delta_l2 = 1e-5

Each variant does a full train+eval cycle and saves metrics to
results/ablation_reg/{variant}/.

Usage:
    python -m src.ablation_regularization
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.config import load_yaml
from src.train_deephedge_gbm import run_from_cfg

VARIANTS = [
    {"name": "BASELINE", "reg_delta_l2": 1e-4},
    {"name": "ZERO", "reg_delta_l2": 0.0},
    {"name": "SMALLER", "reg_delta_l2": 1e-5},
]

METRICS_KEYS = ("ES_loss_0.95", "ES_loss_0.99", "std_PL", "mean_PL", "mean_turnover")
BASE_CFG_PATH = "configs/gbm_es95.yaml"


def main() -> None:
    base_cfg = load_yaml(BASE_CFG_PATH)
    results: dict[str, dict[str, float]] = {}

    for variant in VARIANTS:
        name = variant["name"]
        reg_val = variant["reg_delta_l2"]
        print(f"\n{'='*60}")
        print(f"  ABLATION VARIANT: {name}  (reg_delta_l2 = {reg_val})")
        print(f"{'='*60}\n")

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("train", {})["reg_delta_l2"] = reg_val
        cfg["out_dir"] = f"results/ablation_reg/{name}"

        result = run_from_cfg(cfg)
        metrics = dict(result.metrics_nn)

        # Also read turnover from the saved metrics file
        metrics_path = Path(cfg["out_dir"]) / "metrics_nn.json"
        if metrics_path.exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(saved)

        results[name] = metrics

    _print_comparison_table(results)
    _print_pct_differences(results)
    _print_recommendation(results)


def _print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    print(f"\n{'='*80}")
    print("  ABLATION RESULTS: L2 regularization penalty on hedge positions")
    print(f"{'='*80}\n")

    header = f"{'Variant':<12}"
    for key in METRICS_KEYS:
        header += f" | {key:>14}"
    print(header)
    print("-" * len(header))

    for name in ("BASELINE", "ZERO", "SMALLER"):
        m = results[name]
        row = f"{name:<12}"
        for key in METRICS_KEYS:
            val = m.get(key, float("nan"))
            row += f" | {val:>14.8f}"
        print(row)
    print()


def _print_pct_differences(results: dict[str, dict[str, float]]) -> None:
    print("Percentage differences vs BASELINE:")
    print("-" * 60)
    baseline = results["BASELINE"]

    for compare_name in ("ZERO", "SMALLER"):
        compare = results[compare_name]
        row = f"  {compare_name:<10}"
        for key in METRICS_KEYS:
            bval = baseline.get(key, float("nan"))
            cval = compare.get(key, float("nan"))
            if bval != 0.0:
                pct = 100.0 * (cval - bval) / abs(bval)
            else:
                pct = float("nan")
            row += f"  {key}: {pct:+.2f}%"
        print(row)
    print()


def _print_recommendation(results: dict[str, dict[str, float]]) -> None:
    baseline = results["BASELINE"]
    zero = results["ZERO"]

    def _pct_diff(key: str) -> float:
        bval = baseline.get(key, float("nan"))
        zval = zero.get(key, float("nan"))
        if bval == 0.0:
            return float("nan")
        return 100.0 * abs(zval - bval) / abs(bval)

    es95_diff = _pct_diff("ES_loss_0.95")
    es99_diff = _pct_diff("ES_loss_0.99")

    print("=" * 60)
    print(f"  ES_0.95 difference (ZERO vs BASELINE): {es95_diff:.2f}%")
    print(f"  ES_0.99 difference (ZERO vs BASELINE): {es99_diff:.2f}%")
    print()

    if es95_diff < 2.0 and es99_diff < 2.0:
        print(
            "RECOMMENDATION: REMOVE the penalty (it has negligible effect). "
            "Set default reg_delta_l2=0.0"
        )
    elif es95_diff < 10.0 and es99_diff < 10.0:
        print(
            "RECOMMENDATION: PARAMETERIZE — keep the parameter but document it. "
            "Current default 1e-4 has moderate effect."
        )
    else:
        print(
            "RECOMMENDATION: KEEP the penalty at 1e-4, it materially improves tail risk. "
            "Document in dissertation."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
