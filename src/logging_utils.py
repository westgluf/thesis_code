from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


TRAIN_LOG_HEADER = ("epoch", "train_loss", "val_loss", "lr", "w")


def write_json_file(path: str | Path, payload: Any, *, sort_keys: bool = False) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=sort_keys)
        f.write("\n")


def write_run_config(path: str | Path, cfg: Mapping[str, Any]) -> None:
    write_json_file(path, dict(cfg), sort_keys=True)


def write_train_log(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(TRAIN_LOG_HEADER)
        for row in rows:
            writer.writerow(_format_train_log_row(row))


def _format_train_log_row(row: Mapping[str, Any]) -> list[str]:
    missing = [key for key in TRAIN_LOG_HEADER if key not in row]
    extra = sorted(set(row) - set(TRAIN_LOG_HEADER))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        raise ValueError(f"train log row invalid ({'; '.join(parts)})")

    epoch = _normalize_epoch(row["epoch"])
    return [
        str(epoch),
        _format_float(row["train_loss"]),
        _format_float(row["val_loss"]),
        _format_float(row["lr"]),
        _format_float(row["w"]),
    ]


def _normalize_epoch(epoch_value: Any) -> int:
    epoch_float = float(epoch_value)
    if not epoch_float.is_integer():
        raise ValueError(f"epoch must be an integer, got {epoch_value!r}")
    return int(epoch_float)


def _format_float(value: Any) -> str:
    return format(float(value), ".17g")
