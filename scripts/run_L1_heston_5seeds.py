"""L.1 Heston-source 5-seed expansion.

Runs the existing ``run_L1_multi_source`` function from
``deep_hedging.experiments.transfer_extended`` with sources=["heston"] and
seeds=[7001..7005], writing to a new file
``results/transfer_v2/L1_heston_5seeds.json``.

The original ``L1_multi_source_5seeds.json`` is **not** touched; it remains as
the archival single-seed (gbm-source seed 7001) record.

Run from repo root::

    python scripts/run_L1_heston_5seeds.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Detach from parent process group so the harness cannot kill us via signals
# sent to the parent shell's group. macOS does not have a CLI ``setsid`` but
# this is equivalent.
try:
    os.setsid()
except (PermissionError, OSError):
    pass  # already a session leader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_hedging.experiments.transfer_extended import (
    run_L1_multi_source,
    OUT_DIR,
)

def main() -> None:
    out_path = OUT_DIR / "L1_heston_5seeds.json"
    print(f"Writing to: {out_path}", flush=True)

    result = run_L1_multi_source(
        seeds=[7001, 7002, 7003, 7004, 7005],
        sources=["heston"],
        out_path=out_path,
    )

    # Print a compact summary
    heston = result["results"]["heston"]
    print("\n=== L.1 Heston-source 5-seed summary ===", flush=True)
    for seed_str, m in heston["per_seed"].items():
        if "error" in m:
            print(f"  seed={seed_str}: ERROR {m['error']}", flush=True)
        else:
            print(
                f"  seed={seed_str}: ES_0.95 = {m['es_95']:.4f}  "
                f"first_weight_sum = {m.get('first_weight_sum', 'n/a')}",
                flush=True,
            )
    ag = heston.get("aggregate", {}).get("es_95", {})
    if ag:
        print(
            f"\n  Aggregate: ES_0.95 = {ag['mean']:.4f} ± {ag['std']:.4f}  "
            f"(95% CI [{ag['ci95_lower']:.4f}, {ag['ci95_upper']:.4f}])",
            flush=True,
        )

    print(f"\nDone. Output saved to: {out_path}", flush=True)

if __name__ == "__main__":
    main()
