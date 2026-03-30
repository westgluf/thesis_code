#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source venv/bin/activate

BASELINE="$(ls -t results/archive/gbm_baseline_metrics_*.json | head -n 1)"
echo "Using baseline: $BASELINE"

rm -rf results/gbm_deephedge
mkdir -p results/gbm_deephedge

python -m src.train_deephedge_gbm >/dev/null

python - <<'PY' "$BASELINE" "results/gbm_deephedge/metrics_nn.json"
import json, sys

base = json.load(open(sys.argv[1]))
cur  = json.load(open(sys.argv[2]))

# Lower is better for these (risk/dispersion)
keys = ["std_PL", "ES_loss_0.95", "VaR_loss_0.95", "ES_loss_0.99", "VaR_loss_0.99"]

def worsened(a, b, tol=1e-10):
    return (b - a) > tol

bad = []
for k in keys:
    if k in base and k in cur and worsened(base[k], cur[k]):
        bad.append((k, base[k], cur[k]))

print("BASE:", {k: base.get(k) for k in keys})
print("CUR: ", {k: cur.get(k) for k in keys})

if bad:
    print("\nFAIL: metrics worsened:")
    for k, a, b in bad:
        print(f"  {k}: {a} -> {b}")
    raise SystemExit(2)

print("\nPASS: metrics not worse than baseline.")
PY
