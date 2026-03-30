#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./tools/clean.sh
./tools/compile.sh

python -m src.run_gbm_baseline
python -m src.train_deephedge_gbm

test -f results/gbm_deephedge/metrics_nn.json
test -f results/gbm_deephedge/metrics_bs.json
test -f results/gbm_deephedge/train_log.csv
test -f results/gbm_deephedge/best_state.pt
test -f results/gbm_deephedge/last_state.pt
test -f results/gbm_deephedge/feature_norm.json

./tools/guard.sh
echo "smoke OK"
