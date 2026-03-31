#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

./tools/clean.sh
./tools/compile.sh

python -m src.run_gbm_baseline
python -m src.train_deephedge_gbm

test -f results/gbm_deephedge/metrics_nn.json
test -f results/gbm_deephedge/metrics_bs.json
test -f results/gbm_deephedge/hist_pl_bs_vs_nn.png
test -f results/gbm_deephedge/tail_metrics_bs_vs_nn.png

./tools/guard.sh
echo "smoke OK"
