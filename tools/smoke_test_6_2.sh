#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

python -V
which python

# clean junk
rm -rf src/__pycache__ 2>/dev/null || true
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -type f -delete 2>/dev/null || true

# compile all
python -m py_compile src/*.py tools/*.py

# run baseline + training
python -m src.run_gbm_baseline
python -m src.train_deephedge_gbm

# check expected outputs exist
test -f results/gbm_deephedge/metrics_nn.json
test -f results/gbm_deephedge/metrics_bs.json
test -f results/gbm_deephedge/hist_pl_bs_vs_nn.png
test -f results/gbm_deephedge/tail_metrics_bs_vs_nn.png

# run guard (requires baseline saved in results/archive)
python tools/guard_train_gbm.py
