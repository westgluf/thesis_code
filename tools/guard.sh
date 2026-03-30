#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/guard_train_gbm.py
echo "guard OK"
