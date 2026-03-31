#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
python tools/guard_train_gbm.py
echo "guard OK"
