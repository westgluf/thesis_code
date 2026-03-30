#!/usr/bin/env bash
set -euo pipefail

FILE="${1:?usage: tools/apply_patch.sh path/to/file.py}"
PATCHER="${2:?usage: tools/apply_patch.sh path/to/file.py path/to/patcher.py}"

cd "$(dirname "$0")/.."

TS="$(date +"%Y%m%d_%H%M%S")"
BASENAME="$(basename "$FILE")"
BAK="archive/${BASENAME}.bak_${TS}"

cp "$FILE" "$BAK"
echo "Backup: $BAK"

python "$PATCHER" "$FILE"

python -m py_compile src/*.py tools/*.py

python tools/guard_train_gbm.py

echo "OK: patch applied + compile OK + guard OK"
