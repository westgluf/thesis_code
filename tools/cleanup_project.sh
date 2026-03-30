#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/6] Remove macOS junk + caches"
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -name "*.pyo" -type f -delete 2>/dev/null || true
find . -name "*.pyd" -type f -delete 2>/dev/null || true
find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name ".pytest_cache" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "[2/6] Move backup/temporary files out of src/"
mkdir -p src/_archive
# python patch backups
find src -maxdepth 1 -type f \( -name "*.bak*" -o -name "*~" -o -name "*.tmp" \) -print -exec mv -f {} src/_archive/ \; 2>/dev/null || true

echo "[3/6] Make sure tools dir exists"
mkdir -p tools

echo "[4/6] (Optional) keep results, just remove macOS junk inside results"
find results -name ".DS_Store" -type f -delete 2>/dev/null || true

echo "[5/6] Quick compile check"
source venv/bin/activate
python -m py_compile src/*.py tools/*.py

echo "[6/6] Done."
echo "Moved backups to: src/_archive/"
