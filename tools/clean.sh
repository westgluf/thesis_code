#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf src/__pycache__ 2>/dev/null || true
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
echo "clean OK"
