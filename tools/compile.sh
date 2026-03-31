#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
python -m py_compile src/*.py tools/*.py
echo "compile OK"
