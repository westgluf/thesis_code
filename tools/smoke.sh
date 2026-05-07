#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
python -m src.tools_cli smoke
