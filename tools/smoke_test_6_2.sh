#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./tools/smoke.sh
