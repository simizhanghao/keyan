#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="$ROOT/results/final_training/short"; PY="/data1/hcc/LlamaFactory/.venv/bin/python"; DATA="$ROOT/../../../.x1_source_20260823"
while tmux has-session -t final_shen 2>/dev/null; do sleep 30; done
test "$(find "$ROOT/results/final_training/shen" -name '*.json' -type f | wc -l)" -eq 10
exec bash "$ROOT/scripts/launch_final_short_4gpu.sh"
