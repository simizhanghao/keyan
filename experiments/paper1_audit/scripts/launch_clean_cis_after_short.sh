#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="/data1/hcc/LlamaFactory/.venv/bin/python"; DATA="$ROOT/../../../.x1_source_20260823"; OUT="$ROOT/results/final_training/shen_clean/Shen-CIS"
while tmux has-session -t final_short 2>/dev/null; do sleep 30; done
test "$(find "$ROOT/results/final_training/short" -name '*.json' -type f | wc -l)" -eq 15
export LD_LIBRARY_PATH="/data1/hcc/LlamaFactory/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH="$ROOT/../../src:$ROOT/scripts:${PYTHONPATH:-}"; mkdir -p "$OUT"
for seed in 0 1 2 3; do "$PY" "$ROOT/scripts/run_final_shen.py" --data-root "$DATA" --out "$OUT/seed_${seed}.json" --arm Shen-CIS --seed "$seed" --gpu "$seed" --epochs 74 >"$OUT/seed_${seed}.log" 2>&1 & done
wait
