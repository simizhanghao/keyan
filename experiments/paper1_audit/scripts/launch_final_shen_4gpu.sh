#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DATA_ROOT="${DATA_ROOT:-$ROOT/../../../.x1_source_20260823}"; OUT_ROOT="${OUT_ROOT:-$ROOT/results/final_training/shen}"; PYTHON="${PYTHON:-/data1/hcc/LlamaFactory/.venv/bin/python}"
export LD_LIBRARY_PATH="/data1/hcc/LlamaFactory/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH="$ROOT/../../src:$ROOT/scripts:${PYTHONPATH:-}"; mkdir -p "$OUT_ROOT"
jobs=("Shen-CIS 0 74" "Shen-CIS 1 74" "Shen-CIS 2 74" "Shen-CIS 3 74" "Shen-CIS 4 74" "Shen-RA 0 71" "Shen-RA 1 71" "Shen-RA 2 71" "Shen-RA 3 71" "Shen-RA 4 71")
worker(){ local gpu="$1" i=0; for job in "${jobs[@]}"; do read -r arm seed epochs <<<"$job"; if ((i%4==gpu)); then out="$OUT_ROOT/${arm}/seed_${seed}.json"; if [[ ! -f "$out" ]]; then "$PYTHON" "$ROOT/scripts/run_final_shen.py" --data-root "$DATA_ROOT" --out "$out" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs "$epochs" --resume; fi; fi; i=$((i+1)); done; }
for gpu in 0 1 2 3; do worker "$gpu" >"$OUT_ROOT/worker_${gpu}.log" 2>&1 & done; wait; echo "Final Shen complete: $OUT_ROOT"
