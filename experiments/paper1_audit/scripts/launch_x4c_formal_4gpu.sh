#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../../../.x1_source_20260823}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/external_rx_audit/x4c_formal}"
PYTHON="${PYTHON:-/data1/hcc/LlamaFactory/.venv/bin/python}"
export LD_LIBRARY_PATH="/data1/hcc/LlamaFactory/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/../../src:${PYTHONPATH:-}"

folds=(rtl_2 rtl_5 b200_1 b200_mini_1 b210_1 pluto_1)
arms=(Shen-CIS Shen-RA)
mkdir -p "$OUT_ROOT"

worker() {
  local gpu="$1" worker_id="$2" i=0
  for arm in "${arms[@]}"; do
    for fold in "${folds[@]}"; do
      for seed in 0 1; do
        if (( i % 4 == worker_id )); then
          out="$OUT_ROOT/$arm/${fold}_s${seed}.json"
          if [ ! -f "$out" ]; then
            "$PYTHON" "$ROOT/scripts/run_x4c_shen_port.py" \
              --data-root "$DATA_ROOT" --out "$out" --fold "$fold" \
              --seed "$seed" --arm "$arm" --gpu "$gpu" \
              --epochs 500 --patience 20
          fi
        fi
        i=$((i+1))
      done
    done
  done
}

for gpu in 0 1 2 3; do
  worker "$gpu" "$gpu" >"$OUT_ROOT/worker_${gpu}.log" 2>&1 &
done
wait
echo "X4-C formal complete: $OUT_ROOT"
