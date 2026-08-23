#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/experiments/paper1_audit/results/external_rx_audit/x2_formal_runs"
LOG="$OUT/logs"
mkdir -p "$LOG"
folds=(rtl_2 rtl_5 b200_1 b200_mini_1 b210_1 pluto_1)
jobs=(); for fold in "${folds[@]}"; do for seed in 0 1; do for model in B1 Cprime; do jobs+=("$fold $model $seed"); done; done; done
for gpu in 0 1 2 3; do
  (
    for ((j=gpu; j<${#jobs[@]}; j+=4)); do
        read -r fold model seed <<<"${jobs[$j]}"
        NV=/data1/hcc/LlamaFactory/.venv/lib/python3.12/site-packages/nvidia
        PYTHONPATH="$ROOT/src" LD_LIBRARY_PATH="$NV/nvjitlink/lib:$NV/cusparse/lib:${LD_LIBRARY_PATH:-}" \
          /data1/hcc/LlamaFactory/.venv/bin/python "$ROOT/experiments/paper1_audit/scripts/run_x2_formal.py" \
          --source-root /data1/hcc/llm4RF/.x1_source_20260823 --out-root "$OUT" \
          --fold "$fold" --model "$model" --seed "$seed" --gpu "$gpu" --epochs 5 \
          >"$LOG/gpu${gpu}_${fold}_${model}_s${seed}.log" 2>&1
    done
  ) &
done
wait
