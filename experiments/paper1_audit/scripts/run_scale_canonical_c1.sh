#!/usr/bin/env bash
# Paper 2 Phase 2A-1: matched C1 (ratio_rms) on seed 0/1 only.
# Frozen C' is not retrained. Day5 unused. No real RX2.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEEDS="${SEEDS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NAME="C_full_ratio_rms"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"

if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

cd "${KEYAN}"
export PYTHONPATH="${KEYAN}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
if [[ ${#SEED_ARR[@]} -gt 2 ]]; then
  echo "2A-1 allows seeds 0,1 only" >&2
  exit 1
fi
for s in "${SEED_ARR[@]}"; do
  if [[ "${s}" != "0" && "${s}" != "1" ]]; then
    echo "refusing seed=${s}; only 0 and 1" >&2
    exit 1
  fi
done

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_dir="${OUT_ROOT}/runs/${NAME}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_val/${NAME}/seed_${seed}"
  local log="${LOG_DIR}/${NAME}_seed${seed}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${NAME} seed=${seed} (metrics exist)"
    return 0
  fi
  {
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} oob_norm=ratio_rms Day4 ckpt, not Day5 ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --root "${DATA_ROOT}" \
      --batch-size 128 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --train-split train \
      --val-split val \
      --eval-split val \
      --input-norm iq_rms \
      --fft-norm log_zscore \
      --window-size 8192 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --epochs 80 \
      --lr 3e-3 \
      --loss-type ce \
      --checkpoint-metric acc \
      --weight-decay 5e-4 \
      --label-smoothing 0 \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio_rms \
      --out-dir "${run_dir}"
    echo "=== EVAL VAL ${NAME} seed=${seed} (Day4, not Day5) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --root "${DATA_ROOT}" \
      --batch-size 128 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --train-split train \
      --val-split val \
      --eval-split val \
      --input-norm iq_rms \
      --fft-norm log_zscore \
      --window-size 8192 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio_rms \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
m = json.loads((Path("${eval_dir}") / "metrics.json").read_text())
print(f"${NAME} seed=${seed} file={100*m['file_acc']:.1f}% window={100*m['window_acc']:.1f}% files={m['num_files']}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("oob_norm") not in ("ratio_rms",):
    # evaluate may report ckpt oob_norm
    pass
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=${SEED_ARR[*]}"
echo "model=${NAME}"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "training=C1_only"

N_GPU=${#GPU_ARR[@]}
idx=0
while [[ ${idx} -lt ${#SEED_ARR[@]} ]]; do
  pids=()
  for ((j = 0; j < N_GPU && idx + j < ${#SEED_ARR[@]}; j++)); do
    echo "======== seed=${SEED_ARR[$((idx + j))]} gpu=${GPU_ARR[j]} ========"
    run_one "${GPU_ARR[j]}" "${SEED_ARR[$((idx + j))]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "C1 train/eval failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  idx=$((idx + N_GPU))
done

echo "C1 seed 0/1 finished. Day5 unused. Compare to frozen C_full_ratio only."
echo "OOB-scale / full-RX stress is a later eval, not this job."
