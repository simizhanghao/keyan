#!/usr/bin/env bash
# Paper 1 Audit 1C.mech: RX-style eval on frozen 1C C' checkpoints.
# No training. Day4 val only. Day5 is not evaluated.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-${SEED}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SMOKE="${SMOKE:-0}"
NAME="C_full_ratio_rx_style"
CKPT_NAME="C_full_ratio"
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
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "GPUS is empty" >&2
  exit 1
fi
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
if [[ "${SMOKE}" == "1" ]]; then
  SEED_ARR=(0)
  NUM_WORKERS=0
fi

run_one() {
  local gpu="$1"
  local seed="$2"
  local ckpt="${OUT_ROOT}/runs/${CKPT_NAME}/seed_${seed}/best.pt"
  local eval_dir="${OUT_ROOT}/eval_val/${NAME}/seed_${seed}"
  local log="${LOG_DIR}/${NAME}_seed${seed}.log"
  if [[ "${SMOKE}" == "1" ]]; then
    eval_dir="${OUT_ROOT}/eval_val/${NAME}/smoke_seed_${seed}"
    log="${LOG_DIR}/${NAME}_smoke_seed${seed}.log"
  fi
  mkdir -p "${eval_dir}"
  if [[ "${SMOKE}" != "1" && -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${NAME} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing frozen C' checkpoint: ${ckpt}" >&2
    return 1
  fi
  {
    echo "=== EVAL RX ${NAME} seed=${seed} GPU=${gpu} smoke=${SMOKE} (Day4, not Day5) ==="
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
      --oob-norm ratio \
      --rx-style-eval \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${ckpt}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME}  file_acc={100*m['file_acc']:.1f}%  window_acc={100*m['window_acc']:.1f}%  files={m['num_files']}  rx={m.get('rx_style_eval')}")
if m.get("rx_style_eval") is not True:
    raise SystemExit("rx_style_eval must be true")
if m.get("rx_inband_locked") is not True:
    raise SystemExit("rx_inband_locked must be true")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=${SEED_ARR[*]}"
echo "smoke=${SMOKE}"
echo "day5_eval=FORBIDDEN"
echo "training=FORBIDDEN"

N_GPU=${#GPU_ARR[@]}
idx=0
while [[ ${idx} -lt ${#SEED_ARR[@]} ]]; do
  pids=()
  for ((j = 0; j < N_GPU && idx + j < ${#SEED_ARR[@]}; j++)); do
    local_seed="${SEED_ARR[$((idx + j))]}"
    echo "======== seed=${local_seed} gpu=${GPU_ARR[j]} ========"
    run_one "${GPU_ARR[j]}" "${local_seed}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "an RX eval job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  idx=$((idx + N_GPU))
done

if [[ "${SMOKE}" == "1" ]]; then
  echo "RX smoke finished. Day5 was not evaluated. Not a 5-seed verdict."
else
  echo "RX-style eval seeds finished (${SEEDS}). Day5 was not evaluated."
  "${PY}" "${KEYAN}/experiments/paper1_audit/scripts/audit_rx_style.py"
fi
