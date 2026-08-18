#!/usr/bin/env bash
# Paper 1 Audit 1C.mech: OOB identity shuffle negative control.
# Primary model only: C_full_ratio with --oob-identity-shuffle.
# Same 1C recipe. Day4 val. Day5 is not evaluated. C zscore shuffle is not this step.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-${SEED}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SMOKE="${SMOKE:-0}"
NAME="C_full_ratio_oob_shuffle"
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

EPOCHS=80
BATCH=128
SAMPLES=256
EVAL_SAMPLES=256
MAX_FILES_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS=1
  BATCH=8
  SAMPLES=16
  EVAL_SAMPLES=16
  NUM_WORKERS=0
  MAX_FILES_ARGS=(--max-files 8)
  SEED_ARR=(0)
fi

TRAIN_ONLY=(
  --epochs "${EPOCHS}"
  --lr 3e-3
  --loss-type ce
  --checkpoint-metric acc
  --weight-decay 5e-4
  --label-smoothing 0
)

STEM=(
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --use-chirp-embedding
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --oob-norm ratio
  --oob-identity-shuffle
)

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_dir="${OUT_ROOT}/runs/${NAME}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_val/${NAME}/seed_${seed}"
  local log="${LOG_DIR}/${NAME}_seed${seed}.log"
  local common=(
    --manifest "${MANIFEST}"
    --root "${DATA_ROOT}"
    --batch-size "${BATCH}"
    --samples-per-file "${SAMPLES}"
    --eval-samples-per-file "${EVAL_SAMPLES}"
    --dim 64
    --depth 2
    --device cuda
    --train-split train
    --val-split val
    --eval-split val
    --input-norm iq_rms
    --fft-norm log_zscore
    --window-size 8192
    --num-workers "${NUM_WORKERS}"
    --seed "${seed}"
    "${MAX_FILES_ARGS[@]}"
  )
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ "${SMOKE}" != "1" && -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${NAME} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    run_dir="${OUT_ROOT}/runs/${NAME}/smoke_seed_${seed}"
    eval_dir="${OUT_ROOT}/eval_val/${NAME}/smoke_seed_${seed}"
    log="${LOG_DIR}/${NAME}_smoke_seed${seed}.log"
    mkdir -p "${run_dir}" "${eval_dir}"
  fi
  {
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} smoke=${SMOKE} ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      "${common[@]}" "${STEM[@]}" "${TRAIN_ONLY[@]}" --out-dir "${run_dir}"
    echo "=== EVAL VAL ${NAME} (Day4, not Day5) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      "${common[@]}" "${STEM[@]}" \
      --mode classifier --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME}  file_acc={100*m['file_acc']:.1f}%  window_acc={100*m['window_acc']:.1f}%  files={m['num_files']}  donor_mismatch={m.get('oob_donor_mismatch_rate')}")
if m.get("oob_identity_shuffle") is not True:
    raise SystemExit("checkpoint/eval did not enable oob_identity_shuffle")
if float(m.get("oob_donor_mismatch_rate") or 0) != 1.0:
    raise SystemExit("donor mismatch rate must be 1.0")
if "${SMOKE}" != "1" and m["num_files"] != 24:
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

if [[ "${SMOKE}" == "1" ]]; then
  "${PY}" "${KEYAN}/experiments/paper1_audit/scripts/audit_oob_identity_shuffle.py" --check-donors
fi

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
    echo "a shuffle job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  idx=$((idx + N_GPU))
done

if [[ "${SMOKE}" == "1" ]]; then
  echo "smoke finished. Day5 was not evaluated. Not a 5-seed verdict."
else
  echo "OOB identity shuffle seeds finished (${SEEDS}). Day5 was not evaluated."
  "${PY}" "${KEYAN}/experiments/paper1_audit/scripts/audit_oob_identity_shuffle.py"
fi
