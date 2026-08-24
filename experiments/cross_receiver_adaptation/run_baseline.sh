#!/usr/bin/env bash
# Strict source-only cross-receiver baseline (thesis Chapter 4 starting point).
# Trains CNN-IQ and OOB-cross-attn Hybrid on RX1->RX2 and RX2->RX1.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
EPOCHS=${EPOCHS:-80}
SEEDS=${SEEDS:-0,1,2}
BATCH_SIZE=${BATCH_SIZE:-128}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT_BASE=${OUT_BASE:-experiments/cross_receiver_adaptation/results/baseline_$(date +%Y%m%d)}
RUN_BASE="${OUT_BASE}/runs"
EVAL_BASE="${OUT_BASE}/outputs"
mkdir -p "${RUN_BASE}" "${EVAL_BASE}" logs

IFS=',' read -ra SEED_ARR <<< "${SEEDS}"

COMMON=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr 3e-3
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64 --depth 2
  --device cuda
  --train-split train --val-split val --eval-split test
)
HYBRID=(
  --model-type rf_hstu
  --patch-embed-type cnn_stem --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention --use-chirp-embedding
  --input-norm iq_rms --fft-norm log_zscore --oob-norm ratio
  --weight-decay 5e-4
)

cd "${ROOT}"
LOG="logs/thesis_cross_rx_baseline_$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local direction="$1" model="$2" seed="$3"
  local manifest="data/paper/${direction}_source_only.csv"
  local tag="${direction}/${model}/seed_${seed}"
  local run_dir="${RUN_BASE}/${tag}"
  local out_dir="${EVAL_BASE}/${tag}"

  echo "==> train ${tag}"
  if [[ "${model}" == "cnn" ]]; then
    "${PY}" scripts/finetune.py --manifest "${manifest}" --out-dir "${run_dir}" \
      "${COMMON[@]}" --seed "${seed}" \
      --model-type osu_cnn --cnn-input-type iq --input-norm iq_rms
  else
    "${PY}" scripts/finetune.py --manifest "${manifest}" --out-dir "${run_dir}" \
      "${COMMON[@]}" --seed "${seed}" "${HYBRID[@]}"
  fi

  echo "==> eval ${tag}"
  "${PY}" scripts/evaluate.py --manifest "${manifest}" \
    --checkpoint "${run_dir}/best.pt" \
    --mode classifier --file-vote-mode mean_logits \
    --out-dir "${out_dir}" \
    --batch-size "${BATCH_SIZE}" --device cuda \
    --samples-per-file 256 --eval-samples-per-file 256 \
    --train-split train --val-split val --eval-split test
}

{
  echo "thesis cross-receiver baseline started $(date --iso-8601=seconds)"
  echo "epochs=${EPOCHS} seeds=${SEEDS} gpu=${GPU_ID}"

  for direction in rx1_to_rx2 rx2_to_rx1; do
    for seed in "${SEED_ARR[@]}"; do
      run_one "${direction}" cnn "${seed}"
      run_one "${direction}" hybrid "${seed}"
    done
  done

  echo "==> summarize"
  "${PY}" scripts/summarize_results.py --input-dir "${EVAL_BASE}" \
    --out "${OUT_BASE}/baseline_summary.csv" || true

  echo "thesis cross-receiver baseline finished $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"

echo "Results: ${OUT_BASE}/baseline_summary.csv"
