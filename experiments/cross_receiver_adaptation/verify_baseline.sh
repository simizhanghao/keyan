#!/usr/bin/env bash
# Re-evaluate existing Phase5-clean checkpoints (no retraining).
# Use this to verify baseline numbers match IoTJ Table IV before new experiments.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}
OUT=${OUT:-experiments/cross_receiver_adaptation/results/verify_baseline_$(date +%Y%m%d)}
mkdir -p "${OUT}"

cd "${ROOT}"

declare -A MANIFESTS=(
  [rx1_to_rx2]=data/paper/rx1_to_rx2_source_only.csv
  [rx2_to_rx1]=data/paper/rx2_to_rx1_source_only.csv
)

for model in A_cnn_iq F_cross_attn_chirp_plain; do
  for direction in rx1_to_rx2 rx2_to_rx1; do
    for seed in 0 1 2; do
      ckpt="${PHASE5}/runs/${model}/${direction}/seed_${seed}/best.pt"
      if [[ ! -f "${ckpt}" ]]; then
        echo "SKIP missing ${ckpt}"
        continue
      fi
      out_dir="${OUT}/${model}/${direction}/seed_${seed}"
      echo "==> eval ${model} ${direction} seed=${seed}"
      "${PY}" scripts/evaluate.py \
        --manifest "${MANIFESTS[$direction]}" \
        --checkpoint "${ckpt}" \
        --mode classifier --file-vote-mode mean_logits \
        --out-dir "${out_dir}" \
        --batch-size 128 --device cuda \
        --samples-per-file 256 --eval-samples-per-file 256 \
        --train-split train --val-split val --eval-split test
    done
  done
done

"${PY}" scripts/summarize_results.py --root "${OUT}" > "${OUT}/verify_summary.txt" 2>&1 || true
echo "Verified results: ${OUT}/verify_summary.txt"
