#!/usr/bin/env bash
# EM-CR full training (run only after smoke passes).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PYTHON:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-5}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="experiments/em_robustness_openset/results/emcr_full_$(date +%Y%m%d)"
mkdir -p "${OUT}/logs"
INIT_CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"

"${PY}" experiments/em_robustness_openset/train_em_consistency.py \
  --manifest data/paper/cross_day_day1to5_source_only.csv \
  --init-checkpoint "${INIT_CKPT}" \
  --out-dir "${OUT}/checkpoints" \
  --mode em_cr \
  --lambda-kl 0.5 \
  --lambda-emb 0.0 \
  --epochs 20 \
  --samples-per-file 128 \
  --batch-size 32 \
  --lr 5e-4 \
  --device cuda \
  --num-workers 4 \
  2>&1 | tee "${OUT}/logs/train.log"

echo "EM-CR full -> ${OUT}"
