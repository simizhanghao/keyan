#!/usr/bin/env bash
# Open-set under EM (clean-trained Ours, 3 seeds, Proto/Mahalanobis).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PYTHON:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-5}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="experiments/em_robustness_openset/results/openset_under_em_20260628"
mkdir -p "${OUT}/logs"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"

"${PY}" experiments/em_robustness_openset/eval_openset_under_em.py \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT}" \
  --device cuda \
  --samples-per-file 256 \
  --batch-size 64 \
  --num-workers 4 \
  2>&1 | tee "${OUT}/logs/run.log"

echo "Done -> ${OUT}"
