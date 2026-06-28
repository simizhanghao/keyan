#!/usr/bin/env bash
# Open-set smoke only (split + MSP/Energy/Proto/Maha on clean Day5).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-python3}
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT="${ROOT}/experiments/em_robustness_openset/results/openset_smoke"
mkdir -p "${OUT}"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
DEVICE=${DEVICE:-cpu}

"${PY}" experiments/em_robustness_openset/build_openset_splits.py --seeds 0
"${PY}" experiments/em_robustness_openset/eval_openset_auth.py \
  --openset-manifest experiments/em_robustness_openset/results/openset_splits/openset_split_seed0.csv \
  --checkpoint "${CKPT}" \
  --samples-per-file 64 \
  --batch-size 16 \
  --device "${DEVICE}" \
  --out-csv "${OUT}/openset_clean.csv"

echo "Done: ${OUT}"
