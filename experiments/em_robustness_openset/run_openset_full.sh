#!/usr/bin/env bash
# Full open-set evaluation (3 split seeds, clean).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="${ROOT}/experiments/em_robustness_openset/results/openset_full_$(date +%Y%m%d_%H%M)"
mkdir -p "${OUT}"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"

"${PY}" experiments/em_robustness_openset/build_openset_splits.py --seeds 0 1 2

for seed in 0 1 2; do
  "${PY}" experiments/em_robustness_openset/eval_openset_auth.py \
    --openset-manifest "experiments/em_robustness_openset/results/openset_splits/openset_split_seed${seed}.csv" \
    --checkpoint "${CKPT}" --device cuda \
    --out-csv "${OUT}/openset_seed${seed}.csv"
done

echo "Open-set full -> ${OUT}"
