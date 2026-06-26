#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/tta_threshold_sweep_$(date +%Y%m%d)}"
SPLIT="${ROOT}/experiments/cross_receiver_calibration/results/quick_20260626_1709/support_query_split.csv"
CKPT="${ROOT}/outputs/paper_ready_v3/phase5_clean_cross_receiver/runs/F_cross_attn_chirp_plain/rx1_to_rx2/seed_0/best.pt"

cd "${ROOT}"
mkdir -p "${OUT}"

"${PY}" experiments/cross_receiver_calibration/run_tta_threshold_sweep.py \
  --split-csv "${SPLIT}" --checkpoint "${CKPT}" --out-dir "${OUT}" \
  --thresholds 0.5 0.7 0.8 0.9 0.95 --device cuda

"${PY}" experiments/cross_receiver_calibration/generate_tta_threshold_report.py --out-dir "${OUT}"
echo "Done: ${OUT}"
