#!/usr/bin/env bash
# OOB representation equalization quick mode (auxiliary experiment).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-0}
DIRECTION=${DIRECTION:-rx1_to_rx2}
SEED=${SEED:-0}
SPLIT_SEED=${SPLIT_SEED:-0}
PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/oob_eq_quick_$(date +%Y%m%d_%H%M)}"
CKPT="${PHASE5}/runs/F_cross_attn_chirp_plain/${DIRECTION}/seed_${SEED}/best.pt"
SPLIT_CSV="${OUT}/support_query_split.csv"
EMB_NPZ="${OUT}/embeddings_multipath.npz"

cd "${ROOT}"
mkdir -p "${OUT}"

echo "==> OOB representation equalization quick"
echo "    direction=${DIRECTION} seed=${SEED} split=${SPLIT_SEED} out=${OUT}"

echo "==> [1] Build split manifest"
"${PY}" experiments/cross_receiver_calibration/build_support_query_split.py \
  --direction "${DIRECTION}" --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
  --out-csv "${SPLIT_CSV}"

echo "==> [2] Extract multipath embeddings (main/oob/fused)"
"${PY}" experiments/cross_receiver_calibration/extract_multipath_embeddings.py \
  --split-csv "${SPLIT_CSV}" --checkpoint "${CKPT}" --out-npz "${EMB_NPZ}" --device cuda

echo "==> [3] Run OOB-Eq evaluation"
"${PY}" experiments/cross_receiver_calibration/run_oob_eq_quick.py \
  --split-csv "${SPLIT_CSV}" --embeddings-npz "${EMB_NPZ}" --checkpoint "${CKPT}" \
  --direction "${DIRECTION}" --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
  --shot-ks 0 1 3 5 --out-dir "${OUT}"

echo "==> [4] Plot"
"${PY}" experiments/cross_receiver_calibration/plot_oob_eq.py --out-dir "${OUT}"

echo "==> [5] Report"
"${PY}" experiments/cross_receiver_calibration/generate_oob_eq_report.py --out-dir "${OUT}"

echo "==> Done: ${OUT}"
