#!/usr/bin/env bash
# RCPA calibration grid: quick or full mode.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
MODE=${1:-}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}
DIRECTION=${DIRECTION:-rx1_to_rx2}
SEED=${SEED:-0}
MODEL=${MODEL:-ours_fused}

if [[ "${MODE}" != "--quick" && "${MODE}" != "--full" ]]; then
  echo "Usage: GPU_ID=1 bash $0 --quick|--full"
  exit 1
fi

if [[ "${MODE}" == "--quick" ]]; then
  SHOT_KS="0 1 5 10"
  OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/quick_$(date +%Y%m%d_%H%M)}"
else
  echo "Full mode: use run_full_grid.sh instead"
  exec bash "${ROOT}/experiments/cross_receiver_calibration/run_full_grid.sh"
fi

CKPT="${PHASE5}/runs/F_cross_attn_chirp_plain/${DIRECTION}/seed_${SEED}/best.pt"
SPLIT_CSV="${OUT}/support_query_split.csv"
EMB_NPZ="${OUT}/embeddings_fused.npz"
SUMMARY="${OUT}/summary_quick.csv"
SHOT_CSV="${OUT}/shot_curve_rx1_to_rx2_quick.csv"
FIG_PDF="${OUT}/fig_shot_curve_quick.pdf"
REPORT="${ROOT}/experiments/cross_receiver_calibration/CALIBRATION_REPORT.md"

cd "${ROOT}"
mkdir -p "${OUT}"

echo "==> RCPA calibration ${MODE}"
echo "    direction=${DIRECTION} seed=${SEED} out=${OUT}"

echo "==> [1] Build block-disjoint split manifest"
"${PY}" experiments/cross_receiver_calibration/build_support_query_split.py \
  --direction "${DIRECTION}" \
  --seed "${SEED}" \
  --out-csv "${SPLIT_CSV}"

echo "==> [2] Extract fused embeddings"
"${PY}" experiments/cross_receiver_calibration/extract_calibration_embeddings.py \
  --split-csv "${SPLIT_CSV}" \
  --checkpoint "${CKPT}" \
  --out-npz "${EMB_NPZ}" \
  --device cuda \
  --embedding-path fused

echo "==> [3] Run RCPA evaluation"
"${PY}" experiments/cross_receiver_calibration/run_rcpa_prototypes.py \
  --split-csv "${SPLIT_CSV}" \
  --embeddings-npz "${EMB_NPZ}" \
  --direction "${DIRECTION}" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --shot-ks ${SHOT_KS} \
  --out-csv "${SUMMARY}"

cp "${SUMMARY}" "${SHOT_CSV}"

echo "==> [4] Plot shot curve"
"${PY}" experiments/cross_receiver_calibration/plot_shot_curves.py \
  --summary-csv "${SUMMARY}" \
  --out-pdf "${FIG_PDF}" \
  --out-png "${FIG_PDF%.pdf}.png"

echo "==> [5] Generate report"
"${PY}" experiments/cross_receiver_calibration/generate_calibration_report.py \
  --summary-csv "${SUMMARY}" \
  --split-csv "${SPLIT_CSV}" \
  --out-md "${REPORT}" \
  --direction "${DIRECTION}" \
  --seed "${SEED}"

echo "==> Done: ${OUT}"
echo "    Report: ${REPORT}"
