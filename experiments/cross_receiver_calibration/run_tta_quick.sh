#!/usr/bin/env bash
# TTA negative baseline quick: RX1->RX2, seed0, split0.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-0}
PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/tta_negative_quick_$(date +%Y%m%d_%H%M)}"
CKPT="${PHASE5}/runs/F_cross_attn_chirp_plain/rx1_to_rx2/seed_0/best.pt"
SPLIT="${ROOT}/experiments/cross_receiver_calibration/results/quick_20260626_1709/support_query_split.csv"
RCPA_SUM="${ROOT}/experiments/cross_receiver_calibration/results/full_20260626_1720/runs/rx1_to_rx2_seed0_split0/summary.csv"

cd "${ROOT}"
mkdir -p "${OUT}"

echo "==> TTA negative baseline quick"
"${PY}" experiments/cross_receiver_calibration/run_tta_negative_baseline.py \
  --split-csv "${SPLIT}" \
  --checkpoint "${CKPT}" \
  --direction rx1_to_rx2 --seed 0 --split-seed 0 \
  --out-dir "${OUT}" \
  --rcpa-summary "${RCPA_SUM}" \
  --device cuda

"${PY}" experiments/cross_receiver_calibration/generate_tta_report.py --out-dir "${OUT}"

echo "==> Paper 2 main table (from frozen full RCPA)"
PAPER2="${ROOT}/experiments/cross_receiver_calibration/results/paper2_main"
mkdir -p "${PAPER2}"
"${PY}" experiments/cross_receiver_calibration/generate_paper2_main_table.py \
  --summary-full "${ROOT}/experiments/cross_receiver_calibration/results/full_20260626_1720/summary_full.csv" \
  --out-dir "${PAPER2}"

echo "==> Done"
echo "    TTA report: ${OUT}/TTA_NEGATIVE_BASELINE_REPORT.md"
echo "    Paper2 table: ${PAPER2}/PAPER2_MAIN_TABLE.md"
