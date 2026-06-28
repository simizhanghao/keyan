#!/usr/bin/env bash
# Phase A.5: smoke audit — must pass before GPU full runs.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-python3}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"

OUT="${ROOT}/experiments/em_robustness_openset/results/smoke_audit_$(date +%Y%m%d_%H%M)"
mkdir -p "${OUT}/logs"

MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
DEVICE=${DEVICE:-cuda}
SAMPLES=${SAMPLES:-32}
BATCH=${BATCH:-16}

echo "==> Smoke audit -> ${OUT}"
"${PY}" experiments/em_robustness_openset/audit_smoke_consistency.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CKPT}" \
  --samples-per-file "${SAMPLES}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --out-dir "${OUT}" \
  2>&1 | tee "${OUT}/logs/smoke_audit.log"

echo "==> Done. Read ${OUT}/SMOKE_AUDIT_REPORT.md"
