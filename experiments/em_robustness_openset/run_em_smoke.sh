#!/usr/bin/env bash
# Phase A smoke: AWGN + CFO curves, open-set split seed0, CPU-friendly settings.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-python3}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
OUT="${ROOT}/experiments/em_robustness_openset/results/smoke_$(date +%Y%m%d_%H%M)"
mkdir -p "${OUT}"

MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
DEVICE=${DEVICE:-cpu}
SAMPLES=${SAMPLES:-64}
BATCH=${BATCH:-16}

echo "==> Build EM benchmark config"
"${PY}" experiments/em_robustness_openset/build_em_perturbation_benchmark.py

echo "==> AWGN smoke curve"
"${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CKPT}" \
  --perturb-type awgn_snr_db \
  --strengths 30 20 10 \
  --samples-per-file "${SAMPLES}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --out-csv "${OUT}/awgn_smoke.csv"

echo "==> CFO smoke curve"
"${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CKPT}" \
  --perturb-type cfo_norm \
  --strengths 0.0 0.03 0.10 \
  --samples-per-file "${SAMPLES}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --out-csv "${OUT}/cfo_smoke.csv"

echo "==> Open-set splits"
"${PY}" experiments/em_robustness_openset/build_openset_splits.py --seeds 0

OPEN_MANIFEST="experiments/em_robustness_openset/results/openset_splits/openset_split_seed0.csv"

echo "==> Open-set auth smoke"
"${PY}" experiments/em_robustness_openset/eval_openset_auth.py \
  --openset-manifest "${OPEN_MANIFEST}" \
  --checkpoint "${CKPT}" \
  --samples-per-file "${SAMPLES}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --out-csv "${OUT}/openset_clean_smoke.csv"

echo "==> Smoke complete: ${OUT}"
