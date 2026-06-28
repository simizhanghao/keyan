#!/usr/bin/env bash
# Full closed-set EM robustness curves (all perturbation types).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

MANIFEST=${MANIFEST:-data/paper/cross_day_day1to5_source_only.csv}
CKPT=${CKPT:-outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt}
OUT_DIR=${OUT_DIR:-experiments/em_robustness_openset/results/full_$(date +%Y%m%d_%H%M)}
mkdir -p "${OUT_DIR}"

for p in awgn_snr_db cfo_norm narrowband_sir_db phase_noise_std iq_amp_db filter_tilt_norm; do
  "${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
    --manifest "${MANIFEST}" --checkpoint "${CKPT}" \
    --perturb-type "${p}" --out-csv "${OUT_DIR}/${p}_sweep.csv" --device cuda
done

echo "Full EM curves -> ${OUT_DIR}"
