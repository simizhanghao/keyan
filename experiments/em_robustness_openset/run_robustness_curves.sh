#!/usr/bin/env bash
# Evaluate EM perturbation robustness curves on the cross-day main model.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"

MANIFEST=${MANIFEST:-data/paper/cross_day_day1to5_source_only.csv}
CHECKPOINT=${CHECKPOINT:-outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt}
OUT_DIR=${OUT_DIR:-experiments/em_robustness_openset/results}
mkdir -p "${OUT_DIR}"

PERTURB_TYPES=(
  awgn_snr_db
  cfo_hz
  phase_noise_std
  narrowband_amplitude
  iq_imbalance
  filter_tilt_db
)

echo "==> EM robustness curves"
echo "    manifest=${MANIFEST}"
echo "    checkpoint=${CHECKPOINT}"
echo "    out_dir=${OUT_DIR}"

for ptype in "${PERTURB_TYPES[@]}"; do
  out_csv="${OUT_DIR}/${ptype}_sweep.csv"
  echo "--> ${ptype}"
  "${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
    --manifest "${MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --perturb-type "${ptype}" \
    --out-csv "${out_csv}" \
    --device cuda
done

echo "==> Done. Results in ${OUT_DIR}/"
