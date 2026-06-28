#!/usr/bin/env bash
# CNN-IQ EM robustness full (seed0 only).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PYTHON:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-7}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="experiments/em_robustness_openset/results/em_full_20260628_cnn"
mkdir -p "${OUT}/logs"
CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/A_cnn_iq/seed_0/best.pt"
MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
SAMPLES=256
BATCH=64

run_curve() {
  local ptype="$1"
  local strengths="$2"
  echo "--> CNN ${ptype} ${strengths}"
  "${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
    --manifest "${MANIFEST}" --checkpoint "${CKPT}" \
    --perturb-type "${ptype}" --strengths ${strengths} \
    --samples-per-file "${SAMPLES}" --batch-size "${BATCH}" \
    --num-workers 4 --device cuda \
    --out-csv "${OUT}/${ptype}_sweep.csv" \
    2>&1 | tee -a "${OUT}/logs/${ptype}.log"
}

if [ ! -f "${CKPT}" ]; then
  echo "CNN checkpoint missing: ${CKPT}"
  exit 1
fi

run_curve awgn_snr_db "100 40 30 25 20 15 10 5 0"
run_curve cfo_norm "0 0.001 0.003 0.005 0.01 0.02 0.03 0.05 0.10"
run_curve narrowband_sir_db "30 20 10 5 0"
run_curve phase_noise_std "0 0.01 0.03 0.05 0.10"
run_curve iq_amp_db "0 1 3 5"
run_curve filter_tilt_norm "0 0.1 0.2 0.4"

"${PY}" experiments/em_robustness_openset/_eval_mixed_quick.py \
  --checkpoint "${CKPT}" \
  --out-csv "${OUT}/mixed_stress_sweep.csv" \
  --device cuda \
  2>&1 | tee -a "${OUT}/logs/mixed.log"

"${PY}" experiments/em_robustness_openset/summarize_cnn_em.py --cnn-dir "${OUT}" --ours-dir experiments/em_robustness_openset/results/em_full_20260628

echo "CNN EM full -> ${OUT}"
