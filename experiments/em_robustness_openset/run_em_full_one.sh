#!/usr/bin/env bash
# Run one perturbation family full eval (GPU_ID + PERTURB required).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-0}
PERTURB=${PERTURB:?set PERTURB=awgn|cfo|narrowband|phase_iq|filter_mixed}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
OUT_DIR=${OUT_DIR:-experiments/em_robustness_openset/results/em_full_$(date +%Y%m%d)}
mkdir -p "${OUT_DIR}/logs"

MANIFEST=${MANIFEST:-data/paper/cross_day_day1to5_source_only.csv}
CKPT=${CKPT:-outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt}
SAMPLES=${SAMPLES:-256}
BATCH=${BATCH:-64}

run_curve() {
  local ptype="$1"
  local strengths="$2"
  echo "--> ${ptype} ${strengths}"
  "${PY}" experiments/em_robustness_openset/eval_robustness_curves.py \
    --manifest "${MANIFEST}" --checkpoint "${CKPT}" \
    --perturb-type "${ptype}" --strengths ${strengths} \
    --samples-per-file "${SAMPLES}" --batch-size "${BATCH}" \
    --num-workers 4 --device cuda \
    --out-csv "${OUT_DIR}/${ptype}_sweep.csv" \
    2>&1 | tee -a "${OUT_DIR}/logs/${ptype}.log"
}

case "${PERTURB}" in
  awgn)
    run_curve awgn_snr_db "100 40 30 25 20 15 10 5 0"
    ;;
  cfo)
    run_curve cfo_norm "0 0.001 0.003 0.005 0.01 0.02 0.03 0.05 0.10"
    ;;
  narrowband)
    run_curve narrowband_sir_db "30 20 10 5 0"
    ;;
  phase_iq)
    run_curve phase_noise_std "0 0.01 0.03 0.05 0.10"
    run_curve iq_amp_db "0 1 3 5"
    ;;
  filter_mixed)
    run_curve filter_tilt_norm "0 0.1 0.2 0.4"
    ;;
  *)
    echo "Unknown PERTURB=${PERTURB}"
    exit 1
    ;;
esac

echo "Done ${PERTURB} -> ${OUT_DIR}"
