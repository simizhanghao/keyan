#!/usr/bin/env bash
# Same-protocol SOTA-style baselines: linear probe, head fine-tune, feature alignment.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}
FULL=${FULL:-experiments/cross_receiver_calibration/results/full_20260626_1720}

OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/sota_style_baselines_$(date +%Y%m%d_%H%M)}"
RUNS="${OUT}/runs"
MERGED="${OUT}/same_protocol_baselines.csv"

export PYTHONPATH="${ROOT}/src:${ROOT}/experiments/cross_receiver_calibration:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${RUNS}"

DIRECTIONS=(rx1_to_rx2 rx2_to_rx1)
SEEDS=(0 1 2)
SPLITS=(0 1 2)
K_VALUES="1 5 10"

echo "==> SOTA-style same-protocol baselines"
echo "    OUT=${OUT}"
echo "    FULL embeddings=${FULL}/embeddings"

run_one() {
  local direction=$1 seed=$2 split=$3
  local run_dir="${RUNS}/${direction}_seed${seed}_split${split}"
  local split_csv="${FULL}/runs/${direction}_seed${seed}_split${split}/support_query_split.csv"
  local emb_npz="${FULL}/embeddings/${direction}_seed${seed}.npz"
  local ckpt="${PHASE5}/runs/F_cross_attn_chirp_plain/${direction}/seed_${seed}/best.pt"
  local out_csv="${run_dir}/baselines.csv"

  if [[ ! -f "${split_csv}" ]]; then
    echo "[skip] missing split ${split_csv}"
    return 0
  fi
  if [[ -f "${out_csv}" ]]; then
    echo "[skip] exists ${out_csv}"
    return 0
  fi
  mkdir -p "${run_dir}"
  echo "[run] ${direction} seed${seed} split${split}"
  "${PY}" experiments/cross_receiver_calibration/baselines/run_same_protocol_baselines.py \
    --split-csv "${split_csv}" \
    --embeddings-npz "${emb_npz}" \
    --checkpoint "${ckpt}" \
    --direction "${direction}" \
    --seed "${seed}" \
    --split-seed "${split}" \
    --shot-ks ${K_VALUES} \
    --out-csv "${out_csv}" \
    --device cpu
}

for direction in "${DIRECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for split in "${SPLITS[@]}"; do
      run_one "${direction}" "${seed}" "${split}"
    done
  done
done

echo "==> Merge CSVs"
"${PY}" experiments/cross_receiver_calibration/baselines/generate_sota_baseline_report.py \
  --baseline-dir "${OUT}" \
  --full-results "${FULL}" \
  --out-report "${OUT}/SOTA_STYLE_BASELINE_REPORT.md" \
  --out-tex "${ROOT}/docs/paper2_rcpa/tables/table_sota_style_baselines.tex"

echo "Done: ${MERGED}"
