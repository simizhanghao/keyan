#!/usr/bin/env bash
# Full-mode RCPA grid: 2 directions × 3 seeds × 3 split repeats.
# Embeddings extracted once per (direction, seed); eval reuses npz with role reassignment.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}
MODEL=${MODEL:-ours_fused}
NUM_GPUS=${NUM_GPUS:-6}

DIRECTIONS=(rx1_to_rx2 rx2_to_rx1)
SEEDS=(0 1 2)
SPLIT_SEEDS=(0 1 2)
SHOT_KS="0 1 3 5 10 20"

OUT="${OUT:-${ROOT}/experiments/cross_receiver_calibration/results/full_$(date +%Y%m%d_%H%M)}"
RUNS="${OUT}/runs"
EMB="${OUT}/embeddings"
REPORT="${ROOT}/experiments/cross_receiver_calibration/CALIBRATION_REPORT.md"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${RUNS}" "${EMB}"

echo "==> Full RCPA grid"
echo "    OUT=${OUT}"
echo "    GPUs=${NUM_GPUS}"

extract_one() {
  local gpu=$1 direction=$2 seed=$3
  local ckpt="${PHASE5}/runs/F_cross_attn_chirp_plain/${direction}/seed_${seed}/best.pt"
  local split_csv="${EMB}/${direction}_seed${seed}_split0.csv"
  local emb_npz="${EMB}/${direction}_seed${seed}.npz"
  if [[ -f "${emb_npz}" ]]; then
    echo "[GPU${gpu}] skip extract ${direction} seed${seed} (exists)"
    return 0
  fi
  echo "[GPU${gpu}] extract ${direction} seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" experiments/cross_receiver_calibration/build_support_query_split.py \
    --direction "${direction}" --seed "${seed}" --split-seed 0 \
    --out-csv "${split_csv}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" experiments/cross_receiver_calibration/extract_calibration_embeddings.py \
    --split-csv "${split_csv}" --checkpoint "${ckpt}" --out-npz "${emb_npz}" \
    --device cuda --embedding-path fused
}

eval_one() {
  local direction=$1 seed=$2 split_seed=$3
  local run_dir="${RUNS}/${direction}_seed${seed}_split${split_seed}"
  local split_csv="${run_dir}/support_query_split.csv"
  local emb_npz="${EMB}/${direction}_seed${seed}.npz"
  local summary="${run_dir}/summary.csv"
  local alpha_csv="${run_dir}/alpha_sensitivity.csv"
  mkdir -p "${run_dir}"
  if [[ -f "${summary}" ]]; then
    echo "[eval] skip ${direction} seed${seed} split${split_seed} (exists)"
    return 0
  fi
  echo "[eval] ${direction} seed${seed} split${split_seed}"
  "${PY}" experiments/cross_receiver_calibration/build_support_query_split.py \
    --direction "${direction}" --seed "${seed}" --split-seed "${split_seed}" \
    --out-csv "${split_csv}"
  "${PY}" experiments/cross_receiver_calibration/run_rcpa_prototypes.py \
    --split-csv "${split_csv}" --embeddings-npz "${emb_npz}" \
    --direction "${direction}" --model "${MODEL}" --seed "${seed}" --split-seed "${split_seed}" \
    --shot-ks ${SHOT_KS} --out-csv "${summary}" \
    --alpha-sensitivity-csv "${alpha_csv}"
}

# --- Phase 1: parallel embedding extraction (6 jobs, one per direction×seed) ---
jobs=0
gpu=0
for direction in "${DIRECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    extract_one "${gpu}" "${direction}" "${seed}" &
    jobs=$((jobs + 1))
    gpu=$(( (gpu + 1) % NUM_GPUS ))
    if [[ "${jobs}" -ge "${NUM_GPUS}" ]]; then
      wait
      jobs=0
    fi
  done
done
wait
echo "==> Phase 1 extraction done"

# --- Phase 2: eval all split repeats (CPU, parallel) ---
jobs=0
for direction in "${DIRECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for split_seed in "${SPLIT_SEEDS[@]}"; do
      eval_one "${direction}" "${seed}" "${split_seed}" &
      jobs=$((jobs + 1))
      if [[ "${jobs}" -ge 18 ]]; then
        wait
        jobs=0
      fi
    done
  done
done
wait
echo "==> Phase 2 eval done"

# --- Phase 3: aggregate, plot, report ---
"${PY}" experiments/cross_receiver_calibration/aggregate_full_results.py \
  --runs-dir "${RUNS}" --out-dir "${OUT}"

"${PY}" experiments/cross_receiver_calibration/plot_full_shot_curves.py \
  --out-dir "${OUT}"

"${PY}" experiments/cross_receiver_calibration/generate_full_report.py \
  --out-dir "${OUT}" --out-md "${REPORT}"

echo "==> Full mode complete: ${OUT}"
echo "    Report: ${REPORT}"
