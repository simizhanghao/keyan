#!/usr/bin/env bash
# Paper 2 Phase 2A: C1 ratio_rms matched train, seeds 0 and 1 only.
# Copy 1C C' recipe. Unique change: --oob-norm ratio_rms.
# Day4 val checkpoint. Day5 unused. No RX2. No C2. No 5-seed.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEEDS="${SEEDS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NAME="C_full_ratio_rms"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"

if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

cd "${KEYAN}"
export PYTHONPATH="${KEYAN}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for seed in "${SEED_ARR[@]}"; do
  if [[ "${seed}" != "0" && "${seed}" != "1" ]]; then
    echo "this runner allows seeds 0 and 1 only, got ${seed}" >&2
    exit 1
  fi
done

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_dir="${OUT_ROOT}/runs/${NAME}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_val/${NAME}/seed_${seed}"
  local log="${LOG_DIR}/${NAME}_seed${seed}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${NAME} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  {
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} oob_norm=ratio_rms Day4-only ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --root "${DATA_ROOT}" \
      --batch-size 128 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --train-split train \
      --val-split val \
      --eval-split val \
      --input-norm iq_rms \
      --fft-norm log_zscore \
      --window-size 8192 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --epochs 80 \
      --lr 3e-3 \
      --loss-type ce \
      --checkpoint-metric acc \
      --weight-decay 5e-4 \
      --label-smoothing 0 \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio_rms \
      --out-dir "${run_dir}"
    echo "=== EVAL VAL ${NAME} seed=${seed} (Day4, not Day5) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --root "${DATA_ROOT}" \
      --batch-size 128 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --train-split train \
      --val-split val \
      --eval-split val \
      --input-norm iq_rms \
      --fft-norm log_zscore \
      --window-size 8192 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio_rms \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("oob_norm") not in {"ratio_rms", None} and False:
    pass
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=${SEED_ARR[*]}"
echo "model=${NAME}"
echo "oob_norm=ratio_rms"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "training=C1_only"

N_GPU=${#GPU_ARR[@]}
idx=0
while [[ ${idx} -lt ${#SEED_ARR[@]} ]]; do
  pids=()
  for ((j = 0; j < N_GPU && idx + j < ${#SEED_ARR[@]}; j++)); do
    seed="${SEED_ARR[$((idx + j))]}"
    echo "======== seed=${seed} gpu=${GPU_ARR[j]} ========"
    run_one "${GPU_ARR[j]}" "${seed}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "a C1 job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  idx=$((idx + N_GPU))
done

"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
print("# C1 vs frozen C' (Day4 window; gate Δ_clean ≥ -2pp)")
print("| seed | C' win | C1 win | Δ pp | C' file | C1 file |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
rows = []
for seed in (0, 1):
    c0 = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
    c1 = json.loads((root / "C_full_ratio_rms" / f"seed_{seed}" / "metrics.json").read_text())
    dw = 100 * c1["window_acc"] - 100 * c0["window_acc"]
    print(
        f"| {seed} | {100*c0['window_acc']:.1f} | {100*c1['window_acc']:.1f} | {dw:+.1f} | "
        f"{100*c0['file_acc']:.1f} | {100*c1['file_acc']:.1f} |"
    )
    rows.append({"seed": seed, "delta_window_pp": round(dw, 1), "clean_gate_pass": dw >= -2.0})
out = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/c1_clean_vs_cprime.json")
out.write_text(json.dumps({"day5_used": False, "rows": rows}, indent=2) + "\n")
print("wrote", out)
PY

echo "C1 seed 0/1 finished. Day5 unused. Scale/RX stress is a later eval."
