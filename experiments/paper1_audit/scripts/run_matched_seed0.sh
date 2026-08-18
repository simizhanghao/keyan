#!/usr/bin/env bash
# Paper 1 Audit 1C seed0: matched CNN / Main-only / Full-zscore / Full-ratio.
# Keep Paper 1 batch=128, lr=3e-3. Speed comes from 4-GPU parallel + DataLoader workers.
# Checkpoint and reported metrics use Day4 val only. Day5 is not evaluated.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "GPUS is empty" >&2
  exit 1
fi

COMMON=(
  --manifest "${MANIFEST}"
  --root "${DATA_ROOT}"
  --batch-size 128
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64
  --depth 2
  --device cuda
  --train-split train
  --val-split val
  --eval-split val
  --input-norm iq_rms
  --fft-norm log_zscore
  --window-size 8192
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
)

TRAIN_ONLY=(
  --epochs 80
  --lr 3e-3
  --loss-type ce
  --checkpoint-metric acc
  --weight-decay 5e-4
  --label-smoothing 0
)

STEM=(
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --use-chirp-embedding
)

run_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  local run_dir="${OUT_ROOT}/runs/${name}/seed_${SEED}"
  local eval_dir="${OUT_ROOT}/eval_val/${name}/seed_${SEED}"
  local log="${LOG_DIR}/${name}_seed${SEED}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  {
    echo "=== TRAIN ${name} seed=${SEED} GPU=${gpu} batch=128 workers=${NUM_WORKERS} ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      "${COMMON[@]}" "$@" "${TRAIN_ONLY[@]}" --out-dir "${run_dir}"
    echo "=== EVAL VAL ${name} (Day4, not Day5) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      "${COMMON[@]}" "$@" \
      --mode classifier --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${name}  file_acc={100*m['file_acc']:.1f}%  window_acc={100*m['window_acc']:.1f}%  file_f1={100*m['file_macro_f1']:.1f}%  files={m['num_files']}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
PY
  } >"${log}" 2>&1
}

start_job() {
  local idx="$1"
  local gpu="$2"
  case "${idx}" in
    0) run_one A_cnn_iq "${gpu}" --model-type osu_cnn --cnn-input-type iq ;;
    1) run_one B_exact_main_no_oob "${gpu}" "${STEM[@]}" --no-oob --oob-fusion-type no_oob --oob-norm none ;;
    2) run_one C_full_zscore "${gpu}" "${STEM[@]}" --oob-fusion-type cross_attn_oob --use-oob-cross-attention --oob-norm zscore ;;
    3) run_one C_full_ratio "${gpu}" "${STEM[@]}" --oob-fusion-type cross_attn_oob --use-oob-cross-attention --oob-norm ratio ;;
    *) echo "bad job index ${idx}" >&2; return 1 ;;
  esac
}

echo "python=${PY}"
echo "gpus=${GPUS}  (4 jobs, ${#GPU_ARR[@]} concurrent)"
echo "batch=128 (Paper 1 matched; not filling 80GB)"
echo "day5_eval=FORBIDDEN"

N_GPU=${#GPU_ARR[@]}
for ((i = 0; i < 4; i += N_GPU)); do
  pids=()
  echo "=== wave $((i / N_GPU + 1)): jobs ${i}..$((i + N_GPU < 4 ? i + N_GPU - 1 : 3)) ==="
  for ((j = 0; j < N_GPU && i + j < 4; j++)); do
    start_job "$((i + j))" "${GPU_ARR[j]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
rows = []
for name in ["A_cnn_iq", "B_exact_main_no_oob", "C_full_zscore", "C_full_ratio"]:
    m = json.loads((root / name / "seed_0" / "metrics.json").read_text())
    rows.append({
        "model": name,
        "split": "val_day4",
        "file_acc_pct": round(100 * m["file_acc"], 1),
        "window_acc_pct": round(100 * m["window_acc"], 1),
        "file_macro_f1_pct": round(100 * m["file_macro_f1"], 1),
        "n_files": m["num_files"],
    })
out = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/summary_val.json")
out.write_text(json.dumps({"day5_used": False, "seed": 0, "batch_size": 128, "rows": rows}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
print("wrote", out)
PY

echo "1C seed0 finished. Day5 was not evaluated."
