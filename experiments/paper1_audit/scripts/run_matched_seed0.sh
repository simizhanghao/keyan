#!/usr/bin/env bash
# Paper 1 Audit 1C seed0: matched CNN / Main-only / Full-zscore / Full-ratio.
# Checkpoint and reported metrics use Day4 val only. Day5 is not evaluated.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPU="${GPU:-1}"
SEED="${SEED:-0}"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"

if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

cd "${KEYAN}"
export PYTHONPATH="${KEYAN}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

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
  shift
  local run_dir="${OUT_ROOT}/runs/${name}/seed_${SEED}"
  local eval_dir="${OUT_ROOT}/eval_val/${name}/seed_${SEED}"
  mkdir -p "${run_dir}" "${eval_dir}"
  echo "=== TRAIN ${name} seed=${SEED} GPU=${GPU} ==="
  "${PY}" scripts/finetune.py "${COMMON[@]}" "$@" "${TRAIN_ONLY[@]}" --out-dir "${run_dir}"
  echo "=== EVAL VAL ${name} (Day4, not Day5) ==="
  "${PY}" scripts/evaluate.py "${COMMON[@]}" "$@" \
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
}

mkdir -p "${OUT_ROOT}"
echo "python=${PY}"
echo "day5_eval=FORBIDDEN"

run_one A_cnn_iq --model-type osu_cnn --cnn-input-type iq
run_one B_exact_main_no_oob "${STEM[@]}" --no-oob --oob-fusion-type no_oob --oob-norm none
run_one C_full_zscore "${STEM[@]}" --oob-fusion-type cross_attn_oob --use-oob-cross-attention --oob-norm zscore
run_one C_full_ratio "${STEM[@]}" --oob-fusion-type cross_attn_oob --use-oob-cross-attention --oob-norm ratio

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
out.write_text(json.dumps({"day5_used": False, "seed": 0, "rows": rows}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
print("wrote", out)
PY

echo "1C seed0 finished. Day5 was not evaluated."
