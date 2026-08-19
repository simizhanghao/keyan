#!/usr/bin/env bash
# 2A-5 Beat 2: C_fft-only matched train, seeds 0 and 1.
# Copy 1C C' recipe. Unique change: --fft-source inband.
# OOB stays C' ratio. Day4 val. Day5 unused. No RX2. No D. No 5-seed.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEEDS="${SEEDS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NAME="C_full_ratio_inband"
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
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} fft_source=inband oob_norm=ratio Day4-only ==="
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
      --fft-source inband \
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
      --oob-norm ratio \
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
      --fft-source inband \
      --window-size 8192 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} fft_source={m.get('fft_source')} oob={m.get('oob_norm')}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("eval_split") not in {"val", None}:
    raise SystemExit("eval_split must be val")
if m.get("day5_used") is True:
    raise SystemExit("day5_used must be false")
if m.get("fft_source") != "inband":
    raise SystemExit(f"fft_source must be inband, got {m.get('fft_source')}")
if m.get("oob_norm") != "ratio":
    raise SystemExit(f"oob_norm must be ratio, got {m.get('oob_norm')}")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=${SEED_ARR[*]}"
echo "model=${NAME}"
echo "fft_source=inband"
echo "oob_norm=ratio"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "cell_d=FORBIDDEN"
echo "training=C_fft_only"

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
    echo "a C_fft job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  idx=$((idx + N_GPU))
done

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
print("# C_fft-only vs frozen C' (Day4 window; gate Δ ≥ -2 pp; no collapse)")
print("| seed | C' win | C_fft win | Δ pp | gate | C' file | C_fft file |")
print("| ---: | ---: | ---: | ---: | --- | ---: | ---: |")
rows = []
all_pass = True
for seed in (0, 1):
    cprime = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
    cfft = json.loads((root / "C_full_ratio_inband" / f"seed_{seed}" / "metrics.json").read_text())
    dw = 100 * cfft["window_acc"] - 100 * cprime["window_acc"]
    collapsed = cfft["window_acc"] < 0.15
    gate = (dw >= -2.0) and (not collapsed)
    all_pass = all_pass and gate
    print(
        f"| {seed} | {100*cprime['window_acc']:.1f} | {100*cfft['window_acc']:.1f} | {dw:+.1f} | "
        f"{'PASS' if gate else 'FAIL'} | {100*cprime['file_acc']:.1f} | {100*cfft['file_acc']:.1f} |"
    )
    rows.append(
        {
            "seed": seed,
            "cprime_window": round(100 * cprime["window_acc"], 1),
            "cfft_window": round(100 * cfft["window_acc"], 1),
            "delta_window_pp": round(dw, 1),
            "collapsed": collapsed,
            "clean_gate_pass": gate,
        }
    )
verdict = "CLEAN_PASS" if all_pass else "CLEAN_FAIL"
out = {
    "day5_used": False,
    "fft_source": "inband",
    "oob_norm": "ratio",
    "cell": "C",
    "verdict": verdict,
    "rows": rows,
}
path = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/cfft_clean_vs_cprime.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print("verdict", verdict)
print("wrote", path)
if verdict != "CLEAN_PASS":
    print("clean gate FAIL; do not retune; do not open stress/D")
PY

echo "C_fft-only seed 0/1 finished. Day5 unused. Stress is a later eval."
