#!/usr/bin/env bash
# S0 diagnostic on seeds 2/3/4 after S1 5-seed CLEAN_FAIL.
# Frozen recipe. No S1 retrain. No stress. No RX2. No Day5.
# Does not overwrite s0_s1_clean_vs_cprime.json.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"
NAME="C_full_ratio_paired_clean"
NEW_SEEDS=(2 3 4)

if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

cd "${KEYAN}"
export PYTHONPATH="${KEYAN}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "need two GPUs, got GPUS=${GPUS}" >&2
  exit 1
fi

for seed in 0 1; do
  if [[ ! -f "${OUT_ROOT}/eval_val/${NAME}/seed_${seed}/metrics.json" ]]; then
    echo "missing frozen S0 seed ${seed}" >&2
    exit 1
  fi
done
for seed in 2 3 4; do
  if [[ ! -f "${OUT_ROOT}/eval_val/C_full_ratio/seed_${seed}/metrics.json" ]]; then
    echo "missing frozen C' seed ${seed}" >&2
    exit 1
  fi
  if [[ ! -f "${OUT_ROOT}/eval_val/C_full_ratio_paired_scale/seed_${seed}/metrics.json" ]]; then
    echo "missing frozen S1 seed ${seed}" >&2
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
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} paired_view=clean fft_source=full oob_norm=ratio Day4-only ==="
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
      --fft-source full \
      --paired-view clean \
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
    echo "=== EVAL VAL ${NAME} seed=${seed} (clean Day4, not Day5) ==="
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
      --fft-source full \
      --paired-view off \
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
print(f"${NAME} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} paired={m.get('paired_view')}")
if m["num_files"] != 24 or m.get("day5_used") is True:
    raise SystemExit("split/files check failed")
if m.get("fft_source") != "full" or m.get("oob_norm") != "ratio":
    raise SystemExit("norm check failed")
if m.get("paired_view") != "clean":
    raise SystemExit(f"paired_view must be clean from ckpt, got {m.get('paired_view')}")
if m.get("rx_style_eval") is True:
    raise SystemExit("clean eval must not set rx_style_eval")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "train_seeds=${NEW_SEEDS[*]}"
echo "s1_retrain=FORBIDDEN"
echo "stress=FORBIDDEN"
echo "day5=FORBIDDEN"
echo "rx2=FORBIDDEN"

i=0
while [[ ${i} -lt ${#NEW_SEEDS[@]} ]]; do
  pids=()
  status=0
  j=0
  while [[ ${j} -lt ${#GPU_ARR[@]} && $((i + j)) -lt ${#NEW_SEEDS[@]} ]]; do
    run_one "${GPU_ARR[${j}]}" "${NEW_SEEDS[$((i + j))]}" &
    pids+=("$!")
    j=$((j + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "a S0 job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  i=$((i + ${#GPU_ARR[@]}))
done

"${PY}" - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
out_json = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s0_seeds234_diag.json")
out_md = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s0_seeds234_diag.md")
FOCUS = [2, 3, 4]
ALL = [0, 1, 2, 3, 4]


def load(model, seed):
    return json.loads((root / model / f"seed_{seed}" / "metrics.json").read_text())


def pct(x):
    return round(100.0 * x, 1)


def row(seed):
    c = load("C_full_ratio", seed)
    s0 = load("C_full_ratio_paired_clean", seed)
    s1 = load("C_full_ratio_paired_scale", seed)
    d0 = pct(s0["window_acc"]) - pct(c["window_acc"])
    d1 = pct(s1["window_acc"]) - pct(c["window_acc"])
    return {
        "seed": seed,
        "cprime_window": pct(c["window_acc"]),
        "s0_window": pct(s0["window_acc"]),
        "s1_window": pct(s1["window_acc"]),
        "delta_s0_pp": round(d0, 1),
        "delta_s1_pp": round(d1, 1),
        "s0_file": pct(s0["file_acc"]),
        "s1_file": pct(s1["file_acc"]),
        "s0_pass": d0 >= -2.0 and d0 > -15.0,
        "s0_collapse": d0 <= -15.0,
        "s1_pass": d1 >= -2.0 and d1 > -15.0,
    }


rows = [row(s) for s in ALL]
focus = [r for r in rows if r["seed"] in FOCUS]
n_s0_pass = sum(1 for r in focus if r["s0_pass"])
n_s0_fail = 3 - n_s0_pass
mean_s0 = statistics.mean(r["delta_s0_pp"] for r in focus)
mean_s1 = statistics.mean(r["delta_s1_pp"] for r in focus)
if n_s0_fail >= 2:
    reading = "PAIRING_TAX"
elif n_s0_pass >= 2:
    reading = "SCALE_TAX"
else:
    reading = "MIXED"

payload = {
    "day5_used": False,
    "rx2_used": False,
    "s1_retrained": False,
    "stress": False,
    "focus_seeds": FOCUS,
    "rows": rows,
    "focus": {
        "n_s0_pass": f"{n_s0_pass}/3",
        "mean_delta_s0_pp": round(mean_s0, 2),
        "mean_delta_s1_pp": round(mean_s1, 2),
    },
    "reading": reading,
    "note": (
        "S1 5-seed CLEAN_FAIL is not moved. "
        "PAIRING_TAX: S0 also fails on >=2/3 of seeds 2-4. "
        "SCALE_TAX: S0 holds on >=2/3 while S1 already failed those seeds. "
        "Do not open RX2. Do not retune."
    ),
}
out_json.write_text(json.dumps(payload, indent=2) + "\n")
lines = [
    "# S0 seeds 2/3/4 diagnostic (after S1 CLEAN_FAIL)",
    "",
    f"reading={reading}  s1_5seed=CLEAN_FAIL_unchanged  day5=unused  rx2=unused",
    "",
    "| seed | C' win | S0 win | S1 win | Δ S0 | Δ S1 | S0 gate |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for r in rows:
    lines.append(
        f"| {r['seed']} | {r['cprime_window']:.1f} | {r['s0_window']:.1f} | {r['s1_window']:.1f} | "
        f"{r['delta_s0_pp']:+.1f} | {r['delta_s1_pp']:+.1f} | "
        f"{'PASS' if r['s0_pass'] else 'FAIL'} |"
    )
lines.extend(
    [
        "",
        f"Focus 2/3/4: S0 mean Δ={mean_s0:+.2f} ({n_s0_pass}/3 PASS); S1 mean Δ={mean_s1:+.2f}",
        "",
        "PAIRING_TAX = S0 fails ≥2/3 of seeds 2–4 (two-forward / pairing).",
        "SCALE_TAX = S0 holds ≥2/3 while S1 already failed those seeds.",
        "S1 CLEAN_FAIL is not moved. RX2 closed. Do not retune.",
        "",
    ]
)
out_md.write_text("\n".join(lines))
print(out_md.read_text())
print("wrote", out_json)
print("wrote", out_md)
print("READING", reading)
PY

echo "S0 seeds 2/3/4 finished. S1 not retrained. Stress unused. RX2 unused."
