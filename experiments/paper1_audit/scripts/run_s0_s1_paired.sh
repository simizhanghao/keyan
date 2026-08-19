#!/usr/bin/env bash
# 2B-0: S0 paired-clean then S1 paired-scale. Frozen C' architecture.
# Unique flag: --paired-view clean | oob_scale.
# Day4 val. Day5 unused. No RX2. No stress. No retune.
# Gate 0 FAIL stops before S1.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"
SUMMARY="${OUT_ROOT}/s0_s1_clean_vs_cprime.json"

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

run_one() {
  local gpu="$1"
  local seed="$2"
  local name="$3"
  local paired="$4"
  local run_dir="${OUT_ROOT}/runs/${name}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_val/${name}/seed_${seed}"
  local log="${LOG_DIR}/${name}_seed${seed}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${name} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  {
    echo "=== TRAIN ${name} seed=${seed} GPU=${gpu} paired_view=${paired} fft_source=full oob_norm=ratio Day4-only ==="
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
      --paired-view "${paired}" \
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
    echo "=== EVAL VAL ${name} seed=${seed} (clean Day4, not Day5, no paired view) ==="
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
print(f"${name} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} paired={m.get('paired_view')} fft={m.get('fft_source')} oob={m.get('oob_norm')}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("eval_split") not in {"val", None}:
    raise SystemExit("eval_split must be val")
if m.get("day5_used") is True:
    raise SystemExit("day5_used must be false")
if m.get("fft_source") != "full":
    raise SystemExit(f"fft_source must be full, got {m.get('fft_source')}")
if m.get("oob_norm") != "ratio":
    raise SystemExit(f"oob_norm must be ratio, got {m.get('oob_norm')}")
if m.get("paired_view") != "${paired}":
    raise SystemExit(f"paired_view must be ${paired} from ckpt, got {m.get('paired_view')}")
if m.get("rx_style_eval") is True:
    raise SystemExit("clean eval must not set rx_style_eval")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_arm() {
  local name="$1"
  local paired="$2"
  local status=0
  echo "======== arm=${name} paired_view=${paired} gpu0=${GPU_ARR[0]} gpu1=${GPU_ARR[1]} ========"
  run_one "${GPU_ARR[0]}" 0 "${name}" "${paired}" &
  local pid0=$!
  run_one "${GPU_ARR[1]}" 1 "${name}" "${paired}" &
  local pid1=$!
  if ! wait "${pid0}"; then status=1; fi
  if ! wait "${pid1}"; then status=1; fi
  if [[ "${status}" -ne 0 ]]; then
    echo "a ${name} job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=0 1"
echo "fft_source=full"
echo "oob_norm=ratio"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "stress=FORBIDDEN"
echo "training=S0_then_S1"

run_arm "C_full_ratio_paired_clean" "clean"

"${PY}" - <<PY
import json
from pathlib import Path

root = Path("${OUT_ROOT}/eval_val")
print("# Gate 0 — S0 paired-clean vs frozen C' (Δ ≥ -2 pp, 2/2)")
print("| seed | C' win | S0 win | Δ | gate | S0 file |")
print("| ---: | ---: | ---: | ---: | --- | ---: |")
rows = []
all_pass = True
for seed in (0, 1):
    cprime = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
    s0 = json.loads((root / "C_full_ratio_paired_clean" / f"seed_{seed}" / "metrics.json").read_text())
    dw = 100 * s0["window_acc"] - 100 * cprime["window_acc"]
    collapsed = s0["window_acc"] < 0.15
    gate = (dw >= -2.0) and (not collapsed)
    all_pass = all_pass and gate
    print(
        f"| {seed} | {100*cprime['window_acc']:.1f} | {100*s0['window_acc']:.1f} | {dw:+.1f} | "
        f"{'PASS' if gate else 'FAIL'} | {100*s0['file_acc']:.1f} |"
    )
    rows.append(
        {
            "seed": seed,
            "cprime_window": round(100 * cprime["window_acc"], 1),
            "s0_window": round(100 * s0["window_acc"], 1),
            "delta_s0_pp": round(dw, 1),
            "s0_gate": gate,
            "s0_file": round(100 * s0["file_acc"], 1),
        }
    )
out = {
    "day5_used": False,
    "fft_source": "full",
    "oob_norm": "ratio",
    "gate0_pass": all_pass,
    "gate1_pass": None,
    "s1_trained": False,
    "rows": rows,
    "verdict": "GATE0_PASS" if all_pass else "GATE0_FAIL",
}
Path("${SUMMARY}").write_text(json.dumps(out, indent=2) + "\n")
print("gate0", "PASS" if all_pass else "FAIL")
print("wrote", "${SUMMARY}")
if not all_pass:
    print("Gate 0 FAIL; do not train S1; do not retune; do not interpret scale robustness")
    raise SystemExit(2)
PY
gate0_status=$?
if [[ "${gate0_status}" -eq 2 ]]; then
  echo "S0 finished. S1 skipped. Stress unused."
  exit 0
fi
if [[ "${gate0_status}" -ne 0 ]]; then
  exit "${gate0_status}"
fi

run_arm "C_full_ratio_paired_scale" "oob_scale"

"${PY}" - <<PY
import json
from pathlib import Path

root = Path("${OUT_ROOT}/eval_val")
path = Path("${SUMMARY}")
out = json.loads(path.read_text())
print("# Gate 1 — S1 paired-scale vs frozen C' (Δ ≥ -2 pp, 2/2)")
print("| seed | C' win | S0 win | S1 win | Δ S1 | gate | S1 file |")
print("| ---: | ---: | ---: | ---: | ---: | --- | ---: |")
all_pass = True
for row in out["rows"]:
    seed = row["seed"]
    cprime = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
    s1 = json.loads((root / "C_full_ratio_paired_scale" / f"seed_{seed}" / "metrics.json").read_text())
    dw = 100 * s1["window_acc"] - 100 * cprime["window_acc"]
    collapsed = s1["window_acc"] < 0.15
    gate = (dw >= -2.0) and (not collapsed)
    all_pass = all_pass and gate
    row["s1_window"] = round(100 * s1["window_acc"], 1)
    row["delta_s1_pp"] = round(dw, 1)
    row["s1_gate"] = gate
    row["s1_file"] = round(100 * s1["file_acc"], 1)
    print(
        f"| {seed} | {row['cprime_window']:.1f} | {row['s0_window']:.1f} | {row['s1_window']:.1f} | "
        f"{dw:+.1f} | {'PASS' if gate else 'FAIL'} | {row['s1_file']:.1f} |"
    )
out["gate1_pass"] = all_pass
out["s1_trained"] = True
out["verdict"] = "CLEAN_PASS" if (out["gate0_pass"] and all_pass) else "GATE1_FAIL"
path.write_text(json.dumps(out, indent=2) + "\n")
print("verdict", out["verdict"])
print("wrote", path)
if out["verdict"] != "CLEAN_PASS":
    print("clean gate FAIL; do not retune; do not open oob_scale stress")
else:
    print("clean 2/2 PASS; stress is a later eval")
PY

echo "2B-0 S0/S1 finished. Day5 unused. Stress unused."
