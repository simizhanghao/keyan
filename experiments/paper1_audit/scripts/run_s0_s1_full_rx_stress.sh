#!/usr/bin/env bash
# 2B-0 Gate 3: S0 and S1 seeds 0/1, eval-only full RX-style.
# No training. Day5 unused. No RX2. No new module.
# Does not overwrite C_full_ratio_rx_style or oob_scale dirs.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
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
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "need two GPUs, got GPUS=${GPUS}" >&2
  exit 1
fi

for name in C_full_ratio_paired_clean C_full_ratio_paired_scale; do
  for seed in 0 1; do
    ckpt="${OUT_ROOT}/runs/${name}/seed_${seed}/best.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "missing checkpoint: ${ckpt}" >&2
      exit 1
    fi
  done
done

run_eval() {
  local gpu="$1"
  local seed="$2"
  local ckpt_name="$3"
  local eval_name="$4"
  local ckpt="${OUT_ROOT}/runs/${ckpt_name}/seed_${seed}/best.pt"
  local eval_dir="${OUT_ROOT}/eval_val/${eval_name}/seed_${seed}"
  local log="${LOG_DIR}/${eval_name}_seed${seed}.log"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${eval_name} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  {
    echo "=== EVAL ${eval_name} seed=${seed} GPU=${gpu} factor=all fft_source=full oob_norm=ratio Day4-only ==="
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
      --rx-style-eval \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${ckpt}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${eval_name} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} rx={m.get('rx_style_eval')} factor={m.get('rx_factor')} fft={m.get('fft_source')}")
if m.get("rx_style_eval") is not True:
    raise SystemExit("rx_style_eval must be true")
if m.get("rx_inband_locked") is not True:
    raise SystemExit("rx_inband_locked must be true")
if m.get("training") is not False:
    raise SystemExit("training must be false")
if m.get("eval_split") != "val":
    raise SystemExit("eval_split must be val")
if m.get("day5_used") is not False:
    raise SystemExit("day5_used must be false")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("fft_source") != "full":
    raise SystemExit(f"fft_source must be full, got {m.get('fft_source')}")
if m.get("oob_norm") != "ratio":
    raise SystemExit(f"oob_norm must be ratio, got {m.get('oob_norm')}")
if m.get("rx_factor") not in {"all", None, ""}:
    raise SystemExit(f"full RX must have rx_factor=all, got {m.get('rx_factor')}")
atoms = set(m.get("rx_enabled_atoms") or [])
expect = {"tilt", "oob_scale", "gain", "phase", "noise"}
if atoms != expect:
    raise SystemExit(f"full RX atoms {atoms} != {expect}")
for key in ("rx_tilt_enabled", "rx_oob_scale_enabled", "rx_gain_enabled", "rx_phase_enabled", "rx_noise_enabled"):
    if m.get(key) is not True:
        raise SystemExit(f"{key} must be true on full RX")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_arm() {
  local ckpt_name="$1"
  local eval_name="$2"
  local status=0
  echo "======== ${eval_name} gpu0=${GPU_ARR[0]} gpu1=${GPU_ARR[1]} ========"
  run_eval "${GPU_ARR[0]}" 0 "${ckpt_name}" "${eval_name}" &
  local pid0=$!
  run_eval "${GPU_ARR[1]}" 1 "${ckpt_name}" "${eval_name}" &
  local pid1=$!
  if ! wait "${pid0}"; then status=1; fi
  if ! wait "${pid1}"; then status=1; fi
  if [[ "${status}" -ne 0 ]]; then
    echo "a ${eval_name} job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=0 1"
echo "fft_source=full"
echo "oob_norm=ratio"
echo "rx_factor=all"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "training=FORBIDDEN"

run_arm "C_full_ratio_paired_clean" "C_full_ratio_paired_clean_rx_style"
run_arm "C_full_ratio_paired_scale" "C_full_ratio_paired_scale_rx_style"

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")


def win(m):
    return 100.0 * m["window_acc"]


def fil(m):
    return 100.0 * m["file_acc"]


def full_bin(drop):
    if drop < 15:
        return "TRACKS_SCALE"
    if drop < 20:
        return "WITHIN_IDEAL"
    return "RESIDUAL"


print("# Gate 3 — full RX vs own clean (Day4 window; headline = S1; ideal <15–20)")
print("| arm | seed | clean | full RX | Δ | C' Δ | bin |")
print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
rows = []
for arm, clean_name, stress_name in (
    ("S0", "C_full_ratio_paired_clean", "C_full_ratio_paired_clean_rx_style"),
    ("S1", "C_full_ratio_paired_scale", "C_full_ratio_paired_scale_rx_style"),
):
    for seed in (0, 1):
        clean = json.loads((root / clean_name / f"seed_{seed}" / "metrics.json").read_text())
        stress = json.loads((root / stress_name / f"seed_{seed}" / "metrics.json").read_text())
        cprime = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
        cprime_full = json.loads((root / "C_full_ratio_rx_style" / f"seed_{seed}" / "metrics.json").read_text())
        drop = win(clean) - win(stress)
        row = {
            "arm": arm,
            "seed": seed,
            "clean_window": round(win(clean), 1),
            "full_rx_window": round(win(stress), 1),
            "full_rx_drop_pp": round(drop, 1),
            "full_rx_file_drop_pp": round(fil(clean) - fil(stress), 1),
            "cprime_full_rx_drop_pp": round(win(cprime) - win(cprime_full), 1),
            "full_reading": full_bin(drop),
        }
        rows.append(row)
        print(
            f"| {arm} | {seed} | {row['clean_window']:.1f} | {row['full_rx_window']:.1f} | "
            f"{row['full_rx_drop_pp']:.1f} | {row['cprime_full_rx_drop_pp']:.1f} | {row['full_reading']} |"
        )

s0 = [r for r in rows if r["arm"] == "S0"]
s1 = [r for r in rows if r["arm"] == "S1"]
mean_s0 = sum(r["full_rx_drop_pp"] for r in s0) / len(s0)
mean_s1 = sum(r["full_rx_drop_pp"] for r in s1) / len(s1)
headline = full_bin(mean_s1)
out = {
    "day5_used": False,
    "training": False,
    "fft_source": "full",
    "oob_norm": "ratio",
    "rx_factor": "all",
    "rows": rows,
    "mean_s0_full_rx_drop_pp": round(mean_s0, 1),
    "mean_s1_full_rx_drop_pp": round(mean_s1, 1),
    "full_reading": headline,
    "frozen_cprime_mean_full_rx_drop_pp": 30.3,
    "s1_oob_scale_drop_pp": 1.3,
}
path = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s0_s1_rx_full.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"mean S0 full drop {out['mean_s0_full_rx_drop_pp']:.1f}  mean S1 full drop {out['mean_s1_full_rx_drop_pp']:.1f}")
print("full_reading", headline)
print("wrote", path)
print("Do not retune. Do not open RX2 / 5-seed / a new module from this file.")
PY

echo "2B-0 Gate 3 full RX finished. Day5 unused. RX2 unused."
