#!/usr/bin/env bash
# Paper 2 Phase 2A-2: C1 seed 0 only, eval-only OOB-scale + full-RX stress.
# No training. Day5 unused. Seed 1 unused. No RX2. No 5-seed.
# Frozen C' R0/R6 / 7-arm tables are not rerun.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_NAME="C_full_ratio_rms"
SEED=0
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"
CKPT="${OUT_ROOT}/runs/${CKPT_NAME}/seed_${SEED}/best.pt"

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
if [[ ! -f "${CKPT}" ]]; then
  echo "missing C1 seed 0 checkpoint: ${CKPT}" >&2
  exit 1
fi

run_eval() {
  local gpu="$1"
  local name="$2"
  local factor="${3:-}"
  local eval_dir="${OUT_ROOT}/eval_val/${name}/seed_${SEED}"
  local log="${LOG_DIR}/${name}_seed${SEED}.log"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${name} seed=${SEED} (metrics exist; recipe freeze)"
    return 0
  fi
  {
    echo "=== EVAL ${name} seed=${SEED} GPU=${gpu} factor=${factor:-full} oob_norm=ratio_rms Day4-only ==="
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
      --seed "${SEED}" \
      --model-type rf_hstu \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --use-chirp-embedding \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --oob-norm ratio_rms \
      --rx-style-eval \
      ${factor:+--rx-factor "${factor}"} \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${CKPT}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${name}  file_acc={100*m['file_acc']:.1f}%  window_acc={100*m['window_acc']:.1f}%  files={m['num_files']}  rx={m.get('rx_style_eval')} factor={m.get('rx_factor')}")
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
if m.get("oob_norm") != "ratio_rms":
    raise SystemExit(f"oob_norm must be ratio_rms, got {m.get('oob_norm')}")
factor = "${factor}"
if factor:
    if m.get("rx_factor") != factor:
        raise SystemExit(f"rx_factor must be {factor}, got {m.get('rx_factor')}")
    atoms = set(m.get("rx_enabled_atoms") or [])
    if atoms != {"oob_scale"}:
        raise SystemExit(f"enabled atoms {atoms} != {{oob_scale}}")
    if m.get("rx_oob_scale_enabled") is not True:
        raise SystemExit("rx_oob_scale_enabled must be true")
    for key in ("rx_tilt_enabled", "rx_gain_enabled", "rx_phase_enabled", "rx_noise_enabled"):
        if m.get(key):
            raise SystemExit(f"{key} must be false on oob_scale arm")
else:
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

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seed=${SEED}"
echo "ckpt=${CKPT}"
echo "oob_norm=ratio_rms"
echo "day5_eval=FORBIDDEN"
echo "training=FORBIDDEN"
echo "seed1=FORBIDDEN"

run_eval "${GPU_ARR[0]}" "C_full_ratio_rms_rx_oob_scale" "oob_scale" &
pid0=$!
run_eval "${GPU_ARR[1]}" "C_full_ratio_rms_rx_style" "" &
pid1=$!
status=0
if ! wait "${pid0}"; then status=1; fi
if ! wait "${pid1}"; then status=1; fi
if [[ "${status}" -ne 0 ]]; then
  echo "a C1 seed0 RX stress job failed; see ${LOG_DIR}" >&2
  exit 1
fi

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
c1 = json.loads((root / "C_full_ratio_rms" / "seed_0" / "metrics.json").read_text())
c1_scale = json.loads((root / "C_full_ratio_rms_rx_oob_scale" / "seed_0" / "metrics.json").read_text())
c1_full = json.loads((root / "C_full_ratio_rms_rx_style" / "seed_0" / "metrics.json").read_text())
cprime = json.loads((root / "C_full_ratio" / "seed_0" / "metrics.json").read_text())
cprime_scale = json.loads((root / "C_full_ratio_rx_oob_scale" / "seed_0" / "metrics.json").read_text())
cprime_full = json.loads((root / "C_full_ratio_rx_style" / "seed_0" / "metrics.json").read_text())

def win(m):
    return 100.0 * m["window_acc"]

def fil(m):
    return 100.0 * m["file_acc"]

c1_scale_drop = win(c1) - win(c1_scale)
c1_full_drop = win(c1) - win(c1_full)
cprime_scale_drop = win(cprime) - win(cprime_scale)
cprime_full_drop = win(cprime) - win(cprime_full)

if c1_scale_drop < 5:
    scale_read = "KILLED"
elif c1_scale_drop < 15:
    scale_read = "PARTIAL"
else:
    scale_read = "NOT_TRANSFERRED"

out = {
    "day5_used": False,
    "seed": 0,
    "seed1_used": False,
    "training": False,
    "c1_clean_window": round(win(c1), 1),
    "c1_oob_scale_window": round(win(c1_scale), 1),
    "c1_full_rx_window": round(win(c1_full), 1),
    "c1_oob_scale_drop_pp": round(c1_scale_drop, 1),
    "c1_full_rx_drop_pp": round(c1_full_drop, 1),
    "c1_oob_scale_file_drop_pp": round(fil(c1) - fil(c1_scale), 1),
    "c1_full_rx_file_drop_pp": round(fil(c1) - fil(c1_full), 1),
    "cprime_seed0_oob_scale_drop_pp": round(cprime_scale_drop, 1),
    "cprime_seed0_full_rx_drop_pp": round(cprime_full_drop, 1),
    "frozen_cprime_mean_oob_scale_drop_pp": 28.7,
    "frozen_cprime_mean_full_rx_drop_pp": 30.3,
    "scale_reading": scale_read,
}
path = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/c1_seed0_rx_stress.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print("# C1 seed 0 RX stress (Day4 window; seed 1 unused)")
print(f"| arm | C1 clean | C1 stressed | C1 drop | C' seed0 drop | C' mean |")
print(f"| --- | ---: | ---: | ---: | ---: | ---: |")
print(f"| oob_scale | {out['c1_clean_window']:.1f} | {out['c1_oob_scale_window']:.1f} | {out['c1_oob_scale_drop_pp']:.1f} | {out['cprime_seed0_oob_scale_drop_pp']:.1f} | 28.7 |")
print(f"| full_rx | {out['c1_clean_window']:.1f} | {out['c1_full_rx_window']:.1f} | {out['c1_full_rx_drop_pp']:.1f} | {out['cprime_seed0_full_rx_drop_pp']:.1f} | 30.3 |")
print("scale_reading", scale_read)
print("wrote", path)
PY

echo "C1 seed 0 RX stress finished. Day5 unused. Seed 1 unused. No 5-seed."
