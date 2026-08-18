#!/usr/bin/env bash
# Paper 1 Audit 1C.mech: RX-style / factor eval on frozen 1C C' checkpoints.
# No training. Day4 val only. Day5 is not evaluated.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-${SEED}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SMOKE="${SMOKE:-0}"
FACTOR="${FACTOR:-}"
FACTORS="${FACTORS:-}"
NAME="C_full_ratio_rx_style"
CKPT_NAME="C_full_ratio"
VALID_FACTORS="tilt oob_scale gain phase noise spec nonspec"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"

if [[ -n "${FACTORS}" && -n "${FACTOR}" ]]; then
  echo "set FACTORS or FACTOR, not both" >&2
  exit 1
fi

validate_factor() {
  local f="$1"
  case " ${VALID_FACTORS} " in
    *" ${f} "*) ;;
    *)
      echo "invalid FACTOR=${f}; expected one of ${VALID_FACTORS}" >&2
      exit 1
      ;;
  esac
}

apply_factor_name() {
  if [[ -n "${FACTOR}" ]]; then
    validate_factor "${FACTOR}"
    NAME="C_full_ratio_rx_${FACTOR}"
  else
    NAME="C_full_ratio_rx_style"
  fi
}

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
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
if [[ "${SMOKE}" == "1" ]]; then
  SEED_ARR=(0)
  NUM_WORKERS=0
fi

run_one() {
  local gpu="$1"
  local seed="$2"
  local ckpt="${OUT_ROOT}/runs/${CKPT_NAME}/seed_${seed}/best.pt"
  local eval_dir="${OUT_ROOT}/eval_val/${NAME}/seed_${seed}"
  local log="${LOG_DIR}/${NAME}_seed${seed}.log"
  if [[ "${SMOKE}" == "1" ]]; then
    eval_dir="${OUT_ROOT}/eval_val/${NAME}/smoke_seed_${seed}"
    log="${LOG_DIR}/${NAME}_smoke_seed${seed}.log"
  fi
  mkdir -p "${eval_dir}"
  if [[ "${SMOKE}" != "1" && -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${NAME} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing frozen C' checkpoint: ${ckpt}" >&2
    return 1
  fi
  {
    echo "=== EVAL RX ${NAME} seed=${seed} GPU=${gpu} smoke=${SMOKE} (Day4, not Day5) ==="
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
      --oob-norm ratio \
      --rx-style-eval \
      ${FACTOR:+--rx-factor "${FACTOR}"} \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${ckpt}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME}  file_acc={100*m['file_acc']:.1f}%  window_acc={100*m['window_acc']:.1f}%  files={m['num_files']}  rx={m.get('rx_style_eval')} factor={m.get('rx_factor')}")
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
if "${FACTOR}":
    if m.get("rx_factor") != "${FACTOR}":
        raise SystemExit(f"rx_factor must be ${FACTOR}, got {m.get('rx_factor')}")
    expect = {
        "tilt": {"tilt"},
        "oob_scale": {"oob_scale"},
        "gain": {"gain"},
        "phase": {"phase"},
        "noise": {"noise"},
        "spec": {"tilt", "oob_scale", "gain"},
        "nonspec": {"phase", "noise"},
    }["${FACTOR}"]
    atoms = set(m.get("rx_enabled_atoms") or [])
    if atoms != expect:
        raise SystemExit(f"enabled atoms {atoms} != {expect}")
    for atom, key in (
        ("tilt", "rx_tilt_enabled"),
        ("oob_scale", "rx_oob_scale_enabled"),
        ("gain", "rx_gain_enabled"),
        ("phase", "rx_phase_enabled"),
        ("noise", "rx_noise_enabled"),
    ):
        if bool(m.get(key)) != (atom in expect):
            raise SystemExit(f"{key} mismatch")
    if "C_full_ratio_rx_style" in str(p) or str(p).endswith("C_full_ratio/metrics.json"):
        raise SystemExit("factor eval must not overwrite R0/R6")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_seed_wave() {
  apply_factor_name
  echo "======== factor=${FACTOR:-all} name=${NAME} ========"
  local n_gpu=${#GPU_ARR[@]}
  local idx=0
  while [[ ${idx} -lt ${#SEED_ARR[@]} ]]; do
    local pids=()
    local j
    for ((j = 0; j < n_gpu && idx + j < ${#SEED_ARR[@]}; j++)); do
      local local_seed="${SEED_ARR[$((idx + j))]}"
      echo "======== seed=${local_seed} gpu=${GPU_ARR[j]} ========"
      run_one "${GPU_ARR[j]}" "${local_seed}" &
      pids+=("$!")
    done
    local status=0
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if [[ "${status}" -ne 0 ]]; then
      echo "an RX eval job failed; see ${LOG_DIR}" >&2
      exit 1
    fi
    idx=$((idx + n_gpu))
  done
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "seeds=${SEED_ARR[*]}"
echo "smoke=${SMOKE}"
echo "day5_eval=FORBIDDEN"
echo "training=FORBIDDEN"
echo "factor=${FACTOR:-}"
echo "factors=${FACTORS:-}"

if [[ -n "${FACTORS}" ]]; then
  IFS=',' read -r -a FACTOR_ARR <<< "${FACTORS}"
  for FACTOR in "${FACTOR_ARR[@]}"; do
    run_seed_wave
  done
else
  run_seed_wave
fi

if [[ "${SMOKE}" == "1" ]]; then
  echo "RX smoke finished. Day5 was not evaluated. Not a 5-seed verdict."
elif [[ -n "${FACTORS}" ]]; then
  echo "RX-factor eval finished factors=${FACTORS} seeds=${SEEDS}. Day5 unused. R0/R6 not rerun."
  "${PY}" "${KEYAN}/experiments/paper1_audit/scripts/audit_rx_factor.py"
elif [[ -n "${FACTOR}" ]]; then
  echo "RX-factor eval finished factor=${FACTOR} seeds=${SEEDS}. Day5 unused. R0/R6 not rerun."
  echo "Attribution table waits until all 7 arms exist."
else
  echo "RX-style eval seeds finished (${SEEDS}). Day5 was not evaluated."
  "${PY}" "${KEYAN}/experiments/paper1_audit/scripts/audit_rx_style.py"
fi
