#!/usr/bin/env bash
# 2A-5 Beat 3: C_fft-only seeds 0 and 1, eval-only oob_scale + full RX.
# No training. Day5 unused. No RX2. No cell D. No 5-seed.
# Does not overwrite C_full_ratio_rx_* or C_full_ratio_rms_rx_*.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_NAME="C_full_ratio_inband"
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

for seed in 0 1; do
  ckpt="${OUT_ROOT}/runs/${CKPT_NAME}/seed_${seed}/best.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing C_fft seed ${seed} checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

run_eval() {
  local gpu="$1"
  local seed="$2"
  local name="$3"
  local factor="${4:-}"
  local ckpt="${OUT_ROOT}/runs/${CKPT_NAME}/seed_${seed}/best.pt"
  local eval_dir="${OUT_ROOT}/eval_val/${name}/seed_${seed}"
  local log="${LOG_DIR}/${name}_seed${seed}.log"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${name} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  {
    echo "=== EVAL ${name} seed=${seed} GPU=${gpu} factor=${factor:-full} fft_source=inband oob_norm=ratio Day4-only ==="
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
      --rx-style-eval \
      ${factor:+--rx-factor "${factor}"} \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${ckpt}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${name} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} fft={m.get('fft_source')} oob={m.get('oob_norm')} rx={m.get('rx_style_eval')} factor={m.get('rx_factor')}")
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
if m.get("fft_source") != "inband":
    raise SystemExit(f"fft_source must be inband, got {m.get('fft_source')}")
if m.get("oob_norm") != "ratio":
    raise SystemExit(f"oob_norm must be ratio, got {m.get('oob_norm')}")
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
echo "seeds=0 1"
echo "model=${CKPT_NAME}"
echo "fft_source=inband"
echo "oob_norm=ratio"
echo "day5_eval=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "cell_d=FORBIDDEN"
echo "training=FORBIDDEN"

wave=0
for seed in 0 1; do
  echo "======== wave=${wave} seed=${seed} gpu0=${GPU_ARR[0]} gpu1=${GPU_ARR[1]} ========"
  run_eval "${GPU_ARR[0]}" "${seed}" "C_full_ratio_inband_rx_oob_scale" "oob_scale" &
  pid0=$!
  run_eval "${GPU_ARR[1]}" "${seed}" "C_full_ratio_inband_rx_style" "" &
  pid1=$!
  status=0
  if ! wait "${pid0}"; then status=1; fi
  if ! wait "${pid1}"; then status=1; fi
  if [[ "${status}" -ne 0 ]]; then
    echo "a C_fft RX stress job failed; see ${LOG_DIR}" >&2
    exit 1
  fi
  wave=$((wave + 1))
done

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")


def win(m):
    return 100.0 * m["window_acc"]


def fil(m):
    return 100.0 * m["file_acc"]


def scale_bin(drop):
    if drop < 5:
        return "KILLED"
    if drop < 15:
        return "PARTIAL"
    return "NOT_KILLED"


rows = []
print("# C_fft RX stress vs own clean (Day4 window; primary = oob_scale drop)")
print("| seed | clean | oob_scale | Δ oob | full RX | Δ full | C' oob Δ | C' full Δ | bin |")
print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for seed in (0, 1):
    clean = json.loads((root / "C_full_ratio_inband" / f"seed_{seed}" / "metrics.json").read_text())
    scale = json.loads((root / "C_full_ratio_inband_rx_oob_scale" / f"seed_{seed}" / "metrics.json").read_text())
    full = json.loads((root / "C_full_ratio_inband_rx_style" / f"seed_{seed}" / "metrics.json").read_text())
    cprime = json.loads((root / "C_full_ratio" / f"seed_{seed}" / "metrics.json").read_text())
    cprime_scale = json.loads((root / "C_full_ratio_rx_oob_scale" / f"seed_{seed}" / "metrics.json").read_text())
    cprime_full = json.loads((root / "C_full_ratio_rx_style" / f"seed_{seed}" / "metrics.json").read_text())
    if clean.get("fft_source") != "inband" or scale.get("fft_source") != "inband" or full.get("fft_source") != "inband":
        raise SystemExit("C_fft stress metrics must keep fft_source=inband")
    if clean.get("oob_norm") != "ratio" or scale.get("oob_norm") != "ratio" or full.get("oob_norm") != "ratio":
        raise SystemExit("C_fft stress metrics must keep oob_norm=ratio")
    oob_drop = win(clean) - win(scale)
    full_drop = win(clean) - win(full)
    row = {
        "seed": seed,
        "clean_window": round(win(clean), 1),
        "oob_scale_window": round(win(scale), 1),
        "full_rx_window": round(win(full), 1),
        "oob_scale_drop_pp": round(oob_drop, 1),
        "full_rx_drop_pp": round(full_drop, 1),
        "oob_scale_file_drop_pp": round(fil(clean) - fil(scale), 1),
        "full_rx_file_drop_pp": round(fil(clean) - fil(full), 1),
        "cprime_oob_scale_drop_pp": round(win(cprime) - win(cprime_scale), 1),
        "cprime_full_rx_drop_pp": round(win(cprime) - win(cprime_full), 1),
        "scale_reading": scale_bin(oob_drop),
    }
    rows.append(row)
    print(
        f"| {seed} | {row['clean_window']:.1f} | {row['oob_scale_window']:.1f} | "
        f"{row['oob_scale_drop_pp']:.1f} | {row['full_rx_window']:.1f} | {row['full_rx_drop_pp']:.1f} | "
        f"{row['cprime_oob_scale_drop_pp']:.1f} | {row['cprime_full_rx_drop_pp']:.1f} | {row['scale_reading']} |"
    )

mean_oob = sum(r["oob_scale_drop_pp"] for r in rows) / len(rows)
mean_full = sum(r["full_rx_drop_pp"] for r in rows) / len(rows)
headline = scale_bin(mean_oob)
bins = {r["scale_reading"] for r in rows}
if len(bins) > 1:
    headline = "DISAGREE_" + "+".join(sorted(bins))

out = {
    "day5_used": False,
    "training": False,
    "fft_source": "inband",
    "oob_norm": "ratio",
    "cell": "C",
    "seeds": [0, 1],
    "rows": rows,
    "mean_oob_scale_drop_pp": round(mean_oob, 1),
    "mean_full_rx_drop_pp": round(mean_full, 1),
    "scale_reading": headline,
    "frozen_cprime_mean_oob_scale_drop_pp": 28.7,
    "frozen_cprime_mean_full_rx_drop_pp": 30.3,
}
path = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/cfft_rx_stress.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"mean oob_scale drop {out['mean_oob_scale_drop_pp']:.1f}  mean full RX drop {out['mean_full_rx_drop_pp']:.1f}")
print("scale_reading", headline)
print("wrote", path)
print("Do not open D / 5-seed / Day5 / RX2 from this file automatically.")
PY

echo "C_fft seed 0/1 RX stress finished. Day5 unused. No training. No D."
