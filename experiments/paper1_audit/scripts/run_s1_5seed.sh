#!/usr/bin/env bash
# S1 five-seed stability. Frozen recipe. Seeds 0/1 already done; train 2/3/4 only.
# No S0 retrain. No RX2. No Day5. No retune. Clean FAIL skips stress.
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MANIFEST="${KEYAN}/data/paper/cross_day_day1to5_source_only.csv"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/matched_seed0"
LOG_DIR="${OUT_ROOT}/logs"
NAME="C_full_ratio_paired_scale"
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
    echo "missing frozen S1 seed ${seed}" >&2
    exit 1
  fi
done
for seed in 0 1 2 3 4; do
  if [[ ! -f "${OUT_ROOT}/eval_val/C_full_ratio/seed_${seed}/metrics.json" ]]; then
    echo "missing frozen C' seed ${seed}" >&2
    exit 1
  fi
done

COMMON_EVAL=(
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
  --fft-source full
  --paired-view off
  --window-size 8192
  --num-workers "${NUM_WORKERS}"
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --use-chirp-embedding
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --oob-norm ratio
  --mode classifier
  --file-vote-mode mean_logits
)

run_train() {
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
    echo "=== TRAIN ${NAME} seed=${seed} GPU=${gpu} paired_view=oob_scale fft_source=full oob_norm=ratio Day4-only ==="
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
      --paired-view oob_scale \
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
      "${COMMON_EVAL[@]}" \
      --seed "${seed}" \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${NAME} seed=${seed} file_acc={100*m['file_acc']:.1f}% window_acc={100*m['window_acc']:.1f}% files={m['num_files']} paired={m.get('paired_view')}")
if m["num_files"] != 24:
    raise SystemExit("expected 24 Day4 files")
if m.get("day5_used") is True:
    raise SystemExit("day5_used must be false")
if m.get("fft_source") != "full":
    raise SystemExit("fft_source must be full")
if m.get("oob_norm") != "ratio":
    raise SystemExit("oob_norm must be ratio")
if m.get("paired_view") != "oob_scale":
    raise SystemExit(f"paired_view must be oob_scale from ckpt, got {m.get('paired_view')}")
if m.get("rx_style_eval") is True:
    raise SystemExit("clean eval must not set rx_style_eval")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_stress() {
  local gpu="$1"
  local seed="$2"
  local eval_name="$3"
  local extra=("${@:4}")
  local ckpt="${OUT_ROOT}/runs/${NAME}/seed_${seed}/best.pt"
  local eval_dir="${OUT_ROOT}/eval_val/${eval_name}/seed_${seed}"
  local log="${LOG_DIR}/${eval_name}_seed${seed}.log"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${eval_name} seed=${seed} (metrics exist; recipe freeze)"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing S1 ckpt: ${ckpt}" >&2
    return 1
  fi
  {
    echo "=== EVAL ${eval_name} seed=${seed} GPU=${gpu} Day4-only ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      "${COMMON_EVAL[@]}" \
      --seed "${seed}" \
      --rx-style-eval \
      "${extra[@]}" \
      --checkpoint "${ckpt}" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"${eval_name} seed=${seed} window_acc={100*m['window_acc']:.1f}% rx={m.get('rx_style_eval')} factor={m.get('rx_factor')}")
if m.get("rx_style_eval") is not True or m.get("rx_inband_locked") is not True:
    raise SystemExit("rx lock failed")
if m.get("day5_used") is not False or m["num_files"] != 24:
    raise SystemExit("split/files check failed")
if m.get("fft_source") != "full" or m.get("oob_norm") != "ratio":
    raise SystemExit("norm check failed")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_wave() {
  local fn="$1"
  shift
  local extra=("$@")
  local i=0
  while [[ ${i} -lt ${#NEW_SEEDS[@]} ]]; do
    local pids=()
    local status=0
    local j=0
    while [[ ${j} -lt ${#GPU_ARR[@]} && $((i + j)) -lt ${#NEW_SEEDS[@]} ]]; do
      local seed="${NEW_SEEDS[$((i + j))]}"
      local gpu="${GPU_ARR[${j}]}"
      "${fn}" "${gpu}" "${seed}" "${extra[@]}" &
      pids+=("$!")
      j=$((j + 1))
    done
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then status=1; fi
    done
    if [[ "${status}" -ne 0 ]]; then
      echo "a job failed; see ${LOG_DIR}" >&2
      exit 1
    fi
    i=$((i + ${#GPU_ARR[@]}))
  done
}

echo "python=${PY}"
echo "gpus=${GPUS}"
echo "train_seeds=${NEW_SEEDS[*]}"
echo "s0_retrain=FORBIDDEN"
echo "day5=FORBIDDEN"
echo "rx2=FORBIDDEN"
echo "retune=FORBIDDEN"

run_wave run_train

"${PY}" - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
out_json = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s1_5seed_stability.json")
out_md = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s1_5seed_stability.md")
SEEDS = [0, 1, 2, 3, 4]


def load(model, seed):
    return json.loads((root / model / f"seed_{seed}" / "metrics.json").read_text())


def pct(x):
    return round(100.0 * x, 1)


rows = []
for seed in SEEDS:
    c = load("C_full_ratio", seed)
    s = load("C_full_ratio_paired_scale", seed)
    dw = pct(s["window_acc"]) - pct(c["window_acc"])
    collapse = dw <= -15.0
    clean_pass = (dw >= -2.0) and (not collapse)
    rows.append(
        {
            "seed": seed,
            "cprime_window": pct(c["window_acc"]),
            "s1_window": pct(s["window_acc"]),
            "delta_clean_pp": round(dw, 1),
            "s1_file": pct(s["file_acc"]),
            "clean_pass": clean_pass,
            "collapse": collapse,
        }
    )

deltas = [r["delta_clean_pp"] for r in rows]
n_pass = sum(1 for r in rows if r["clean_pass"])
n_collapse = sum(1 for r in rows if r["collapse"])
mean_d = statistics.mean(deltas)
clean_ok = (n_pass >= 4) and (n_collapse == 0) and (mean_d >= -2.0)
payload = {
    "day5_used": False,
    "rx2_used": False,
    "s0_retrained": False,
    "recipe": "frozen S1 paired oob_scale",
    "clean": {
        "rows": rows,
        "n_pass": f"{n_pass}/5",
        "n_collapse": n_collapse,
        "mean_delta_pp": round(mean_d, 2),
        "pass": clean_ok,
    },
    "scale": None,
    "full_rx": None,
    "verdict": "CLEAN_PASS" if clean_ok else "CLEAN_FAIL",
    "note": "Stress skipped on CLEAN_FAIL. RX2 closed.",
}
out_json.write_text(json.dumps(payload, indent=2) + "\n")
print("# S1 5-seed clean vs frozen C'")
print("| seed | C' win | S1 win | Δ | gate | collapse |")
print("| ---: | ---: | ---: | ---: | --- | --- |")
for r in rows:
    print(
        f"| {r['seed']} | {r['cprime_window']:.1f} | {r['s1_window']:.1f} | "
        f"{r['delta_clean_pp']:+.1f} | {'PASS' if r['clean_pass'] else 'FAIL'} | {r['collapse']} |"
    )
print(f"mean Δ={mean_d:+.2f}  pass={n_pass}/5  collapse={n_collapse}  verdict={payload['verdict']}")
print("wrote", out_json)
if not clean_ok:
    out_md.write_text(
        f"# S1 5-seed\n\nverdict=CLEAN_FAIL  meanΔ={mean_d:+.2f}  pass={n_pass}/5  collapse={n_collapse}\n"
        "Do not interpret scale. Do not open RX2. Do not retune.\n"
    )
    raise SystemExit(2)
PY
clean_status=$?
if [[ "${clean_status}" -eq 2 ]]; then
  echo "CLEAN_FAIL; stress skipped; RX2 closed"
  exit 0
fi
if [[ "${clean_status}" -ne 0 ]]; then
  exit "${clean_status}"
fi

echo "======== S1 oob_scale stress seeds ${NEW_SEEDS[*]} ========"
run_wave run_stress C_full_ratio_paired_scale_rx_oob_scale --rx-factor oob_scale
echo "======== S1 full RX stress seeds ${NEW_SEEDS[*]} ========"
run_wave run_stress C_full_ratio_paired_scale_rx_style

"${PY}" - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/eval_val")
path = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s1_5seed_stability.json")
out_md = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0/s1_5seed_stability.md")
SEEDS = [0, 1, 2, 3, 4]


def load(model, seed):
    return json.loads((root / model / f"seed_{seed}" / "metrics.json").read_text())


def pct(x):
    return round(100.0 * x, 1)


payload = json.loads(path.read_text())
scale_rows = []
full_rows = []
for seed in SEEDS:
    clean = load("C_full_ratio_paired_scale", seed)
    sc = load("C_full_ratio_paired_scale_rx_oob_scale", seed)
    fu = load("C_full_ratio_paired_scale_rx_style", seed)
    d_scale = pct(clean["window_acc"]) - pct(sc["window_acc"])
    d_full = pct(clean["window_acc"]) - pct(fu["window_acc"])
    if d_scale < 8:
        sbin = "STRONG"
    elif d_scale < 15:
        sbin = "PASS"
    else:
        sbin = "FAIL"
    fbin = "TRACKS_SCALE" if d_full < 15 else ("WITHIN_IDEAL" if d_full < 20 else "RESIDUAL")
    scale_rows.append(
        {
            "seed": seed,
            "clean": pct(clean["window_acc"]),
            "oob_scale": pct(sc["window_acc"]),
            "d_scale": round(d_scale, 1),
            "bin": sbin,
        }
    )
    full_rows.append(
        {
            "seed": seed,
            "clean": pct(clean["window_acc"]),
            "full_rx": pct(fu["window_acc"]),
            "d_full": round(d_full, 1),
            "bin": fbin,
        }
    )

scale_ds = [r["d_scale"] for r in scale_rows]
full_ds = [r["d_full"] for r in full_rows]
mean_s = statistics.mean(scale_ds)
std_s = statistics.stdev(scale_ds)
mean_f = statistics.mean(full_ds)
std_f = statistics.stdev(full_ds)
all_lt15 = all(d < 15 for d in scale_ds)
if any(d >= 15 for d in scale_ds) or mean_s >= 15:
    scale_verdict = "SCALE_FAIL"
elif mean_s < 8 and all_lt15:
    scale_verdict = "SCALE_STRONG"
else:
    scale_verdict = "SCALE_PASS"
full_verdict = "TRACKS_SCALE" if mean_f < 15 else ("WITHIN_IDEAL" if mean_f < 20 else "RESIDUAL")
overall = "S1_5SEED_GO" if (payload["clean"]["pass"] and scale_verdict != "SCALE_FAIL") else "S1_5SEED_HOLD"
payload["scale"] = {
    "rows": scale_rows,
    "mean_pp": round(mean_s, 2),
    "std_pp": round(std_s, 2),
    "verdict": scale_verdict,
}
payload["full_rx"] = {
    "rows": full_rows,
    "mean_pp": round(mean_f, 2),
    "std_pp": round(std_f, 2),
    "verdict": full_verdict,
    "hard_gate": False,
}
payload["verdict"] = overall
payload["note"] = "RX2 still closed. Do not retune from this table."
path.write_text(json.dumps(payload, indent=2) + "\n")

lines = [
    "# S1 5-seed stability (Day4, frozen recipe)",
    "",
    f"verdict={overall}  clean={payload['clean']['pass']}  scale={scale_verdict}  full={full_verdict}",
    f"s0_retrained=false  day5=unused  rx2=unused",
    "",
    "## Clean vs C'",
    "",
    "| seed | C' win | S1 win | Δ | gate | collapse |",
    "| ---: | ---: | ---: | ---: | --- | --- |",
]
for r in payload["clean"]["rows"]:
    lines.append(
        f"| {r['seed']} | {r['cprime_window']:.1f} | {r['s1_window']:.1f} | "
        f"{r['delta_clean_pp']:+.1f} | {'PASS' if r['clean_pass'] else 'FAIL'} | {r['collapse']} |"
    )
lines.extend(
    [
        "",
        f"mean Δ={payload['clean']['mean_delta_pp']:+.2f}  {payload['clean']['n_pass']}  collapse={payload['clean']['n_collapse']}",
        "",
        "## oob_scale drop vs own clean",
        "",
        "| seed | clean | oob_scale | D | bin |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
)
for r in scale_rows:
    lines.append(f"| {r['seed']} | {r['clean']:.1f} | {r['oob_scale']:.1f} | {r['d_scale']:.1f} | {r['bin']} |")
lines.extend(
    [
        "",
        f"mean D_scale={mean_s:.1f}±{std_s:.1f}  {scale_verdict}",
        "",
        "## full RX drop vs own clean (recorded, not a hard gate)",
        "",
        "| seed | clean | full RX | D | bin |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
)
for r in full_rows:
    lines.append(f"| {r['seed']} | {r['clean']:.1f} | {r['full_rx']:.1f} | {r['d_full']:.1f} | {r['bin']} |")
lines.extend(
    [
        "",
        f"mean D_full={mean_f:.1f}±{std_f:.1f}  {full_verdict}",
        "",
        "RX2 / Day5 / utility / factorization stay closed. Do not retune.",
        "",
    ]
)
out_md.write_text("\n".join(lines))
print(out_md.read_text())
print("wrote", path)
print("wrote", out_md)
print("VERDICT", overall)
PY

echo "S1 5-seed finished. Day5 unused. RX2 unused."
