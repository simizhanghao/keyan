#!/usr/bin/env bash
# Phase 2B-2: source-only real RX. Retrain C' then F0. Seeds 0/1, both directions.
# Do NOT load Day4 matched_seed0 checkpoints. No oracle. No F1. No CNN.
# Checkpoint = source val acc. Target test is eval-only.
# Default GPUS=5,6. Do not use GPU 4 (foreign VLLM).
set -euo pipefail

KEYAN="${KEYAN:-/data1/hcc/llm4RF/new_phase}"
DATA_ROOT="${DATA_ROOT:-/data1/hcc/llm4RF}"
PY="${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}"
GPUS="${GPUS:-5,6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUT_ROOT="${KEYAN}/experiments/paper1_audit/results/real_rx_source_only"
LOG_DIR="${OUT_ROOT}/logs"
CPRIME="C_full_ratio"
F0="C_full_ratio_init_paired_scale"
SEEDS=(0 1)
DIRECTIONS=(rx1_to_rx2 rx2_to_rx1)

if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

cd "${KEYAN}"
export PYTHONPATH="${KEYAN}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "need at least one GPU, got GPUS=${GPUS}" >&2
  exit 1
fi

manifest_for() {
  echo "${KEYAN}/data/paper/${1}_source_only.csv"
}

COMMON_MODEL=(
  --root "${DATA_ROOT}"
  --batch-size 128
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64
  --depth 2
  --device cuda
  --train-split train
  --val-split val
  --input-norm iq_rms
  --fft-norm log_zscore
  --fft-source full
  --window-size 8192
  --num-workers "${NUM_WORKERS}"
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --use-chirp-embedding
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --oob-norm ratio
)

run_cprime() {
  local gpu="$1"
  local direction="$2"
  local seed="$3"
  local manifest
  manifest="$(manifest_for "${direction}")"
  local run_dir="${OUT_ROOT}/runs/${direction}/${CPRIME}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_test/${direction}/${CPRIME}/seed_${seed}"
  local log="${LOG_DIR}/${direction}_${CPRIME}_seed${seed}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${direction} ${CPRIME} seed=${seed}"
    return 0
  fi
  {
    echo "=== TRAIN C'_RX ${direction} seed=${seed} GPU=${gpu} paired=off Day4_ckpt=FORBIDDEN ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      --manifest "${manifest}" \
      "${COMMON_MODEL[@]}" \
      --eval-split val \
      --paired-view off \
      --seed "${seed}" \
      --epochs 80 \
      --lr 3e-3 \
      --loss-type ce \
      --checkpoint-metric acc \
      --weight-decay 5e-4 \
      --label-smoothing 0 \
      --out-dir "${run_dir}"
    echo "=== EVAL TEST C'_RX ${direction} seed=${seed} (target only) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      --manifest "${manifest}" \
      "${COMMON_MODEL[@]}" \
      --eval-split test \
      --paired-view off \
      --seed "${seed}" \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"C' ${direction} seed=${seed} win={100*m['window_acc']:.1f} file={100*m['file_acc']:.1f} files={m['num_files']} split={m.get('eval_split')}")
if m.get("eval_split") != "test" or m["num_files"] != 24:
    raise SystemExit("C' test/files check failed")
if m.get("rx_style_eval") is True:
    raise SystemExit("target eval must not be synthetic rx-style")
if "matched_seed0" in str(m.get("checkpoint_path", "")):
    raise SystemExit("Day4 matched_seed0 ckpt leaked into RX")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

run_f0() {
  local gpu="$1"
  local direction="$2"
  local seed="$3"
  local manifest
  manifest="$(manifest_for "${direction}")"
  local init_ckpt="${OUT_ROOT}/runs/${direction}/${CPRIME}/seed_${seed}/best.pt"
  local run_dir="${OUT_ROOT}/runs/${direction}/${F0}/seed_${seed}"
  local eval_dir="${OUT_ROOT}/eval_test/${direction}/${F0}/seed_${seed}"
  local log="${LOG_DIR}/${direction}_${F0}_seed${seed}.log"
  mkdir -p "${run_dir}" "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "SKIP ${direction} ${F0} seed=${seed}"
    return 0
  fi
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "missing same-direction C' ckpt: ${init_ckpt}" >&2
    return 1
  fi
  if [[ "${init_ckpt}" == *matched_seed0* ]]; then
    echo "refusing Day4 init: ${init_ckpt}" >&2
    return 1
  fi
  {
    echo "=== TRAIN F0_RX ${direction} seed=${seed} GPU=${gpu} init=${init_ckpt} ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/finetune.py \
      --manifest "${manifest}" \
      "${COMMON_MODEL[@]}" \
      --eval-split val \
      --paired-view oob_scale \
      --init-checkpoint "${init_ckpt}" \
      --seed "${seed}" \
      --epochs 80 \
      --lr 3e-3 \
      --loss-type ce \
      --checkpoint-metric acc \
      --weight-decay 5e-4 \
      --label-smoothing 0 \
      --out-dir "${run_dir}"
    echo "=== EVAL TEST F0_RX ${direction} seed=${seed} (target only) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate.py \
      --manifest "${manifest}" \
      "${COMMON_MODEL[@]}" \
      --eval-split test \
      --paired-view off \
      --seed "${seed}" \
      --mode classifier \
      --file-vote-mode mean_logits \
      --checkpoint "${run_dir}/best.pt" \
      --out-dir "${eval_dir}"
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${eval_dir}") / "metrics.json"
m = json.loads(p.read_text())
print(f"F0 ${direction} seed=${seed} win={100*m['window_acc']:.1f} file={100*m['file_acc']:.1f} files={m['num_files']} split={m.get('eval_split')} paired={m.get('paired_view')}")
if m.get("eval_split") != "test" or m["num_files"] != 24:
    raise SystemExit("F0 test/files check failed")
if m.get("paired_view") != "oob_scale":
    raise SystemExit(f"F0 ckpt paired_view must be oob_scale, got {m.get('paired_view')}")
if m.get("rx_style_eval") is True:
    raise SystemExit("target eval must not be synthetic rx-style")
if "matched_seed0" in str(m.get("checkpoint_path", "")):
    raise SystemExit("Day4 matched_seed0 ckpt leaked into RX")
PY
  } >"${log}" 2>&1
  echo "log ${log}"
}

JOBS=()
for direction in "${DIRECTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("${direction} ${seed}")
  done
done

run_wave() {
  local fn="$1"
  local i=0
  local n=${#JOBS[@]}
  while [[ ${i} -lt ${n} ]]; do
    local pids=()
    local status=0
    local j=0
    while [[ ${j} -lt ${#GPU_ARR[@]} && $((i + j)) -lt ${n} ]]; do
      local spec="${JOBS[$((i + j))]}"
      local direction="${spec%% *}"
      local seed="${spec##* }"
      local gpu="${GPU_ARR[${j}]}"
      "${fn}" "${gpu}" "${direction}" "${seed}" &
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
echo "directions=${DIRECTIONS[*]}"
echo "seeds=${SEEDS[*]}"
echo "day4_ckpt=FORBIDDEN"
echo "oracle=FORBIDDEN"
echo "f1=FORBIDDEN"
echo "cnn=FORBIDDEN"

echo "======== C'_RX train + target test ========"
run_wave run_cprime
echo "======== F0_RX train + target test ========"
run_wave run_f0

"${PY}" - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/real_rx_source_only/eval_test")
out_json = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/real_rx_source_only/rx_f0_vs_cprime.json")
out_md = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/real_rx_source_only/rx_f0_vs_cprime.md")
DIRS = ["rx1_to_rx2", "rx2_to_rx1"]
SEEDS = [0, 1]


def load(direction, model, seed):
    return json.loads((root / direction / model / f"seed_{seed}" / "metrics.json").read_text())


def pct(x):
    return round(100.0 * x, 1)


rows = []
dir_means = {}
for direction in DIRS:
    deltas = []
    for seed in SEEDS:
        c = load(direction, "C_full_ratio", seed)
        f0 = load(direction, "C_full_ratio_init_paired_scale", seed)
        if "matched_seed0" in str(c.get("checkpoint_path", "")) or "matched_seed0" in str(
            f0.get("checkpoint_path", "")
        ):
            raise SystemExit("Day4 ckpt leaked")
        dw = pct(f0["window_acc"]) - pct(c["window_acc"])
        df = pct(f0["file_acc"]) - pct(c["file_acc"])
        row = {
            "direction": direction,
            "seed": seed,
            "c_window": pct(c["window_acc"]),
            "f0_window": pct(f0["window_acc"]),
            "delta_window_pp": round(dw, 1),
            "c_file": pct(c["file_acc"]),
            "f0_file": pct(f0["file_acc"]),
            "delta_file_pp": round(df, 1),
            "n_files": c["num_files"],
        }
        rows.append(row)
        deltas.append(dw)
    dir_means[direction] = statistics.mean(deltas)

pooled = statistics.mean(dir_means.values())
both_pos = all(v > 0 for v in dir_means.values())
any_neg = any(v <= -2.0 for v in dir_means.values())
if any_neg or pooled < 4.0:
    verdict = "RX_FAIL"
elif both_pos and pooled >= 8.0 and max(dir_means.values()) >= 10.0:
    verdict = "RX_STRONG_GO"
elif both_pos and pooled >= 4.0:
    verdict = "RX_WEAK_GO"
else:
    verdict = "RX_FAIL"

payload = {
    "day4_ckpt_used": False,
    "oracle_used": False,
    "f1_opened": False,
    "cnn_opened": False,
    "rows": rows,
    "direction_mean_delta_window_pp": {k: round(v, 2) for k, v in dir_means.items()},
    "pooled_delta_window_pp": round(pooled, 2),
    "verdict": verdict,
    "gate": {
        "STRONG_GO": "both dirs F0>C', pooled>=8, max dir>=10",
        "WEAK_GO": "both dirs >0 and 4<=pooled<8",
        "FAIL": "pooled<4 or any dir mean <=-2",
    },
    "note": "Do not retune a/lr/epoch from this table. File Acc recorded only.",
}
out_json.write_text(json.dumps(payload, indent=2) + "\n")
lines = [
    "# Real RX source-only: F0 vs C' (window Acc primary)",
    "",
    f"verdict={verdict}  pooled_Δ_window={pooled:+.2f} pp",
    f"rx1_to_rx2 mean Δ={dir_means['rx1_to_rx2']:+.2f}  rx2_to_rx1 mean Δ={dir_means['rx2_to_rx1']:+.2f}",
    "day4_ckpt=false  oracle=false  f1=false  cnn=false",
    "",
    "| direction | seed | C' win | F0 win | Δ win | C' file | F0 file | Δ file |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['direction']} | {r['seed']} | {r['c_window']:.1f} | {r['f0_window']:.1f} | "
        f"{r['delta_window_pp']:+.1f} | {r['c_file']:.1f} | {r['f0_file']:.1f} | "
        f"{r['delta_file_pp']:+.1f} |"
    )
lines.extend(
    [
        "",
        "STRONG_GO: both dirs F0>C', pooled ≥8, at least one dir ≥10.",
        "WEAK_GO: both dirs >0 and 4≤pooled<8. FAIL: pooled<4 or any dir ≤−2.",
        "Do not retune from target Acc. Do not open F1/CNN from FAIL.",
        "",
    ]
)
out_md.write_text("\n".join(lines))
print(out_md.read_text())
print("wrote", out_json)
print("VERDICT", verdict)
PY

echo "RX C'+F0 finished. Day4 ckpts unused. Oracle unused."
