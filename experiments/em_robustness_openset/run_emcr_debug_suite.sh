#!/usr/bin/env bash
# EM-CR debug suite: 4 conservative 3-epoch experiments.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PYTHON:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-5}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="experiments/em_robustness_openset/results/emcr_debug_20260628"
mkdir -p "${OUT}/logs"
INIT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
COMMON=(
  --manifest data/paper/cross_day_day1to5_source_only.csv
  --init-checkpoint "${INIT}"
  --epochs 3
  --max-files 16
  --samples-per-file 32
  --batch-size 16
  --lr 1e-5
  --device cuda
  --num-workers 2
  --freeze-head-only
  --grad-clip 1.0
)

run_exp() {
  local name="$1"
  local mode="$2"
  local extra="${3:-}"
  local dir="${OUT}/${name}"
  mkdir -p "${dir}"
  echo "==> ${name} mode=${mode}"
  "${PY}" experiments/em_robustness_openset/train_em_consistency.py \
    "${COMMON[@]}" --out-dir "${dir}" --mode "${mode}" ${extra} \
    2>&1 | tee "${OUT}/logs/${name}.log"
  "${PY}" experiments/em_robustness_openset/eval_em_consistency.py \
    --checkpoint "${dir}/best.pt" --label "${name}" \
    --samples-per-file 64 --device cuda \
    --out-csv "${OUT}/eval_${name}.csv"
}

run_exp A_clean_only_ft clean_only
run_exp B_em_aug_ce em_aug_ce
run_exp C_weak_cfo weak_cfo
run_exp D_stopgrad_kl em_cr_stopgrad "--lambda-kl 0.1 --kl-temperature 2"

# merge summary
"${PY}" - <<'PY' "${OUT}"
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in sorted(out.glob("eval_*.csv")):
    with p.open() as f:
        for r in csv.DictReader(f):
            r["experiment"] = p.stem.replace("eval_", "")
            rows.append(r)
if rows:
    with (out / "debug_suite_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}/debug_suite_summary.csv")
PY

echo "Debug suite -> ${OUT}"
