#!/usr/bin/env bash
# EM-CR smoke: 3 epochs, small subset, moderate perturbations only.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PYTHON:-/new_nfs/haiyu/anaconda3/bin/python}
GPU_ID=${GPU_ID:-5}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

OUT="experiments/em_robustness_openset/results/emcr_smoke_$(date +%Y%m%d_%H%M)"
mkdir -p "${OUT}/logs"
INIT_CKPT="outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
CLEAN_CKPT="${INIT_CKPT}"

echo "==> EM-CR smoke train"
"${PY}" experiments/em_robustness_openset/train_em_consistency.py \
  --manifest data/paper/cross_day_day1to5_source_only.csv \
  --init-checkpoint "${INIT_CKPT}" \
  --out-dir "${OUT}/checkpoints" \
  --mode em_cr \
  --lambda-kl 0.5 \
  --lambda-emb 0.0 \
  --epochs 3 \
  --max-files 8 \
  --samples-per-file 32 \
  --batch-size 16 \
  --lr 5e-4 \
  --device cuda \
  --num-workers 2 \
  2>&1 | tee "${OUT}/logs/train.log"

echo "==> Eval clean baseline"
"${PY}" experiments/em_robustness_openset/eval_em_consistency.py \
  --checkpoint "${CLEAN_CKPT}" --label clean \
  --samples-per-file 64 --batch-size 32 --device cuda \
  --out-csv "${OUT}/eval_clean.csv"

echo "==> Eval EM-CR checkpoint"
"${PY}" experiments/em_robustness_openset/eval_em_consistency.py \
  --checkpoint "${OUT}/checkpoints/best.pt" --label em_cr \
  --samples-per-file 64 --batch-size 32 --device cuda \
  --out-csv "${OUT}/eval_emcr.csv"

echo "==> Merge eval"
"${PY}" - <<'PY' "${OUT}"
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in [out / "eval_clean.csv", out / "eval_emcr.csv"]:
    with p.open() as f:
        rows.extend(csv.DictReader(f))
merged = out / "emcr_smoke_eval.csv"
with merged.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {merged}")
PY

echo "Smoke complete -> ${OUT}"
