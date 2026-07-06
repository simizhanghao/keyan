#!/usr/bin/env bash
# Cross-receiver failure diagnosis pipeline (run before any new adaptation method).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
SEED=${SEED:-0}
PHASE5=${PHASE5:-outputs/paper_ready_v3/phase5_clean_cross_receiver}
VERIFY=${VERIFY:-experiments/cross_receiver_adaptation/results/verify_baseline_20260626}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT=${OUT:-experiments/cross_receiver_diagnosis/results/run_$(date +%Y%m%d)}
MANIFEST=${MANIFEST:-data/manifest_rx1_to_rx2.csv}

CNN_CKPT=${PHASE5}/runs/A_cnn_iq/rx1_to_rx2/seed_${SEED}/best.pt
OURS_CKPT=${PHASE5}/runs/F_cross_attn_chirp_plain/rx1_to_rx2/seed_${SEED}/best.pt

cd "${ROOT}"
mkdir -p "${OUT}"/{embeddings,distances,probes,figures,path_ablation,oob_spectrum,confusion}

echo "==> [1/8] Extract embeddings (CNN + Ours, RX1-trained)"
"${PY}" experiments/cross_receiver_diagnosis/extract_embeddings.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CNN_CKPT}" \
  --model-name cnn_iq \
  --train-direction rx1_to_rx2 \
  --seed "${SEED}" \
  --out-dir "${OUT}/embeddings/cnn_iq" \
  --device cuda

"${PY}" experiments/cross_receiver_diagnosis/extract_embeddings.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${OURS_CKPT}" \
  --model-name ours_cross_attn \
  --train-direction rx1_to_rx2 \
  --seed "${SEED}" \
  --out-dir "${OUT}/embeddings/ours_cross_attn" \
  --device cuda

echo "==> [2/8] Distance diagnostics (cosine, file-level centroids)"
for model in cnn_iq ours_cross_attn; do
  tag=$([ "${model}" = cnn_iq ] && echo cnn || echo ours)
  for path in fused main oob; do
    "${PY}" experiments/cross_receiver_diagnosis/analyze_distances.py \
      --emb-dir "${OUT}/embeddings/${model}" \
      --path "${path}" --level file --metric cosine \
      --out-csv "${OUT}/distances/${tag}_${path}_file.csv"
  done
done

echo "==> [3/8] Receiver / device probes"
for model in cnn_iq ours_cross_attn; do
  tag=$([ "${model}" = cnn_iq ] && echo cnn || echo ours)
  for path in fused main oob; do
    "${PY}" experiments/cross_receiver_diagnosis/train_probes.py \
      --emb-dir "${OUT}/embeddings/${model}" \
      --path "${path}" \
      --out-csv "${OUT}/probes/${tag}_${path}.csv"
  done
done

echo "==> [4/8] t-SNE visualization (Ours fused, file-level)"
"${PY}" experiments/cross_receiver_diagnosis/plot_embeddings.py \
  --emb-dir "${OUT}/embeddings/ours_cross_attn" \
  --path fused \
  --method tsne \
  --level file \
  --out-dir "${OUT}/figures"

echo "==> [5/8] Main / OOB / fused path ablation"
"${PY}" experiments/cross_receiver_diagnosis/analyze_path_ablation.py \
  --emb-dir "${OUT}/embeddings/ours_cross_attn" \
  --out-csv "${OUT}/path_ablation/ours_paths.csv"

echo "==> [6/8] OOB spectrum receiver profile"
"${PY}" experiments/cross_receiver_diagnosis/analyze_oob_spectrum.py \
  --out-dir "${OUT}/oob_spectrum"

echo "==> [7/8] Confusion matrices"
if [[ -d "${VERIFY}" ]]; then
  "${PY}" experiments/cross_receiver_diagnosis/plot_confusion.py \
    --pred-root "${VERIFY}" \
    --out-dir "${OUT}/confusion"
fi

echo "==> [8/8] Generate diagnosis report"
"${PY}" experiments/cross_receiver_diagnosis/generate_report.py \
  --results-dir "${OUT}" \
  --out-md experiments/cross_receiver_diagnosis/CROSS_RECEIVER_DIAGNOSIS_REPORT.md

echo "==> Diagnosis complete: ${OUT}"
echo "    Report: experiments/cross_receiver_diagnosis/CROSS_RECEIVER_DIAGNOSIS_REPORT.md"
