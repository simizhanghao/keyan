#!/usr/bin/env bash
# 打包毕设全量备份：写作仓 + 代码结果仓 + 原始 IQ 数据（可选）
# 用法:
#   bash scripts/backup_full_project.sh              # 三个包都打
#   bash scripts/backup_full_project.sh --no-raw     # 跳过 67GB 原始数据
#   bash scripts/backup_full_project.sh --raw-only   # 只打原始数据
set -euo pipefail

ROOT_HCC="/data1/hcc"
CODE_DIR="${ROOT_HCC}/llm4RF"
THESIS_DIR="${ROOT_HCC}/lora-rffi-thesis"
OUT_DIR="${ROOT_HCC}/backups"
STAMP="$(date +%Y%m%d)"
MODE="${1:---all}"

mkdir -p "$OUT_DIR"

pack_thesis() {
  local out="${OUT_DIR}/thesis_writing_${STAMP}.tar.gz"
  echo ">>> Packing thesis: $out"
  tar -czf "$out" \
    --exclude='.git' \
    --exclude='build/tectonic' \
    --exclude='build/*.aux' \
    --exclude='build/*.log' \
    --exclude='build/*.blg' \
    -C "$ROOT_HCC" lora-rffi-thesis
  du -sh "$out"
}

pack_code() {
  local out="${OUT_DIR}/code_results_${STAMP}.tar.gz"
  echo ">>> Packing code+results (no raw): $out"
  tar -czf "$out" \
    --exclude='.git' \
    --exclude='data/raw/osu_lora' \
    --exclude='data/raw/**/*.dat' \
    --exclude='data/raw/**/*.sigmf-meta' \
    --exclude='.playwright-browsers' \
    --exclude='__pycache__' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='runs' \
    --exclude='logs' \
    -C "$ROOT_HCC" llm4RF
  du -sh "$out"
}

pack_raw() {
  local raw="${CODE_DIR}/data/raw/osu_lora"
  if [[ ! -d "$raw" ]]; then
    echo "WARN: raw data not found at $raw, skip."
    return 0
  fi
  local out="${OUT_DIR}/raw_osu_lora_${STAMP}.tar.gz"
  echo ">>> Packing raw IQ (this may take 1-2 hours): $out"
  tar -czf "$out" -C "${CODE_DIR}/data/raw" osu_lora
  du -sh "$out"
}

case "$MODE" in
  --no-raw)
    pack_thesis
    pack_code
    ;;
  --raw-only)
    pack_raw
    ;;
  --all|"")
    pack_thesis
    pack_code
    pack_raw
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [--all|--no-raw|--raw-only]"
    exit 1
    ;;
esac

echo ""
echo "=== Backup complete ==="
ls -lh "$OUT_DIR"/*"${STAMP}"* 2>/dev/null || true
echo "Download with: rsync -avP USER@SERVER:${OUT_DIR}/ ./thesis_backup/"
