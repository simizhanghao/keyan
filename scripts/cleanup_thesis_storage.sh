#!/usr/bin/env bash
# 清理 llm4RF 中非主线实验产物，保留 data/raw 与毕设三章冻结结果。
# 用法: bash scripts/cleanup_thesis_storage.sh [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY="${1:-}"

rm_path() {
  if [[ -e "$1" ]]; then
    if [[ "$DRY" == "--dry-run" ]]; then
      du -sh "$1" 2>/dev/null || true
      echo "  [dry-run] would delete: $1"
    else
      echo "Deleting: $1"
      rm -rf "$1"
    fi
  fi
}

echo "=== llm4RF storage cleanup ==="
echo "Root: $ROOT"
du -sh "$ROOT" 2>/dev/null || true

# --- 1. 整目录删除：重复训练 run、日志、旧 paper pipeline ---
for d in runs logs; do
  rm_path "$ROOT/$d"
done

# --- 2. outputs/ 下除 paper_ready_v3 外全部删除 ---
if [[ -d "$ROOT/outputs" ]]; then
  for item in "$ROOT/outputs"/*; do
    base="$(basename "$item")"
    [[ "$base" == "paper_ready_v3" ]] && continue
    rm_path "$item"
  done
fi

# --- 3. paper_ready_v3 内精简：保留 final_*、报告、主线 checkpoint ---
PR3="$ROOT/outputs/paper_ready_v3"
if [[ -d "$PR3" ]]; then
  # 中间日志与 eval 缓存
  rm_path "$PR3/step1_phase7_clean/logs"
  rm_path "$PR3/step1_phase7_clean/outputs"
  rm_path "$PR3/step1_phase7_clean/statistics"
  rm_path "$PR3/step1b_chirp_fusion_ablation/logs"
  rm_path "$PR3/step1b_chirp_fusion_ablation/outputs"
  rm_path "$PR3/step1b_chirp_fusion_ablation/runs"
  rm_path "$PR3/phase5_clean_cross_receiver/logs"
  rm_path "$PR3/phase5_clean_cross_receiver/outputs"

  # 所有 last.pt（保留 best.pt）
  if [[ "$DRY" != "--dry-run" ]]; then
    find "$PR3" -name 'last.pt' -print -delete 2>/dev/null || true
  else
    find "$PR3" -name 'last.pt' -exec du -ch {} + 2>/dev/null | tail -1
    echo "  [dry-run] would delete all last.pt under paper_ready_v3"
  fi

  # 消融模型 checkpoint（结果已在 final_tables CSV）
  for model in B_linear_no_oob C_cnn_stem_chirp_no_oob D_concat_oob_plain H_gated_chirp_plain; do
    rm_path "$PR3/step1_phase7_clean/runs/$model"
  done

  # phase5 仅保留 rx1_to_rx2 seed_0 的 A/F best.pt，删其余 direction/seed
  if [[ -d "$PR3/phase5_clean_cross_receiver/runs" ]]; then
    for model_dir in "$PR3/phase5_clean_cross_receiver/runs"/*; do
      [[ -d "$model_dir" ]] || continue
      for dir_dir in "$model_dir"/*; do
        [[ -d "$dir_dir" ]] || continue
        dname="$(basename "$dir_dir")"
        if [[ "$dname" != "rx1_to_rx2" ]]; then
          rm_path "$dir_dir"
          continue
        fi
        for seed_dir in "$dir_dir"/*; do
          [[ -d "$seed_dir" ]] || continue
          sname="$(basename "$seed_dir")"
          [[ "$sname" == "seed_0" ]] && continue
          rm_path "$seed_dir"
        done
      done
    done
  fi

  # step1 A_cnn_iq 仅保留 seed_0
  if [[ -d "$PR3/step1_phase7_clean/runs/A_cnn_iq" ]]; then
    for seed_dir in "$PR3/step1_phase7_clean/runs/A_cnn_iq"/*; do
      [[ -d "$seed_dir" ]] || continue
      [[ "$(basename "$seed_dir")" == "seed_0" ]] && continue
      rm_path "$seed_dir"
    done
  fi
fi

# --- 4. experiments 非主线结果 ---
EXP="$ROOT/experiments"
rm_path "$EXP/cross_receiver_adaptation/results"

# Ch4 中间 sweep（冻结表在 paper2_main + docs/paper2_rcpa）
for sub in oob_eq_quick_20260626_1731 quick_20260626_1709 \
           tta_threshold_sweep_20260626 tta_negative_quick_20260626_1744 \
           tta_negative_quick_20260626_1743 tta_negative_quick_20260626_1742 \
           sota_style_baselines_20260626_1819 full_run.log; do
  rm_path "$EXP/cross_receiver_calibration/results/$sub"
done
# full run 中间 embedding/run 缓存（保留 CSV/图）
rm_path "$EXP/cross_receiver_calibration/results/full_20260626_1720/runs"
rm_path "$EXP/cross_receiver_calibration/results/full_20260626_1720/embeddings"

# Ch5 EM-CR debug 只留报告与 CSV
EM="$EXP/em_robustness_openset/results"
for sub in A_clean_only_ft B_em_aug_ce C_weak_cfo D_stopgrad_kl; do
  if [[ -d "$EM/emcr_debug_20260628/$sub" ]]; then
    rm_path "$EM/emcr_debug_20260628/$sub"
  fi
done
rm_path "$EM/emcr_debug_20260628/logs"
rm_path "$EM/emcr_smoke_20260628_1309/checkpoints"
rm_path "$EM/emcr_smoke_20260628_1307"
rm_path "$EM/emcr_smoke_20260628_1308"
rm_path "$EM/smoke_audit_20260628_1116"
rm_path "$EM/smoke_audit_20260628_1117"
rm_path "$EM/smoke_20260626_1912"
rm_path "$EM/openset_full_20260628"
rm_path "$EM/openset_full_20260628_1119"

echo ""
echo "=== Done ==="
du -sh "$ROOT" "$ROOT/data" "$ROOT/outputs" "$ROOT/experiments" 2>/dev/null || true
