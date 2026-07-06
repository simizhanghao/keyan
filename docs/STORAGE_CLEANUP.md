# llm4RF 存储清理记录（2026-06-26）

> 原则：**保留原始数据 + 毕设三章冻结结果**；删除探索性 sweep、重复 run、中间 checkpoint。

## 清理前后

| 项 | 清理前 | 清理后 |
|----|--------|--------|
| 仓库总计 | ~70 GB | ~68 GB |
| `data/raw`（原始下载） | 67 GB | **67 GB（未动）** |
| `outputs/` | ~1.3 GB | ~17 MB |
| `runs/` | ~439 MB | **已删除** |
| `logs/` | ~56 MB | **已删除** |

**释放约 2 GB**（非原始数据部分）。若需进一步压缩，只能动 `data/raw` 外的 `.dat` 或 docs 内 PDF 文献副本。

---

## 保留内容（毕设主线）

### 原始数据（完整保留）

- `data/raw/` — OSU LoRa 原始 IQ（67 GB）
- `data/paper/`、`data/manifest_*.csv` — 实验清单

### 第 3 章（Paper 1 / IoTJ）

- `outputs/paper_ready_v3/final_figures/` — 5 张主图 PDF
- `outputs/paper_ready_v3/final_tables/` — 5 张主表 CSV
- `outputs/paper_ready_v3/PAPER_RESULTS_SUMMARY.md`
- `outputs/paper_ready_v3/step1_phase7_clean/` — 报告 + metrics
- **Checkpoint：**
  - `F_cross_attn_chirp_plain/seed_{0..4}/best.pt`（cross-day 5 seeds）
  - `A_cnn_iq/seed_0/best.pt`（CNN baseline，第 5 章 EM 用）

### 第 4 章（Paper 2 / RCPA-T）

- `outputs/paper_ready_v3/phase5_clean_cross_receiver/` — 报告
- **Checkpoint：** `F` / `A` 的 `rx1_to_rx2/seed_0/best.pt`
- `experiments/cross_receiver_diagnosis/results/run_20260626/` — 诊断 embedding/图
- `experiments/cross_receiver_calibration/results/paper2_main/` — 冻结主表
- `experiments/cross_receiver_calibration/results/full_20260626_1720/` — CSV + shot curve 图（已删中间 runs/embeddings）
- `docs/paper2_rcpa/` — Paper 2 LaTeX 与图

### 第 5 章（EM + Open-set）

- `experiments/em_robustness_openset/results/em_full_20260628/`
- `experiments/em_robustness_openset/results/em_full_20260628_cnn/`
- `experiments/em_robustness_openset/results/openset_full_20260628_1123/`
- `experiments/em_robustness_openset/results/openset_under_em_20260628/`
- `experiments/em_robustness_openset/results/emcr_debug_20260628/` — 仅 CSV/报告（checkpoint 已删）
- `experiments/em_robustness_openset/results/emcr_smoke_20260628_1309/` — 报告
- `docs/thesis_chapter5_em_openset/`

### 代码与文档

- `src/`、`scripts/`、`experiments/*/（脚本）`
- `docs/thesis/`、`docs/iotj_paper/`

---

## 已删除（非主线）

### 整目录

- `runs/` — 与 `outputs/` 重复的训练缓存
- `logs/` — 训练日志
- `outputs/paper_runs/` — 旧 paper pipeline（~694 MB）
- `outputs/` 下除 `paper_ready_v3` 外全部（sweep、supcon、lodo、tta、coral 等 ~40 个子目录）

### paper_ready_v3 内精简

- 所有 `last.pt`
- 消融模型 checkpoint：`B_`/`C_`/`D_`/`H_`（数值已在 `final_tables`）
- `step1b` 的 runs/logs/outputs
- phase5 非 `rx1_to_rx2/seed_0` 的 checkpoint

### experiments 中间结果

- `cross_receiver_adaptation/results/`
- Ch4 中间 sweep：`oob_eq_quick`、`quick_*`、`tta_*`、`sota_style_*`
- Ch4 full run 中间 `runs/`、`embeddings/`（保留 CSV 与 curve 图）
- Ch5 EM-CR debug 的 4 组 `.pt` checkpoint
- 重复 smoke / openset 试运行目录

---

## 复现脚本

```bash
# 再次执行相同清理（幂等）
bash scripts/cleanup_thesis_storage.sh

# 预览不删除
bash scripts/cleanup_thesis_storage.sh --dry-run
```

## 如需进一步省空间

1. **不要删** `data/raw/` — 原始 IQ 是重新训练的唯一来源
2. 可选删 `docs/papers/`（28 MB 文献 PDF 副本）和 `docs/paper_draft/`（17 MB）
3. 若确认不再重跑 Ch3 cross-day 多 seed，可只留 `F/seed_0/best.pt`（再省 ~10 MB）
