# 第 3 章实验结果整理（IoTJ 主体 → 毕设创新点 1）

> 来源：`outputs/paper_ready_v3/`（分支 `paper-ready-v3` / `iotj-submit-final-v1`）
> 协议详见 `docs/experiment_protocol.md` 与 `PAPER_RESULTS_SUMMARY.md`

---

## 3.1 最终模型配置

| 项目 | 值 |
|------|-----|
| 模型 ID | `F_cross_attn_chirp_plain` |
| 架构 | CNN-stem + RF-HSTU + OOB cross-attention + chirp embedding |
| 核心机制 | **Cross-attentive OOB fusion**（非 concat / gated） |
| Chirp 角色 | 辅助 LoRa 结构先验，**非主要增益来源** |
| Manifest | `data/paper/cross_day_day1to5_source_only.csv` |
| Split | train=Day1–3, val=Day4, test=Day5 |
| OOB norm | zscore |
| Batch / LR / Epochs | 128 / 3e-3 / 80 |
| Seeds | 5 |
| 投票 | file-level mean_logits |

---

## 3.2 跨日期主结果（Table I 对应）

| Model | F-Acc (%) | F-F1 (%) | W-Acc (%) | W-F1 (%) |
|-------|-----------|----------|-----------|----------|
| CNN-IQ | 54.2±14.2 | 45.6±14.8 | 43.5±5.2 | 38.7±5.7 |
| RF-HSTU no OOB | 66.7±3.4 | 59.1±4.5 | 39.4±13.7 | 36.8±14.3 |
| **Ours (F)** | **75.0±5.3** | **67.9±6.8** | 41.5±2.4 | 39.4±2.1 |

**统计检验（F vs CNN）：**
- 平均增益：+20.8 pp（file-acc）
- Bootstrap 95% CI：[+9.2, +32.5] pp
- 5-seed 胜负：4 win / 1 tie / 0 loss

**Per-seed File-Acc（F）：** 83.3, 70.8, 70.8, 79.2, 70.8

原始报告：`outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md`
CSV：`outputs/paper_ready_v3/final_tables/table1_cross_day_main.csv`

---

## 3.3 Fusion / Chirp 消融（Table II 对应）

| Fusion | Chirp | F-Acc (%) | F-F1 (%) |
|--------|-------|-----------|----------|
| concat | × | 18.3±18.6 | 12.3±15.7 |
| concat | ✓ | 9.7±2.0 | 3.1±1.8 |
| cross-attn | × | 75.0±3.4 | 68.1±4.0 |
| cross-attn | ✓ | **75.0±5.3** | **67.9±6.8** |

**结论：** cross-attention 是稳定增益来源；concat 在跨日期下 collapse；chirp 不改变 cross-attn 均值。

CSV：`outputs/paper_ready_v3/final_tables/table2_fusion_chirp_ablation.csv`

---

## 3.4 部署偏移 LOCO（Table III 对应，附录级扩展）

| Shift | Held-out | CNN (%) | Ours (%) | Winner |
|-------|----------|---------|----------|--------|
| Config | Config1 (SF7) | 0.0 | 4.2 | Ours |
| Config | Config2 (SF8) | 4.2 | 8.3 | Ours |
| Config | Config3 (SF11) | 8.3 | 16.7 | Ours |
| Location | room | 29.2 | 54.2 | Ours |
| Location | office | 37.5 | 54.2 | Ours |
| Location | outdoor | 25.0 | 16.7 | CNN |
| Distance | 10 m | 4.2 | 20.8 | Ours |
| Distance | 15 m | 4.2 | 33.3 | Ours |
| Distance | 20 m | 8.3 | 37.5 | Ours |

**注意：** 部署实验使用 ratio OOB norm，与 Step1 cross-day 的 zscore 不同，论文中有 footnote。

---

## 3.5 跨接收机 Stress Test（Table IV，limitation 引用）

**协议：** strict source-only；val=源 RX；test=目标 RX；F 使用 ratio OOB norm；3 seeds

| Direction | CNN-IQ (%) | Ours (%) |
|-----------|------------|----------|
| RX1→RX2 | 4.2±0.0 | **18.1±3.9** |
| RX2→RX1 | 23.6±15.3 | 15.3±7.1 |

Chance level：4.17%（24 类）

**答辩说法：**
- RX1→RX2：Hybrid 显著优于 CNN（+14 pp 量级），但仍远低于 cross-day
- RX2→RX1：方向不对称，CNN 有时更好 → **不能声称 receiver-invariant**
- 这是第 4 章（跨接收机适配）的动机

原始报告：`outputs/paper_ready_v3/phase5_clean_cross_receiver/PHASE5_REPORT_FOR_GPT.md`

---

## 3.6 边缘部署（Table V）

| Model | Params | Latency @ bs=1 |
|-------|--------|----------------|
| CNN-IQ | 47.7K | ~1.12
 ms |
| Ours | 1.16M | ~2.5 ms |

Hybrid 可部署但更重；适合网关侧而非极端 MCU。

---

## 3.7 第 3 章写作要点

1. **问题定义：** same-receiver cross-day LoRa RFFI，24 类闭集识别
2. **方法贡献：** OOB 带外频谱作为硬件失真证据，cross-attention 选择性注入 RF-HSTU 序列
3. **实验边界：** 主结果只覆盖 cross-day；cross-receiver 作为 limitation 引出第 4 章
4. **不可用结果：** 旧 target-val 协议（`docs/cross_receiver_findings.md`）已废弃

---

## 3.8 引用路径速查

```text
outputs/paper_ready_v3/PAPER_RESULTS_SUMMARY.md
outputs/paper_ready_v3/final_tables/table{1..5}_*.csv
outputs/paper_ready_v3/final_figures/fig{1..5}_*
docs/iotj_paper/tables/table{1..5}_*.tex
docs/iotj_paper/sections/04_method.tex
docs/iotj_paper/sections/06_results.tex
```
