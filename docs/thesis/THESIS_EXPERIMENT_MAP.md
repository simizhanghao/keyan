# 实验—章节映射表

| 章节 | 实验名称 | 数据集 / Split | 模型 | 主要指标 | 主要结论 | 数据路径 / Commit |
|------|----------|----------------|------|----------|----------|-------------------|
| 3 | Cross-day 主实验 | OSU LoRa Day1–3 train, Day4 val, Day5 test | CNN-IQ / RF-HSTU / Ours | File-Acc, Macro-F1 | Ours **75.0±5.3%** vs CNN **54.2±14.2%** (+20.8 pp) | `outputs/paper_ready_v3/final_tables/table1_*.csv` |
| 3 | Fusion/Chirp 消融 | 同上 | concat vs cross-attn | File-Acc | cross-attn ~75%；concat collapse 9–18% | `table2_fusion_chirp_ablation.csv` |
| 3 | 部署偏移 LOCO | leave-one-config/location/distance | CNN vs Ours | File-Acc | 距离 10–20 m Ours 20.8–37.5% | `table3_deployment_shift.csv` |
| 3 | Cross-RX stress (limitation) | RX1↔RX2 strict source-only | CNN vs Ours | File-Acc | RX1→RX2: 18.1% vs 4.2%；仍接近 chance | `table4_cross_receiver_stress.csv` |
| 3 | 边缘部署 profiling | — | CNN vs Hybrid | Params, Latency | 1.16M / 2.45 ms vs 47.7K / 1.12 ms | `table5_edge_profile.csv` |
| 4 | OOB 诊断 | RX1/RX2 mixed | Ours | Probe acc, energy ratio | RX probe **72.7%**；ratio **1.44** | `experiments/cross_receiver_diagnosis/` |
| 4 | Embedding 几何诊断 | RX1→RX2 | CNN vs Ours | Distance ratio, top-1 mass | CNN ratio 1.25；Ours fused 0.22；collapse 95.8% | 同上 + `fig1_diagnosis_summary.pdf` |
| 4 | Source-only baseline | block-disjoint | Source classifier / RCPA-S | File-Acc | ~19–21%，near chance | `paper2_main/` |
| 4 | RCPA-T shot curve | pooled 3 seeds | RCPA-T K=1–20 | File-Acc | K=5 **58.3%**；K=10 **69.4%**；K=20 **75.0%** | `paper2_main/rcpa_shotcurve.csv` |
| 4 | RCPA 消融 K=5 | RX1→RX2 | RCPA-S/T/B | File-Acc | RCPA-T **57.4%** vs S 15.3% | `paper2_main/ablation/` |
| 4 | 同协议 K-shot baseline | pooled | Linear probe / Head FT / RCPA-T | File-Acc K=5/10 | K=5: LP 59.0% vs RCPA-T 58.3% | `paper2_main/baselines/` |
| 4 | 无标签对齐 baseline | pooled | Mean-shift / CORAL + cls | File-Acc | ~20–25%，无效 | `paper2_main/` |
| 5 | Closed-set EM robustness | Day5 test, 256 win/file | Ours seed0 | File-Acc | Clean **83.3%**；CFO≥0.003 **4.2%** | `em_full_20260628/` |
| 5 | CNN-IQ EM baseline | 同上 | CNN-IQ seed0 | File-Acc | Clean 62.5%；NBI 10 dB 29.2% | `em_full_20260628_cnn/` |
| 5 | CNN vs Ours EM | 同上 | 两者对比 | Gain | NBI +58.3 pp；CFO 共同失效 | 两目录 CSV |
| 5 | Open-set clean | 20 known + 4 unknown, 3 seeds | Ours | AUROC, EER | Proto **0.917±0.059**；MSP 0.425 | `openset_full_20260628_1123/` |
| 5 | Open-set under EM | 10 条件 × 3 seeds | Ours | AUROC, known acc | CFO 0.003 AUROC **0.492**，known **3.3%** | `openset_under_em_20260628/` |
| 5 | EM-CR smoke | small subset | EM-CR | File-Acc | Clean 83.3→**20.8%** 崩溃 | `emcr_smoke_20260628_1309/` |
| 5 | EM-CR debug suite | head-only 3 epoch | A/B/C/D | File-Acc | 无鲁棒增益；不 full train | `emcr_debug_20260628/` |
| 5 | Core-change audit | Day5 clean | Ours ckpt | File-Acc | 默认推理仍 **83.33%** | `CORE_CHANGE_AUDIT.md` |

**代码基线 commit：** `f9dbe1c`（分支 `thesis-em-openset`）

**不再扩展的实验：** EM-CR full training；新 backbone；Paper 1/2 协议外对比。
