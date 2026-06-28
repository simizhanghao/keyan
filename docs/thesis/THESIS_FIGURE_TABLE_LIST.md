# 毕业论文图表清单

格式：编号 / 标题 / 来源路径 / 建议章节 / 是否需重画 / 备注

---

## 第 1–2 章（示意图，需手绘或重画）

| 编号 | 标题 | 来源 | 章节 | 重画 | 备注 |
|------|------|------|------|------|------|
| Fig 1-1 | LoRa 物联网认证应用场景 | 待绘 | 1 | 是 | 网关—终端—干扰示意 |
| Fig 2-1 | LoRa CSS  chirp 原理示意 | 待绘 / IoTJ TikZ | 2 | 可选 | 可参考 `docs/iotj_paper/figures/fig1_architecture_tikz.tex` |
| Fig 2-2 | RFFI 闭集 vs 开放集认证流程 | 待绘 | 2 | 是 | known/unknown 划分示意 |
| Table 2-1 | OSU LoRa 数据集统计 | 文献 + 自建 manifest | 2 | 是 | 设备数、天数、接收机 |

---

## 第 3 章

| 编号 | 标题 | 来源路径 | 章节 | 重画 | 备注 |
|------|------|----------|------|------|------|
| Fig 3-1 | OOB-guided RF-HSTU 总体架构 | `outputs/paper_ready_v3/final_figures/fig1_model_architecture.pdf` | 3.2 | 否 | 毕设可加中文标注 |
| Fig 3-2 | 跨日期各 seed 文件级准确率 | `outputs/paper_ready_v3/final_figures/fig2_cross_day_seed_bars.pdf` | 3.4 | 否 | |
| Fig 3-3 | Fusion/Chirp 消融 | `outputs/paper_ready_v3/final_figures/fig3_fusion_chirp_ablation.pdf` | 3.5 | 否 | |
| Fig 3-4 | 部署距离偏移 | `outputs/paper_ready_v3/final_figures/fig4_distance_shift.pdf` | 3.6 | 否 | LOCO distance |
| Fig 3-5 | 跨接收机 stress（limitation） | `outputs/paper_ready_v3/final_figures/fig5_cross_receiver_stress.pdf` | 3.7 | 否 | 引出第 4 章 |
| Table 3-1 | 跨日期主结果 | `outputs/paper_ready_v3/final_tables/table1_cross_day_main.csv` | 3.4 | 是 | 转 Word 三线表 |
| Table 3-2 | Fusion/Chirp 消融 | `table2_fusion_chirp_ablation.csv` | 3.5 | 是 | |
| Table 3-3 | 部署偏移 LOCO | `table3_deployment_shift.csv` | 3.6 | 是 | |
| Table 3-4 | 边缘部署 profile | `table5_edge_profile.csv` | 3.6 或附录 | 是 | 可选 |

---

## 第 4 章

| 编号 | 标题 | 来源路径 | 章节 | 重画 | 备注 |
|------|------|----------|------|------|------|
| Fig 4-1 | 跨接收机诊断四联图 | `docs/paper2_rcpa/figures/fig1_diagnosis_summary.pdf` | 4.2–4.3 | 否 | OOB 谱偏置、probe、ratio、collapse |
| Fig 4-2 | RCPA-T K-shot 曲线 | `docs/paper2_rcpa/figures/fig2_rcpa_shotcurve.pdf` | 4.5 | 否 | |
| Table 4-1 | Source-only vs RCPA-S | `docs/paper2_rcpa/tables/table1_baseline.tex` | 4.4 | 是 | |
| Table 4-2 | RCPA-T 主结果 K=5/10/20 | `table2_rcpa_main.tex` 或 paper2_main CSV | 4.5 | 是 | |
| Table 4-3 | RCPA 消融 | `table3_ablation.tex` | 4.5 | 是 | |
| Table 4-4 | 同协议 K-shot baseline | `table5/6_*.tex` | 4.6 | 是 | Linear probe vs RCPA-T |
| Fig 4-3 | OOB 平均谱（可选） | `experiments/cross_receiver_diagnosis/results/.../rx_mean_oob_spectrum.png` | 4.2 | 否 | 补充材料 |

---

## 第 5 章

| 编号 | 标题 | 来源路径 | 章节 | 重画 | 备注 |
|------|------|----------|------|------|------|
| Fig 5-1 | EM 鲁棒性退化曲线（Ours） | `docs/thesis_chapter5_em_openset/figures/fig5_1_em_robustness_curves.pdf` | 5.3 | 否 | 六类扰动 + mixed |
| Fig 5-2 | CNN-IQ vs Ours（AWGN） | `figures/fig5_cnn_vs_ours_awgn.pdf` | 5.4 | 否 | |
| Fig 5-3 | CNN-IQ vs Ours（CFO） | `figures/fig5_cnn_vs_ours_cfo.pdf` | 5.4 | 否 | 共同失效 |
| Fig 5-4 | CNN-IQ vs Ours（Narrowband） | `figures/fig5_cnn_vs_ours_narrowband.pdf` | 5.4 | 否 | 最大差距 |
| Fig 5-5 | Open-set clean 各 scorer | `figures/fig5_2_openset_clean.pdf` | 5.5 | 否 | |
| Fig 5-6 | 扰动敏感性排序 | `figures/fig5_3_em_stress_ranking.pdf` | 5.3 | 否 | |
| Fig 5-7 | Open-set AUROC under EM | `figures/fig5_4_auroc_under_em.pdf` | 5.6 | 否 | |
| Fig 5-8 | Open-set EER under EM | `figures/fig5_5_eer_under_em.pdf` | 5.6 | 可选 | |
| Fig 5-9 | Known acc under EM | `figures/fig5_6_known_acc_under_em.pdf` | 5.6 | 否 | |
| Table 5-1 | EM perturbation 协议 | `CHAPTER5_TABLES.md` Table 5-1 | 5.2 | 是 | |
| Table 5-2 | Closed-set robustness summary | `em_full_20260628/em_robustness_summary.csv` | 5.3 | 是 | |
| Table 5-3 | CNN vs Ours under EM | `CHAPTER5_TABLES.md` Table 5-3 | 5.4 | 是 | |
| Table 5-4 | Open-set clean | `openset_clean_summary.csv` | 5.5 | 是 | |
| Table 5-5 | Open-set under EM | `openset_under_em_summary.csv` | 5.6 | 是 | |
| Table 5-6 | EM-CR debug | `emcr_debug_20260628/debug_suite_summary.csv` | 5.7 | 是 | negative，附表 |

---

## 缺失 / 待人工补充

| 项 | 说明 |
|----|------|
| 第 1 章应用场景图 | 需手绘 |
| 第 2 章数据集统计表 | 需从 manifest 统计设备/文件数 |
| 全文统一图编号 | 合并三章后重新编号 |
| 中文图注 | PDF 图为英文轴标签，毕设可重画或双语 |

**大图复制：** 毕设排版时从 `outputs/paper_ready_v3/final_figures/` 与 `docs/thesis_chapter5_em_openset/figures/` 直接嵌入矢量 PDF。
