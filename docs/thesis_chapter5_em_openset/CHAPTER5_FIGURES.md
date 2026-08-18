# Chapter 5 Figures

**风格：** IEEE / 毕业论文；PDF 矢量主图 + PNG 预览；无 smoke/internal 标签；纵轴 `File-level accuracy (%)` 或 AUROC/EER。

**绘图环境：** `/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python`

---

## 主图清单

| Figure | 文件 | 内容 | 图注要点 |
|--------|------|------|----------|
| Fig. 5-1 | `figures/fig5_1_em_robustness_curves.pdf` | Ours 六类扰动 + mixed 闭集退化曲线 | Day5 test，256 win/file，voting |
| Fig. 5-2 | `figures/fig5_2_openset_clean.pdf` | Clean open-set：AUROC/EER/known acc by scorer | 20 known + 4 unknown，3 seeds |
| Fig. 5-3 | `figures/fig5_3_em_stress_ranking.pdf` | 扰动族平均鲁棒 acc 与降幅排序 | CFO/AWGN 降幅最大 |
| Fig. 5-4a–c | `figures/fig5_cnn_vs_ours_{awgn,cfo,narrowband}.pdf` | CNN-IQ vs RF-HSTU 对比曲线 | NBI 差距最大；CFO 共同失效 |
| Fig. 5-5 | `figures/fig5_4_auroc_under_em.pdf` | Open-set AUROC under EM（Proto vs Maha） | CFO 0.003 骤降 |
| Fig. 5-6 | `figures/fig5_5_eer_under_em.pdf` | Open-set EER under EM | 附表/补充 |
| Fig. 5-7 | `figures/fig5_6_known_acc_under_em.pdf` | Known classification under EM | 与 AUROC 联合解读 |

PNG 预览：同目录同名 `.png`。

---

## 生成脚本

| 图 | 脚本 |
|----|------|
| Fig. 5-1–3 | `experiments/em_robustness_openset/plot_em_robustness_curves.py` |
| Fig. 5-4 | `experiments/em_robustness_openset/plot_cnn_vs_ours_em.py` |
| Fig. 5-5–7 | `experiments/em_robustness_openset/plot_openset_under_em.py` |

---

## 状态

| 文件 | 存在 |
|------|------|
| fig5_1_em_robustness_curves.pdf | ✅ |
| fig5_2_openset_clean.pdf | ✅ |
| fig5_3_em_stress_ranking.pdf | ✅ |
| fig5_cnn_vs_ours_*.pdf | ✅ (AWGN, CFO, Narrowband) |
| fig5_4–6_*_under_em.pdf | ✅ |

**不纳入主图：** EM-CR 训练曲线（仅 debug 附表/文字分析）。
