# Chapter 5 Final Summary — 第三创新点收口

**分支：** `thesis-em-openset`  
**状态：** 主实验冻结，EM-CR 降级，进入毕业论文写作阶段  
**日期：** 2026-06-28

---

## 1. 第三创新点定位

**复杂电磁扰动与未知设备场景下的 LoRa 射频指纹鲁棒认证方法。**

本章不主打「训练一个更鲁棒的新模型」，而是：

1. 构建可复现的多类型电磁扰动评估协议；
2. 量化闭集 RFFI 在不同物理扰动下的退化规律；
3. 将认证从闭集扩展到开放集未知设备拒识；
4. 证明 embedding-space scoring 优于 softmax confidence；
5. 分析 CFO 作为当前系统最强 failure mode；
6. EM-CR 作为 preliminary / future work，不作为主方法。

与前三章关系：

| 章节 | 核心问题 |
|------|----------|
| 第 3 章 | OOB RF-HSTU：同接收机 / 跨日鲁棒建模 |
| 第 4 章 | RCPA：跨接收机校准与失配诊断 |
| **第 5 章** | **复杂电磁环境 + 开放集：认证可靠性** |

---

## 2. 已完成实验

| 实验 | 目录 / 报告 | 状态 |
|------|-------------|------|
| Phase A.5 Smoke Audit | `smoke_audit_20260628_1118/` | ✅ 通过 |
| Closed-set EM robustness (Ours) | `em_full_20260628/` | ✅ |
| CNN-IQ EM baseline | `em_full_20260628_cnn/` | ✅ |
| Clean open-set (3 seeds) | `openset_full_20260628_1123/` | ✅ |
| Open-set under EM | `openset_under_em_20260628/` | ✅ |
| EM-CR smoke | `emcr_smoke_20260628_1309/` | ❌ 未通过 |
| EM-CR debug suite | `emcr_debug_20260628/` | ✅ 完成，不进入 full |
| Core-change audit | `CORE_CHANGE_AUDIT.md` | ✅ 默认行为不变 |

**协议要点：** Day5 test，256 windows/file，file-level mean-logits voting；open-set 20 known + 4 unknown，阈值在 val 选取。

---

## 3. 主结论（可直接写入本章小结）

1. **CFO 与强 AWGN** 是对当前 LoRa RFFI 破坏最强的扰动；CFO norm ≥ 0.003 时闭集准确率约 **4.2%**。
2. **窄带干扰** 在当前设置下相对温和（SIR 10–30 dB 仍约 83–87.5%）。
3. **IQ imbalance、filter drift** 造成中等退化；phase noise 在 σ ≥ 0.05 时显著下降。
4. **Ours vs CNN-IQ：** clean +20.8 pp；AWGN 30 dB +8.3 pp；NBI 10 dB **+58.3 pp**；CFO 0.003 两者均 ~4.2%（共同短板）。
5. **Open-set clean：** Prototype AUROC **0.917±0.059**、Mahalanobis **0.913±0.062**；MSP **0.425**、Energy **0.575** — embedding 距离明显优于 softmax 置信度。
6. **Open-set under EM：** AWGN 30 dB 仍较稳（Proto AUROC **0.896**）；CFO 0.003 同时摧毁 known acc（**3.3%**）与 AUROC（**0.492**）。
7. **EM-CR：** clean-only FT 不崩；EM-Aug / weak CFO / stopgrad KL **无稳定鲁棒增益**；原 smoke 崩溃来自强 CFO + 全主干 + 过长训练 → **future work**。

---

## 4. 主表与主图索引

- 表格：`CHAPTER5_TABLES.md`
- 图：`CHAPTER5_FIGURES.md`
- LaTeX 草稿：`CHAPTER5_METHOD_DRAFT.tex`
- 详细结果：`CHAPTER5_RESULTS.md`

---

## 5. EM-CR 论文表述（推荐）

> 我们进一步尝试了基于扰动一致性的鲁棒微调，但初步实验表明，在小样本和强 CFO 条件下直接进行一致性训练容易破坏已学习的射频指纹表征。保守实验（冻结主干、弱扰动、stop-gradient KL）能够避免灾难性遗忘，但未带来稳定鲁棒性增益。后续更适合采用冻结主干、课程式扰动增强、teacher-student 稳定训练或 receiver/statistics-level augmentation。

**禁止表述：** 「EM-CR 显著提升鲁棒性」「EM-CR 是第三创新点核心方法」。

---

## 6. 剩余风险与人工决策

| 风险 | 说明 | 建议 |
|------|------|------|
| Open-set seed0 AUROC=1.0 | 小样本 split 现象 | 正文强调 3-seed mean±std |
| EM-CR eval 64 win vs full 256 win | debug clean 79.2% vs 83.3% | 正文注明 debug 协议差异 |
| CFO 无方法差异 | 物理瓶颈 | 讨论中写为系统级短板，非模型排序问题 |

**无需再开实验即可写第 5 章初稿。**
