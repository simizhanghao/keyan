# 第 5 章 复杂电磁扰动与开放集未知设备认证

> 对应实验：commit `f9dbe1c`，分支 `thesis-em-openset`  
> 详细表格：`docs/thesis_chapter5_em_openset/CHAPTER5_TABLES.md`  
> 图表：`docs/thesis_chapter5_em_openset/figures/`  
> **第三创新点定位：** 评估协议 + 开放集认证 + embedding scorer + 失效分析；**非** EM-CR 新模型成功

## 5.1 问题定义

真实 LoRa 部署除跨日与跨接收机外，还面临：

1. **复杂电磁扰动：** AWGN、CFO、窄带干扰（NBI）、相位噪声、I/Q imbalance、滤波器漂移及混合应力；
2. **开放集身份：** 网关需拒绝未登记设备，而非仅在已知类中 argmax。

本章在**第 3 章训练好的 Ours 模型**（same-receiver cross-day，无 EM 训练）上，构建可复现评估协议，回答：

- 各类扰动下闭集识别如何退化？
- OOB RF-HSTU 相对 CNN-IQ 是否在扰动下仍占优？
- 开放集 unknown 拒识应如何评分？
- 扰动是否同时破坏 known 分类与 unknown 拒识？

**协议要点：** Day5 test，256 windows/file，file-level mean-logits；open-set 为 20 known + 4 unknown 设备，阈值在 val 选取；报告 3 seeds mean±std。

## 5.2 复杂电磁扰动建模

在测试阶段对 IQ 施加合成扰动（训练集不变），主要包括：

| 类型 | 参数示例 | 说明 |
|------|----------|------|
| AWGN | SNR 30–50 dB | 加性高斯白噪声 |
| CFO | norm 0.001–0.005 | 载波频偏引起的相位旋转 |
| Narrowband | SIR 10–30 dB | 带内窄带干扰 |
| Phase noise | σ 0.01–0.1 | 本振相位噪声 |
| IQ imbalance | gain/phase 偏差 | 镜像与谱不对称 |
| Filter drift | 滚降变化 | 接收滤波器漂移 |
| Mixed stress | 多扰动组合 | 复合场景 |

扰动强度分档覆盖「轻度—极端」，便于绘制退化曲线（图 5-1）。实现细节见 `experiments/em_robustness_openset/`。

## 5.3 闭集识别下的电磁鲁棒性评估

### 5.3.1 Ours 主结果（seed0，表 5-2）

| 条件 | File-Acc (%) |
|------|--------------|
| Clean | **83.3** |
| AWGN 30 dB | 70.8 |
| AWGN 20 dB | 显著下降 |
| CFO norm ≥ 0.003 | **~4.2** |
| NBI SIR 10 dB | **87.5** |
| Phase noise（大 σ） | 中等下降 |
| IQ imbalance / filter drift | 中等退化 |
| Mixed stress | 接近最差单扰动 |

### 5.3.2 扰动敏感性排序（图 5-6）

**CFO 与强 AWGN** 是对当前系统破坏最强的扰动；**窄带干扰** 在 SIR 10–30 dB 下相对温和（仍约 83–87.5%）。

### 5.3.3 核心结论（闭集）

1. Clean 83.3% 与第 3 章单 seed 峰值一致，说明评估管线与 ckpt 一致（`CORE_CHANGE_AUDIT.md`：默认推理仍 83.33%）；
2. CFO norm ≥ 0.003 时准确率约 **4.2%**，为系统级瓶颈；
3. NBI 10 dB 仍 **87.5%**，相对温和。

## 5.4 CNN-IQ 与 OOB RF-HSTU 对比

表 5-3、图 5-2–5-4。

| 条件 | CNN-IQ (%) | Ours (%) | Δ (pp) |
|------|------------|----------|--------|
| Clean | 62.5 | 83.3 | **+20.8** |
| AWGN 30 dB | 62.5 | 70.8 | +8.3 |
| CFO 0.003 | 4.2 | 4.2 | 0 |
| NBI 10 dB | 29.2 | 87.5 | **+58.3** |

**分析：**

- Ours 在 clean、AWGN 30 dB、NBI 10 dB 下显著优于 CNN-IQ，最大差距在窄带干扰，体现 OOB 路径对部分带内干扰的间接鲁棒性；
- CFO 0.003 下两者均 **~4.2%**，无模型优势 → **物理层结构性失效**，非架构排序问题。

## 5.5 开放集未知设备认证

### 5.5.1 设定

- 20 类 known 设备参与闭集训练；4 类 unknown 设备仅出现在测试；
- Scorer：Prototype distance、Mahalanobis、MSP、Energy；
- 指标：AUROC、EER、FAR、FRR、FPR@95TPR。

### 5.5.2 Clean 结果（3 seeds，表 5-4，图 5-5）

| Scorer | AUROC | EER |
|--------|-------|-----|
| **Prototype** | **0.917±0.059** | **0.0** |
| Mahalanobis | 0.913±0.062 | — |
| Energy | 0.575 | — |
| MSP | 0.425 | — |

**结论：** 嵌入空间 Prototype / Mahalanobis **明显优于** softmax 派生的 MSP / Energy。认证场景应优先采用几何距离评分，而非仅依赖 closed-set 训练后的置信度。

**说明：** 个别 seed AUROC=1.0 与小样本 split 有关，正文以 3-seed 均值为准。

## 5.6 EM stress 下的 open-set 认证

表 5-5、图 5-7–5-9（`openset_under_em_20260628/`）。

Prototype scorer，3 seeds 均值：

| 条件 | AUROC | Known Acc (%) |
|------|-------|---------------|
| Clean | 0.917 | 高 |
| AWGN 30 dB | **0.896** | 仍可用 |
| NBI 10 dB | **0.908** | 高 |
| CFO 0.003 | **0.492** | **3.3** |

**核心结论：**

1. AWGN 30 dB、NBI 10 dB 下 open-set 仍较稳；
2. **CFO 0.003 同时摧毁 known 分类（3.3%）与 unknown 拒识（AUROC 0.492）** — 当前系统最强 failure mode；
3. 开放集问题不能脱离闭集几何：CFO 破坏嵌入结构后，任何阈值均难以兼顾 FAR/FRR。

## 5.7 EM-CR 扰动一致性训练初步分析

EM-CR（EM Consistency Regularization）尝试在 clean 与扰动视图间施加一致性损失，属**初步鲁棒训练探索**，**不作为本章主方法**。

### 5.7.1 Smoke 失败

强 CFO + 全主干微调 + 过长训练：clean **83.3% → 20.8%** 灾难性遗忘。

### 5.7.2 Debug suite（表 5-6，`emcr_debug_20260628/`）

保守设置（冻结主干、head-only、3 epoch）：

| 变体 | Clean (%) | AWGN 30 dB (%) |
|------|-----------|----------------|
| A clean-only FT | ~79.2 | — |
| B EM-Aug | ~79.2 | ~62.5 |
| C weak CFO | — | 70.8 |
| D stopgrad KL | ~79.2 | ~62.5 |

**结论：**

- 保守微调可不崩溃，但 **无稳定鲁棒增益**；
- 简单扰动一致性在小样本与强 CFO 下易导致 clean 退化；
- 后续更适合：课程式扰动、冻结主干、teacher-student、统计风格增强等。

**禁止表述：** 「EM-CR 显著提升鲁棒性」「EM-CR 是第三创新点核心」。

### 5.7.3 协议说明

Debug 使用 64 windows/file，clean 79.2% vs full eval 83.3%，正文需注明协议差异。

## 5.8 本章小结

本章构建 LoRa RFFI **复杂电磁扰动 benchmark** 与 **开放集未知设备认证协议**，系统评估第 3 章 Ours 模型在多种应力下的可靠性边界：

1. **CFO 与强 AWGN** 破坏最强；**NBI** 相对温和；
2. Ours 在 clean、AWGN 30 dB、NBI 10 dB 下优于 CNN-IQ；CFO 0.003 下共同失效；
3. **Prototype / Mahalanobis** 在 open-set 上优于 MSP / Energy；
4. **CFO 同时破坏 known 分类与 unknown 拒识**；
5. **EM-CR** 初步实验未达主方法标准，作 future work，不影响第三创新点（评估协议 + 开放集 + 失效分析）的成立。

全文三条主线至此闭合：第 3 章建模 → 第 4 章跨 RX 校准 → 第 5 章复杂环境与开放集边界。
