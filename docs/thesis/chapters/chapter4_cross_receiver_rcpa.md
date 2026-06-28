# 第 4 章 跨接收机失配诊断与 RCPA-T 校准

> 对应成果：Paper 2  
> 实验数据：`experiments/cross_receiver_calibration/results/paper2_main/`、`docs/paper2_rcpa/`  
> 本章协议为 block-disjoint cross-receiver，与第 3 章 cross-day、第 5 章 EM 不同

## 4.1 跨接收机问题定义

部署新网关或更换接收机后，观测信号分布因接收端射频前端差异而发生系统性偏移。本章在 **cross-receiver** 设定下研究：源接收机 \(R_s\) 上训练，目标接收机 \(R_t\) 上测试。若仅使用源域分类器（source-only），闭集准确率约 **19–21%**，接近 24 类随机水平（4.17%）。

目标：在**少量目标域标签**可用时，校准分类决策使目标接收机上识别性能可实用化，同时诚实说明方法假设（非 source-free、非新 backbone）。

## 4.2 接收机诱导的 OOB 特征纠缠

### 4.2.1 现象描述

第 3 章表明 OOB 谱对硬件指纹敏感；但在跨接收机时，OOB 同时承载**接收机频率响应与噪声底**信息。设备特征与接收机特征在嵌入空间**纠缠**（entanglement），源域学到的类几何在目标域失效。

### 4.2.2 诊断证据

图 4-1（`fig1_diagnosis_summary.pdf`）汇总四类证据：

1. **OOB spectral bias：** RX1 与 RX2 平均带外谱存在系统差异；
2. **Receiver probe：** 仅用 OOB 嵌入训练接收机二分类，准确率 **72.7%** → OOB 含强 RX 信息；
3. **Embedding distance ratio：** 跨 RX 时 fused 嵌入类内/类间距离比 collapse（如 **0.22**），几何不可分；
4. **Prediction collapse：** top-1 预测质量集中（约 **95.8%**），类间决策边界失效。

对比：CNN-IQ 跨 RX ratio 约 1.25，形态不同，但 source-only 仍 near chance。

**结论：** cross-receiver 失败不仅是「域偏移」泛泛描述，而是 OOB 路径上**接收机诱导特征纠缠**可通过探针与几何指标具体验证。

## 4.3 诊断实验设置

- 嵌入提取：冻结第 3 章训练好的 Ours 主干；
- 探针实验：线性分类器区分 RX1/RX2，分别用 IQ-only、OOB-only、fused 嵌入；
- 统计：类内/类间欧氏距离比、预测熵与 top-1 mass。

详细数值见 `experiments/cross_receiver_diagnosis/`。

## 4.4 RCPA-T 方法

### 4.4.1 符号与组件

- **Frozen backbone：** 与第 3 章一致的 OOB RF-HSTU，参数冻结；
- **Source classifier / source prototype：** 源域训练得到的分类头与每类源原型；
- **Target-receiver prototype：** 在目标接收机上，用每设备 K 个 labeled windows 估计类中心 \(\mu_c^{(t)}\)；
- **推理：** 目标域样本嵌入与 \(\mu_c^{(t)}\) 最近原型分类。

### 4.4.2 RCPA 变体

- **RCPA-S：** 仅源原型，无目标校准；
- **RCPA-T：** 目标原型（本章主方法）；
- **RCPA-B：** 双向或其他消融变体（见表 4-3）。

### 4.4.3 Block-disjoint support / query split

Support 集用于估计目标原型或训练 probe；query 集用于测试。二者在数据块级别**不重叠**，避免少样本场景下的信息泄漏。该协议较严格，有利于辩护 K-shot 结论的可信度。

## 4.5 主实验结果

### 4.5.1 Source-only 与 RCPA-S

Source-only / RCPA-S 约 **15–21%**，与 chance 接近，确认无目标校准时跨 RX 不可用。

### 4.5.2 RCPA-T K-shot 曲线（图 4-2）

Pooled 3 seeds，block-disjoint：

| K（windows/device） | File-Acc (%) |
|---------------------|--------------|
| 5 | **58.3** |
| 10 | **69.4** |
| 20 | **75.0** |

K=20 与第 3 章 cross-day 75.0% 同量级，说明在目标域少量标签下可恢复大部分同接收机性能。**注意：** 这是在已知 K 个目标域标签的前提下，非无监督迁移。

### 4.5.3 消融（表 4-3）

K=5 时 RCPA-T **57.4%** vs RCPA-S **15.3%**，目标原型不可或缺。

## 4.6 同协议 Baseline 对比

在相同 block-disjoint、相同 K 下对比：

| 方法 | K=5 (%) | K=10 (%) |
|------|---------|----------|
| Linear probe | **59.0** | — |
| Head fine-tuning | — | 见 paper2_main |
| **RCPA-T** | **58.3** | **69.4** |

K=5 时 linear probe 略高于 RCPA-T（约 +0.7 pp），在 seed 方差内。RCPA-T 优势在于无需训练新头、原型可解释、与开放集距离度量自然衔接。无标签对齐方法（Mean-shift、CORAL + cls）约 **20–25%**，无效。

## 4.7 方法局限

1. **需要 labeled target windows：** 每设备至少 K 个带标签窗口，非 source-free；
2. **非新 backbone：** 表征来自第 3 章预训练，本章侧重校准策略；
3. **接收机数量有限：** 实验仅 RX1/RX2，外推需谨慎；
4. **块级 disjoint 协议：** 比随机窗口划分更严，数值可能低于宽松协议；
5. **未解决 EM 应力与 open-set：** 校准后模型在 CFO 等扰动下仍会失效（第 5 章）。

## 4.8 本章小结

本章针对 cross-receiver 闭集识别失败，给出可验证的诊断：OOB 路径存在 receiver-induced feature entanglement。提出 RCPA-T，在冻结主干下用 K 个目标域 labeled windows 构造目标接收机原型，block-disjoint 协议下 K=5/10/20 分别达到 58.3%、69.4%、75.0%，显著优于 source-only。同时诚实报告：需目标域标签、K=5 时与 linear probe 接近、仅两接收机验证。

跨接收机校准后，系统仍面临现场复杂电磁扰动与未知设备入网——这构成第 5 章的研究动机。
