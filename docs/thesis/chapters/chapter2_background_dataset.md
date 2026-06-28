# 第 2 章 相关理论与数据集

## 2.1 LoRa 物理层与 CSS 基本原理

LoRa 采用 chirp spread spectrum：发射端对带宽 \(B\) 内线性调频 chirp 进行调制，通过改变 chirp 起始频率编码符号。接收端通过 de-chirp 与 FFT 完成解扩，具有较好的抗噪与扩频增益。CSS 结构使信号在时域与频域均呈现可辨识的 chirp 形态，为「chirp-aware」深度模型提供结构先验。

本文实验使用 Sub-GHz LoRa 采集数据，采样率与窗口长度与 OSU 数据集设定一致；每个样本为固定长度的复基带 IQ 片段，可同步提取带外（out-of-band, OOB）功率谱作为辅助输入。

## 2.2 射频指纹的物理来源

设备间可区分性主要来自模拟前端非理想性：

| 来源 | 机理简述 | 对观测的影响 |
|------|----------|--------------|
| PA 非线性 | 功放饱和与 AM-AM/AM-PM | 谐波与带外再生 |
| I/Q 不平衡 | 正交调制器增益/相位误差 | 镜像频率与谱不对称 |
| CFO | 本振频率偏差 | IQ 相位旋转、chirp 偏移 |
| 相位噪声 | 本振随机抖动 | 谱线展宽、短时相位扩散 |
| 滤波器响应 | 带宽与滚降差异 | 带边形状与群延迟 |

接收机侧同样引入频率响应、噪声与 ADC 非线性，在跨接收机场景中与设备指纹**耦合**，第 4 章将专门分析 OOB 路径上的接收机纠缠。

## 2.3 闭集识别与认证的区别

**闭集识别（closed-set identification）** 假设测试样本标签空间与训练集相同，评价指标多为准确率、Macro-F1 等。本文在文件级采用 mean-logits voting：同一文件内多窗口 logits 平均后 argmax。

**认证（authentication）** 在开放集设定下还需判决「是否属于已知登记设备」。已知类样本应被接受（低 FRR），未知类应被拒绝（低 FAR）。仅优化闭集 softmax 不足以保证开放集性能，第 5 章比较多种 scoring 函数。

## 2.4 开放集认证常用指标

设得分函数 \(s(x)\) 越大表示更倾向已知类（或越小表示未知，需统一方向）：

- **AUROC**：以已知为正类，得分排序下的 ROC 曲线下面积，综合反映可分性；
- **EER**：FAR 与 FRR 相等时的错误率；
- **FAR / FRR**：在固定阈值下的假接受率与假拒绝率；
- **FPR@95TPR**：在 TPR=95% 时的 FPR，衡量高检出率下的误报水平。

阈值应在**验证集**选定，不得在测试集上调参。本文 open-set 采用 20 known + 4 unknown 设备划分，报告 3 seeds 的均值与标准差。

## 2.5 OSU LoRa 数据集

OSU LoRa 数据集包含多台 LoRa 发射机在不同日期、配置、位置与接收机下的采集记录。每条记录含复 IQ 及元数据（设备 ID、日期、接收机、扩频因子等）。本文主要使用：

- **Cross-day split：** 训练 Day1–3，验证 Day4，测试 Day5；
- **Cross-receiver：** 源接收机 RX1/RX2 训练，目标接收机测试；
- **EM perturbation：** 在测试阶段对 IQ 施加可控合成扰动；
- **Open-set：** 从设备集合中 hold-out 部分 ID 作为 unknown。

具体设备数与文件数以实验 manifest 为准；论文中应附统计表（见 `THESIS_FIGURE_TABLE_LIST.md` Table 2-1，待从 manifest 汇总）。

## 2.6 实验协议概要

### 2.6.1 Cross-day（第 3 章）

- Manifest：`data/paper/cross_day_day1to5_source_only.csv`
- OOB 归一化：zscore（主实验）
- 训练：batch 128，lr 3e-3，80 epochs，5 seeds
- 评估：Day5 test，file-level mean-logits

### 2.6.2 Cross-receiver 与 RCPA-T（第 4 章）

- **Block-disjoint：** support 与 query 窗口来自不重叠的数据块，防止少样本泄漏；
- RCPA-T：冻结主干与源分类器，用 K 个 labeled target windows/device 估计目标原型；
- 对比：source-only、RCPA-S、linear probe、head fine-tuning 等同协议 baseline。

### 2.6.3 EM 与 open-set（第 5 章）

- 闭集 EM：Day5，256 windows/file，六类单扰动 + mixed stress；
- Open-set clean：20 known + 4 unknown，Prototype / Mahalanobis / MSP / Energy；
- Open-set under EM：在 10 种应力条件下重复开放集评估；
- EM-CR：仅 debug 规模，不进入 full training 结论。

## 2.7 本章涉及的主要符号

| 符号 | 含义 |
|------|------|
| \(x_{\mathrm{IQ}}\) | 带内 IQ 窗口 |
| \(x_{\mathrm{OOB}}\) | 带外谱特征 |
| \(K\) | 每设备目标域 labeled 窗口数 |
| CFO norm | 归一化载波频偏强度 |

## 本章小结

本章介绍了 LoRa CSS 与射频指纹物理来源，区分了闭集识别与开放集认证及相应指标，概述了 OSU LoRa 数据集与本文采用的 cross-day、cross-receiver、EM 与 open-set 协议。后续三章分别在上述协议下展开方法与实验，协议边界在各章开头将再次说明以避免混用结论。
