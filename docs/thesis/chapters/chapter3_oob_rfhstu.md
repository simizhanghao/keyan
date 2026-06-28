# 第 3 章 带外频谱引导的 RF-HSTU 鲁棒建模

> 对应成果：Paper 1 / IoTJ 投稿主体  
> 实验数据：`outputs/paper_ready_v3/`  
> 本章不涉及第 4 章 RCPA 与第 5 章 EM/open-set 协议

## 3.1 问题定义

本章关注 **same-receiver cross-day** 闭集设备识别：训练与测试使用同一接收机，但测试日（Day5）与训练日（Day1–3）不同。目标是在时间与环境漂移存在时，仍能稳定区分多台 LoRa 发射机。该设定排除了换接收机带来的域偏移，聚焦「建模层」能否利用带内与带外互补信息提升鲁棒性。

评价指标：文件级准确率（File-Acc）、文件级 Macro-F1（File-F1），以及窗口级指标作参考。Chance level 约为 4.17%（24 类）。

## 3.2 OOB-guided RF-HSTU 模型结构

最终模型记为 **F_cross_attn_chirp_plain**（下文称 Ours），整体为双分支混合架构（见图 3-1，来源 `fig1_model_architecture.pdf`）。

### 3.2.1 CNN Stem（IQ 分支入口）

对复 IQ 进行浅层卷积下采样，得到多尺度时序特征图，作为后续时序建模的输入。CNN stem 负责局部波形模式提取，计算开销适中。

### 3.2.2 RF-HSTU 时序建模

在 stem 输出上堆叠 RF-HSTU（Hierarchical State Space Token Unit）块，对长序列 token 进行层次化状态空间建模，捕获 chirp 跨时间的依赖。相比纯 CNN，RF-HSTU 更适合 LoRa CSS 的长时结构。

### 3.2.3 OOB 分支

对 IQ 计算带外功率谱（或预提取 OOB 特征），经轻量编码器得到 OOB token 序列。带外谱对 PA 非线性、滤波滚降等硬件差异敏感，与带内 chirp 解调路径信息互补。

### 3.2.4 Cross-attention 融合

**不使用**简单 concat，而采用 in-band token 作为 Query、OOB token 作为 Key/Value 的 cross-attention，使融合发生在对齐后的语义空间。消融表明 concat fusion 在跨日下严重 collapse（见 3.5 节）。

### 3.2.5 Chirp-aware Embedding

引入与 LoRa chirp 结构相关的嵌入或位置编码，作为辅助先验。消融显示在 cross-attention 已启用时，chirp 对均值准确率影响有限，但有助于稳定训练。

### 3.2.6 分类头与投票

全局池化后接线性分类器；测试时对同一文件内多窗口 logits 做 **mean-logits voting** 得到文件级预测。

## 3.3 训练与评估协议

| 项目 | 设定 |
|------|------|
| 数据 manifest | `cross_day_day1to5_source_only.csv` |
| Train / Val / Test | Day1–3 / Day4 / Day5 |
| OOB 归一化 | zscore |
| Batch size | 128 |
| Learning rate | 3e-3 |
| Epochs | 80 |
| Seeds | 5 |
| 基线 | CNN-IQ、RF-HSTU（无 OOB） |

所有超参在验证集上选取；测试集仅用于最终报告。

## 3.4 跨日期主结果

表 3-1 汇总主结果（CSV：`table1_cross_day_main.csv`）。

| 模型 | File-Acc (%) | File-F1 (%) |
|------|--------------|-------------|
| CNN-IQ | 54.2±14.2 | 45.6±14.8 |
| RF-HSTU（无 OOB） | 66.7±3.4 | 59.1±4.5 |
| **Ours** | **75.0±5.3** | **67.9±6.8** |

Ours 相对 CNN-IQ 平均提升 **+20.8 pp**；bootstrap 95% CI 约为 [+9.2, +32.5] pp；5 seeds 上为 4 win / 1 tie / 0 loss。Per-seed File-Acc：83.3, 70.8, 70.8, 79.2, 70.8。

图 3-2 展示各 seed 柱状对比。结论：在 same-receiver cross-day 下，OOB-guided cross-attentive RF-HSTU 显著优于 CNN-IQ 与无 OOB 的 RF-HSTU，支持创新点一的核心 claim。

**克制表述：** 提升显著但绝对精度仍受数据集规模与类数限制，尚不能等同于「野外万能识别器」。

## 3.5 消融实验

### 3.5.1 Fusion 与 Chirp（表 3-2）

| Fusion | Chirp | File-Acc (%) |
|--------|-------|--------------|
| concat | × | 18.3±18.6 |
| concat | ✓ | 9.7±2.0 |
| cross-attn | × | 75.0±3.4 |
| cross-attn | ✓ | **75.0±5.3** |

**结论：** cross-attention 是稳定增益来源；concat 在跨日下 collapse；chirp 不改变 cross-attn 均值。

### 3.5.2 无 OOB

RF-HSTU 无 OOB：66.7±3.4%，低于 Ours 75.0%，说明 OOB 分支贡献明确。

## 3.6 部署偏移分析（LOCO）

表 3-3 报告按配置、位置、距离留一（LOCO）的扩展实验（OOB norm 为 ratio，与 3.4 主实验 footnote 不同）。

要点：

- 配置（SF7/8/11）与多数位置/距离设定下 Ours 优于 CNN；
- 室外场景个别 fold CNN 略优；
- 距离 10–20 m 时 Ours 为 20.8–37.5%，CNN 多为 4.2–8.3%。

该实验说明模型对部分部署因子有一定泛化，但绝对精度随 held-out 条件仍显著波动，属于附录级扩展而非主 claim。

## 3.7 跨接收机 Stress Test（局限性与第 4 章动机）

**协议：** strict source-only；验证集为源 RX；测试集为目标 RX；3 seeds；F 使用 ratio OOB norm。

| 方向 | CNN-IQ (%) | Ours (%) |
|------|------------|----------|
| RX1→RX2 | 4.2±0.0 | **18.1±3.9** |
| RX2→RX1 | 23.6±15.3 | 15.3±7.1 |

Chance：4.17%。

**分析：**

- RX1→RX2：Ours 优于 CNN，但 18.1% 仍远低于 cross-day 75%；
- RX2→RX1：方向不对称，CNN 有时更优 → **不能声称 receiver-invariant**；
- OOB 路径在跨 RX 时可能引入接收机谱偏置，加剧失配。

因此，第 3 章方法解决的是 **同接收机跨日鲁棒建模**；strict cross-receiver 需第 4 章 target-side calibration。

## 3.8 边缘部署参考

表 3-4（可选）：CNN-IQ 约 1.16M 参数、2.45 ms；Hybrid Ours 约 47.7K 参数（仅统计部分分支）、1.12 ms（bs=1）。说明混合架构在特定实现下具备边缘推理可行性，完整认证系统部署仍超出本章范围。

## 3.9 本章小结

本章针对 same-receiver cross-day 闭集识别，提出 OOB-guided cross-attentive RF-HSTU。实验表明：（1）cross-attention 融合优于 concat；（2）OOB 与 RF-HSTU 联合将 File-Acc 提升至 75.0±5.3%，显著优于 CNN-IQ；（3）部署 LOCO 扩展显示一定泛化但波动大；（4）strict cross-receiver 下性能接近 chance，且 OOB 可能参与接收机诱导混淆——这直接引出第 4 章的失配诊断与 RCPA-T 校准。

本章 **未** 解决换接收机、复杂电磁扰动与开放集认证问题，相关协议与结论见第 4、5 章。
