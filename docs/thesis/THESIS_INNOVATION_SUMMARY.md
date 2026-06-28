# 创新点与贡献总结

## 总体定位

本文围绕 **LoRa 设备射频指纹认证在复杂部署条件下的可靠性** 开展研究，从同接收机跨日期建模、跨接收机校准、复杂电磁环境与开放集认证三个层次递进，形成可复现的评估协议与可落地的分析结论。

---

## 创新点一：OOB-guided RF-HSTU 混合建模（第 3 章）

**针对问题：** 同接收机、跨采集日期的 LoRa RFFI 性能退化。

**主要贡献：**

1. 提出将 **带内 IQ 与带外（OOB）频谱证据** 联合建模的 OOB-guided RF-HSTU 架构；
2. 设计 **cross-attentive OOB fusion**，使模型在跨日期条件下选择性利用 OOB 判别信息，避免简单拼接导致的表征塌缩；
3. 在 OSU LoRa 跨日协议下，文件级准确率 **75.0±5.3%**，较 CNN-IQ **54.2±14.2%** 提升 **20.8 pp**（Bootstrap 95% CI：+9.2～+32.5 pp）；
4. 通过部署偏移（配置/位置/距离）实验分析模型在更广部署条件下的行为边界。

**局限（诚实表述）：** strict cross-receiver 下性能仍接近 chance，引出第 4 章。

---

## 创新点二：跨接收机失配诊断与 RCPA-T 校准（第 4 章）

**针对问题：** 换接收机后 source-only 模型失效及 OOB 路径的接收机诱导纠缠。

**主要贡献：**

1. 系统诊断 cross-receiver transfer failure，揭示 **receiver-induced OOB feature entanglement**（OOB 接收机探针 72.7%，设备探针仅 28.9%）；
2. 提出 **RCPA-T**：冻结 backbone，利用少量目标接收机 labeled windows 构造 **target-receiver prototype**，在 block-disjoint 协议下校准；
3. Pooled K=20 时文件级准确率 **75.0±8.0%**，较 source classifier 提升约 **54.9 pp**；
4. 与 linear probe、head FT、无标签对齐等 **同协议 baseline** 对比，说明原型校准在跨接收机场景下的有效性。

**局限：** 需要 labeled target windows；非 source-free；不引入新 backbone。

---

## 创新点三：复杂电磁扰动与开放集未知设备认证（第 5 章）

**针对问题：** 真实电磁干扰下闭集鲁棒性未知，以及未知设备拒识缺乏系统评估。

**主要贡献：**

1. 构建覆盖 AWGN、CFO、窄带干扰、相位噪声、IQ imbalance、filter drift、mixed stress 的 **EM perturbation benchmark**；
2. 量化扰动敏感性：**CFO 与强 AWGN 为最强破坏源**；窄带干扰相对温和；
3. 对比 CNN-IQ：Ours 在 clean、AWGN 30 dB、NBI 10 dB 下显著更优，但 **CFO 0.003 下两者均接近失效**；
4. 将认证扩展至 **open-set**（20 known + 4 unknown），证明 **Prototype / Mahalanobis**（AUROC 0.91+）显著优于 MSP / Energy（0.43～0.58）；
5. 在 EM stress 下分析 open-set 退化，发现 **CFO 同时破坏 known classification 与 unknown rejection**；
6. EM-CR 扰动一致性微调作为 **初步探索**：未达主方法标准，分析失败原因并指向 future work。

**明确不作为贡献：** 「提出 EM-CR 并显著提升鲁棒性」。

---

## 三章逻辑关系

```text
第 3 章：学到稳定 device embedding（同 RX、跨日）
    ↓ 跨 RX 失效
第 4 章：诊断失配 + 目标 RX 原型校准
    ↓ 复杂电磁 + 未知设备
第 5 章：扰动 benchmark + open-set 协议 + 失效模式分析
```

---

## 与已发表论文的关系

| 成果 | 毕设章节 | 说明 |
|------|----------|------|
| IoTJ Paper 1 | 第 3 章 | 主体实验与消融 |
| Paper 2 RCPA | 第 4 章 | 诊断 + RCPA-T + baselines |
| 第 5 章实验 | 第 5 章 | 独立 benchmark，不并入 Paper 2 |

毕设正文应 **整合叙述**，避免三篇割裂；各章协议差异需在文中明确标注。
