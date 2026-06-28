# 第 6 章 总结与展望

## 6.1 全文工作总结

本文围绕「面向复杂部署环境的 LoRa 设备射频指纹鲁棒认证」展开研究。针对物联网低功耗广域网中设备身份可信需求，在 OSU LoRa 数据集及可复现实验协议下，从**时间漂移（跨日）**、**空间失配（跨接收机）**与**复杂电磁环境及开放集身份**三个维度，递进分析并改进 LoRa RFFI 系统的识别与认证可靠性。

全文不追求单一「端到端万能模型」，而是采用**问题分解 + 协议化评估 + 诚实失效分析**的研究范式，使各章 claim 与实验边界清晰可辩护。

## 6.2 创新点回顾

### 6.2.1 创新点一：OOB-guided RF-HSTU（第 3 章）

提出带外频谱引导的 cross-attentive RF-HSTU 混合模型，融合 in-band IQ 与 out-of-band spectral evidence。在 same-receiver cross-day 闭集协议下，文件级准确率 **75.0±5.3%**，较 CNN-IQ **54.2±14.2%** 提升约 **20.8 pp**；消融证实 cross-attention 优于 concat，OOB 分支贡献明确。同时报告 strict cross-receiver stress 下仍 near chance，为第 4 章铺垫。

### 6.2.2 创新点二：跨接收机诊断与 RCPA-T（第 4 章）

诊断 cross-receiver transfer failure，揭示 OOB 路径上 **receiver-induced feature entanglement**（receiver probe 72.7%，距离比 collapse，预测集中）。提出 RCPA-T，在冻结主干下用每设备 K 个 labeled target-receiver windows 构造目标原型；block-disjoint 协议下 K=5/10/20 达 **58.3% / 69.4% / 75.0%**，远高于 source-only ~20%。诚实对比同协议 linear probe（K=5 59.0%），并说明非 source-free。

### 6.2.3 创新点三：复杂电磁扰动与开放集认证（第 5 章）

构建多类型 EM perturbation benchmark；闭集上量化 AWGN、CFO、NBI 等退化规律；扩展 open-set unknown device authentication。Ours clean **83.3%**；CFO ≥0.003 **~4.2%**；NBI 10 dB **87.5%**；Prototype open-set AUROC **0.917±0.059**，优于 MSP **0.425**；CFO 下 known **3.3%** 与 AUROC **0.492** 双崩溃。EM-CR 为 preliminary，不纳入主结论。

## 6.3 实验结果总结

| 维度 | 关键数值 | 含义 |
|------|----------|------|
| Cross-day | 75.0% vs 54.2% | 同 RX 跨日建模有效 |
| Cross-RX source-only | ~20% | 无校准不可用 |
| RCPA-T K=20 | 75.0% | 少样本目标校准可恢复 |
| EM clean | 83.3% | 与 cross-day 峰值一致 |
| EM CFO 0.003 | 4.2% | 系统级瓶颈 |
| Open-set Proto | 0.917 AUROC | 嵌入距离适合认证 |

## 6.4 不足与局限

1. **数据集规模有限：** OSU LoRa 设备数、天数与场景覆盖有限，结论外推需谨慎；
2. **跨接收机仅 RX1/RX2：** 多接收机、多厂商网关未充分验证；
3. **Open-set unknown：** 从已采集设备中 hold-out，非真实野外未知型号；
4. **CFO 鲁棒性不足：** 当前特征与模型均未解决强 CFO，需波形级补偿；
5. **EM-CR 未形成稳定方法：** 扰动一致性训练易遗忘或无效；
6. **认证系统要素不全：** 密钥融合、重放防护、在线阈值管理与安全审计未实现；
7. **第 3–5 章协议各异：** 读者需注意 norm、split、K-shot 等差异，不可混用数值。

## 6.5 未来展望

1. **多接收机、多场景采集：** 扩展 RX 数量与环境，验证 RCPA-T 与诊断指标的可迁移性；
2. **Waveform-level CFO compensation：** 在特征提取前估计并补偿频偏，缓解最强 failure mode；
3. **Receiver-statistics calibration：** 无标签或弱标签下对接收机谱统计做对齐，向 source-free 靠拢；
4. **Open-set 阈值自适应：** 随环境应力动态调整拒识阈值，降低 FAR/FRR 波动；
5. **课程式 EM 增强与 teacher-student：** 替代简单 EM-CR，在冻结主干前提下渐进引入扰动；
6. **与 LoRaWAN 认证体系结合：** 将 RFFI 作为辅助因子嵌入 Join 与会话安全流程；
7. **更大规模预训练与基础模型：** 跨数据集 RF 表征可能提升小样本校准上限。

## 6.6 结束语

本文在可复现协议下，为 LoRa RFFI 提供了从同接收机鲁棒建模、跨接收机少样本校准到复杂电磁与开放集评估的完整研究链条。实验既报告了显著增益，也标定了 CFO 等失效边界与 EM-CR 等未成功探索。希望本工作为后续 LoRa 物理层安全研究提供基准、方法与诚实的经验参考。

## 本章小结

本章总结了全文工作、三项创新点与核心实验数值，分析了数据集、协议与方法上的不足，并展望了 CFO 补偿、多接收机扩展、开放集阈值与系统级部署等方向。硕士论文的学术贡献在于：**提高特定协议下的鲁棒性、提供评估框架、诊断失效机理**，而非宣称已彻底解决 LoRa 设备认证问题。
