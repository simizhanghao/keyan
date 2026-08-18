# 第 5 章：复杂电磁扰动与未知设备场景下的 LoRa 射频指纹鲁棒认证

## 题目

**复杂电磁扰动与未知设备场景下的 LoRa 射频指纹鲁棒认证方法**

英文：EM-Robust Open-Set LoRa RFFI Authentication

## 在毕设中的位置

| 章节 | 创新点 | 核心问题 |
|------|--------|----------|
| 第 3 章 | OOB-guided RF-HSTU | 同接收机 / 跨天识别更稳 |
| 第 4 章 | Cross-receiver 诊断 + RCPA-T | 换接收机后为何失败、如何校准 |
| **第 5 章** | **EM 鲁棒 + 开放集认证** | **复杂电磁环境下如何认证、如何拒识未知设备** |

**不修改** Paper 1 / Paper 2 主线；第三创新点独立成章。

## 问题定义

现有 RFFI 多数评估 **closed-set identification**（测试设备均在训练集合内）。真实物联网网关在认证时面临：

1. 噪声、窄带干扰、载波频偏、相位噪声、I/Q 失衡、接收机频响漂移等 **复杂电磁扰动**；
2. 训练集中未出现的 **未知 / 伪装设备** 需要拒识。

第三创新点将 LoRa RFFI 从 closed-set identification 扩展为 **EM-robust open-set authentication**。

## 研究目标

1. 构建物理合理的 LoRa IQ **电磁扰动 benchmark**；
2. 评估 Paper 1 RF-HSTU backbone 在多种 EM 扰动下的鲁棒性退化；
3. 提出 **EM-aware consistency fine-tuning（EM-CR）**，提高扰动鲁棒性；
4. 构建 **open-set authentication** 协议（20 known + 4 unknown devices）；
5. 分析 EM 扰动对认证阈值、FAR/FRR、AUROC、EER 的影响。

## 方法概要

### EM 扰动 benchmark

- AWGN、窄带干扰（SIR）、CFO、相位噪声、I/Q 失衡、滤波器漂移、混合应力
- 实现：`src/rfhstu/em_perturbations.py`
- 配置：`experiments/em_robustness_openset/results/em_benchmark_config.json`

### EM-CR（电磁扰动一致性约束）

对同一样本 $x$ 生成 clean 与扰动版本 $A_{\mathrm{em}}(x)$：

$$\mathcal{L} = \mathrm{CE}(y, p(x)) + \mathrm{CE}(y, p(A_{\mathrm{em}}(x))) + \lambda \,\mathrm{KL}(p(x)\|p(A_{\mathrm{em}}(x)))$$

训练版本：Clean-trained（已有） / EM-Aug CE / EM-CR / EM-CR+embedding consistency

### Open-set authentication

- 24 设备中 20 known + 4 unknown，3 个 split seeds
- 分数：MSP、Energy、Prototype distance、Mahalanobis
- 指标：AUROC、AUPR、FPR@95TPR、EER、FAR、FRR、Known Acc
- 阈值仅在 validation set 上选取

## 实验协议

- **数据**：OSU LoRa 24 设备，cross-day：Day1–3 train，Day4 val，Day5 test
- **Backbone**：冻结评估 Paper 1 `F_cross_attn_chirp_plain`；EM-CR 在此基础上微调
- **分支**：`thesis-em-openset`

## Claim 边界

**可以说：**

- 构建了面向 LoRa RFFI 的复杂电磁扰动鲁棒性评估协议；
- 提出 EM-CR 提高噪声/频偏/窄带干扰等扰动下的鲁棒性；
- 将 RFFI 扩展到 open-set authentication；
- 分析复杂电磁扰动对未知设备拒识的影响。

**不要说：**

- 解决所有复杂电磁环境；
- 实现完全鲁棒认证或工程可部署；
- 覆盖真实世界所有干扰类型；
- 优于所有 open-set 方法。

## 运行顺序

1. **Phase A（smoke）**：`bash experiments/em_robustness_openset/run_em_smoke.sh`
2. **Phase B**：`run_em_full.sh` — 全扰动 closed-set 曲线
3. **Phase C**：`train_em_consistency.py` — EM-CR 训练
4. **Phase D**：`run_openset_full.sh` + `eval_openset_under_em.py`
5. **Phase E**：Chapter 5 报告与图表

## 主表 / 主图（计划）

- Table 5-1：EM perturbation benchmark
- Table 5-2：Closed-set robustness（CNN-IQ vs Ours vs EM-CR）
- Table 5-3：EM-CR ablation
- Table 5-4：Open-set authentication（clean）
- Table 5-5：Open-set under EM

- Fig. 5-1：EM robustness curves
- Fig. 5-2：EM-CR comparison
- Fig. 5-3：Open-set ROC/DET
- Fig. 5-4：AUROC/EER under EM
