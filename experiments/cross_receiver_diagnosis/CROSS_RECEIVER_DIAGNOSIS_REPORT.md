# Cross-Receiver Failure Diagnosis Report

> Auto-generated from `experiments/cross_receiver_diagnosis/results/run_20260626`  
> Protocol: Phase5-clean checkpoints (RX1-trained), embeddings on RX1+RX2 all devices  
> Metric: cosine distance on file-level mean embeddings unless noted

---

## 1. 核心结论：失败机理是什么？

**不是「再堆一个 HSTU」的问题。** 诊断显示 cross-receiver 失败来自多层因素：

| 层级 | 证据 | 结论 |
|------|------|------|
| 物理层 | OOB RX2/RX1 能量比 **1.4420** | 接收机前端造成带外频谱系统性偏移 |
| 表示层 | OOB path receiver probe **0.73** > main **0.62** | OOB 分支更含 receiver 信息，cross-attn 可能查询到 RX-specific OOB |
| 表示层 | CNN distance ratio **1.2508** > 1 | CNN embedding 中 receiver shift 盖过 device separation |
| 决策层 | Ours device probe **0.2894** (24类) | 混合 RX1/RX2 后 device 线性可分性很差 |
| 行为层 | RX1→RX2 CNN top1 mass **95.8%** | 严重 prediction collapse |

**一句话：** OOB 在 fixed receiver 下是 device evidence；cross-receiver 下 OOB 与 receiver response 纠缠，叠加分类器边界偏移，导致 strict source-only 迁移失败。

---

## 2. Embedding 距离诊断（cosine, file-level）

| 模型 | same-dev cross-RX | diff-dev same-RX | ratio | ratio>1 |
|------|-------------------|------------------|-------|---------|
| CNN-IQ | 0.4104 | 0.3281 | 1.2508 | True |
| Ours fused | 0.1688 | 0.7642 | 0.2208 | False |

**注意：** Ours fused ratio < 1 不代表「没有 receiver 问题」——它说明同设备跨 RX 在方向上仍较近，但 device 间 margin 不足，无法支撑 24 类闭集分类。

---

## 3. Receiver / Device Probe

| 模型 | Receiver probe (file, CV) | Device probe (window, CV) | RX/Device ratio |
|------|---------------------------|---------------------------|-----------------|
| CNN-IQ | 0.7266666666666666 | 0.10856171222322566 | 6.693581482691498 |
| Ours fused | 0.5022222222222222 | 0.2893883502508401 | 1.7354610915985353 |

---

## 4. Main / OOB / Fused 路径对比（Ours）

| Path | Receiver CV acc (file) | Device CV acc (window) |
|------|------------------------|-------------------------|
| main | 0.6244444444444445 | 0.05664071335348797 |
| oob | 0.7266666666666666 | 0.2891433552133308 |
| fused | 0.5022222222222222 | 0.2893883502508401 |

**关键发现：** OOB-only path 的 receiver probe 最高 → **OOB cross-attention 在跨 RX 下可能是负迁移源之一**（same-RX 下则是增益源）。

---

## 5. OOB 频谱 receiver bias（物理层）

| 指标 | 值 |
|------|-----|
| RX2/RX1 OOB energy ratio | 1.4420 |
| Mean per-device OOB shift | 0.7551 |
| Mean ratio spectrum std | 0.1726 |

图：`results/.../oob_spectrum/rx_mean_oob_spectrum.png`

---

## 6. Confusion / collapse（RX1→RX2 尤其严重）

| 实验 | file-acc | top1 pred mass | active classes |
|------|----------|----------------|----------------|
| A_cnn_iq/rx1_to_rx2/seed_0 | 0.042 | 0.958 | 2 |
| A_cnn_iq/rx1_to_rx2/seed_1 | 0.042 | 0.875 | 4 |
| A_cnn_iq/rx1_to_rx2/seed_2 | 0.042 | 0.875 | 4 |
| A_cnn_iq/rx2_to_rx1/seed_0 | 0.417 | 0.167 | 13 |
| A_cnn_iq/rx2_to_rx1/seed_1 | 0.042 | 0.208 | 13 |
| A_cnn_iq/rx2_to_rx1/seed_2 | 0.250 | 0.250 | 11 |
| F_cross_attn_chirp_plain/rx1_to_rx2/seed_0 | 0.208 | 0.208 | 14 |
| F_cross_attn_chirp_plain/rx1_to_rx2/seed_1 | 0.125 | 0.208 | 15 |
| F_cross_attn_chirp_plain/rx1_to_rx2/seed_2 | 0.208 | 0.458 | 9 |
| F_cross_attn_chirp_plain/rx2_to_rx1/seed_0 | 0.250 | 0.208 | 15 |
| F_cross_attn_chirp_plain/rx2_to_rx1/seed_1 | 0.125 | 0.250 | 14 |
| F_cross_attn_chirp_plain/rx2_to_rx1/seed_2 | 0.083 | 0.208 | 13 |

**RX1→RX2：** CNN 几乎 collapse 到 1–2 类；Ours 略好但仍低 acc。  
**RX2→RX1：** 方向不对称，CNN 有时更好 → 两 RX 不是对称域。

---

## 7. 方法选择（基于诊断，不是盲试）

- CNN-IQ：cosine distance ratio > 1 → 跨接收机后 receiver shift 在 CNN embedding 中盖过 device separation
- Ours fused：ratio < 1 → 同设备跨 RX 方向相似度仍保留，但 **device probe 仅 ~29%**，说明分类边界/决策层失效，而非简单 embedding 完全不可分
- OOB path receiver probe (0.73) > main path (0.62) → **OOB 分支更 receiver-entangled**，支持 OOB 带外证据在跨 RX 下引入接收机前端印记
- OOB 频谱 RX2/RX1 能量比 = 1.44 → 存在显著 receiver spectral response bias（物理层证据）
- 推荐主方法：**few-shot receiver calibration (RCPA)** + **OOB receiver response equalization** 作为物理解释分支
- TTA/entropy 可作为 unsupervised 对比，但 RX1→RX2 collapse 严重时 pseudo-label 风险高
- 不推荐：单 source RX 上的 GRL adversarial disentanglement（训练未见过 receiver variation）

---

## 8. 第二篇论文建议叙事

**题目方向：** Diagnosing and Mitigating Receiver-Induced Feature Entanglement in LoRa RFFI

**三段式：**
1. **Diagnosis** — OOB receiver bias + probe + collapse（本章）
2. **Calibration** — RCPA few-shot / OOB equalization
3. **Deployment modes** — 0-shot / unsupervised TTA / K-shot curve

**不能写：** 「更强的 RF-HSTU 解决 cross-receiver」  
**应该写：** 「诊断 receiver entanglement → 轻量 receiver calibration 恢复 transmitter separability」

---

## 附录

```text
experiments/cross_receiver_diagnosis/
├── extract_embeddings.py
├── analyze_distances.py
├── train_probes.py
├── plot_embeddings.py
├── analyze_path_ablation.py
├── analyze_oob_spectrum.py
├── plot_confusion.py
├── generate_report.py
└── run_diagnosis.sh
```
