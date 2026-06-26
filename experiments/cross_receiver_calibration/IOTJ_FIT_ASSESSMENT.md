# IoTJ Fit Assessment — Paper 2 (RCPA)

> Internal decision memo. Not for submission.

## Short answer

**Topic fit: 高。** IoTJ 明确覆盖 IoT security、physical-layer authentication、RFFI；近年有多篇 LoRa/IoT RFFI 论文发表在该刊。

**作为独立第二篇投 IoTJ: 可行，但有审稿风险，需要写清与第一篇的边界。**

**更稳妥策略:** IoTJ 仍是首选候选之一；同时准备 TWC/TIFS/Sensors 作为备选（若编辑认为与第一篇重叠过多）。

---

## IoTJ scope 匹配度

| 维度 | 评估 |
|------|------|
| IoT device authentication | ✅ 核心应用 |
| Physical-layer / RFFI | ✅ IoTJ 2024–2026 多篇 RFFI |
| LoRa / low-power IoT | ✅ 有 LoRa fingerprinting 先例 |
| Security under domain shift | ✅ cross-day UDA SEI 已出现在 IoTJ 2026 |
| System + diagnosis + method | ✅ 符合 IoT-J "enabling technologies + applications" |

参考：IoTJ 已发表 MRFE (2024)、Model-Based RF Fingerprint (2025)、cross-day SEI UDA (2026)、noise-robust RFFI (2026) 等。

---

## 投 IoTJ 的优势

1. **与第一篇形成系列** — Paper 1 诚实报告 cross-RX limitation；Paper 2 专门解决，叙事连贯。
2. **读者群一致** — IoT 部署、LoRa 认证、轻量安全。
3. **结果强度够** — 58–75% vs ~20%，3×3 重复，超过 many IoTJ RFFI 实验深度。
4. **Diagnosis + mechanism** — 不是纯刷分，有 OOB 物理/表示层证据，符合 IoTJ 对 "insights" 的偏好。

---

## 投 IoTJ 的主要风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| 与第一篇 overlap（同 backbone、同数据集） | **高** | 开篇声明 Paper 1 已投/发表；本文 **zero backbone change**；引用并扩展 limitation |
| "只是 domain adaptation / prototype" | 中 | Diagnosis-first framing；OOB entanglement；TTA negative；block-disjoint protocol |
| K-shot vs source-free 不公平 | 中 | 明确 deployment mode；TTA/pseudo 负结果；讨论 labeled cal 的工程可接受性 |
| 未与 SOTA DA 方法数值对比 | 中 | Related work 对比表；讨论 receiver-agnostic / SCRFFI；可选补 CORAL/MMD baseline on embeddings |
| 页数（IoTJ >8页 $175/页） | 低 | 主文 ≤8 页；诊断细节放 supplementary |

---

## 与已发表/近期 cross-RX RFFI 的差异（审稿人可能问）

- **TMC 2023 receiver-agnostic:** 需多 receiver 训练 + adversarial retrain → 我们 **frozen backbone + 少量 target labels**
- **SCRFFI source-free:** 仅 target 无标签 → 我们证明 **TTA/pseudo 不足**，labeled K-windows 更可靠
- **Disentanglement 2025:** 新训练框架 → 我们是 **post-hoc + diagnosis**

---

## 是否建议投 IoTJ？

```text
建议：可以投 IoTJ，但作为「独立 Regular Paper」，不是第一篇的 appendix。

前提：
1. 第一篇 IoTJ 状态在 cover letter 中披露；
2. 摘要/引言第一句就区分 contribution type（calibration vs architecture）；
3. Related work 主动对比 receiver-agnostic / source-free，不回避；
4. 主表冻结 RCPA-T；OOB-Eq + TTA 放 supplementary。
```

### 若 IoTJ 编辑认为 overlap 过大

备选期刊（按 RFFI + security 匹配度）：
- **IEEE TIFS** — 物理层安全、RFFI 常见；门槛更高
- **IEEE TWC** — 无线 + ML；cross-RX 常见
- **IEEE IoT-J Special Issue** — 若有 security/fingerprinting SI 可跟踪
- **Sensors (MDPI)** — SCRFFI 同类；快但需评估认可度

---

## Cover letter 关键句（建议）

> This manuscript is a companion study to our IoTJ submission on OOB-guided RF-HSTU modeling. That work establishes a strong same-receiver backbone and reports cross-receiver as an open limitation. **The present paper does not propose a new backbone.** Instead, it (i) diagnoses receiver-induced OOB feature entanglement under cross-receiver shift, and (ii) introduces a lightweight target-receiver prototype calibration protocol that restores device separability using only K labeled calibration windows per device.

---

## 最终建议

| 问题 | 答案 |
|------|------|
| 能发 IoTJ 吗？ | **能，topic fit 足够** |
| 应该发 IoTJ 吗？ | **首选尝试 IoTJ**，与第一篇形成系列；准备好 overlap 辩护 |
| 现在还要优化方法吗？ | **不要**；优化论文定位、对比、表述 |
| 下一步 | outline ✅ → related work 表 ✅ → Methods/Results 初稿 → TTA threshold appendix（可选小 sweep） |
