# 硕士学位论文提纲

## 题目（候选）

**中文：** 面向复杂部署环境的 LoRa 设备射频指纹鲁棒认证方法研究

**英文：** Robust LoRa Radio-Frequency Fingerprint Authentication under Complex Deployment Conditions

**代码与实验仓库：** `hanCChan/rf-hstu-lora`（分支 `thesis-em-openset`，commit `f9dbe1c`）

---

## 摘要结构（待正式撰写）

1. 背景：LoRa 物联网设备认证需求与射频指纹（RFFI）思路；
2. 问题：跨日期漂移、跨接收机失配、复杂电磁扰动与未知设备；
3. 方法：OOB-guided RF-HSTU；跨接收机诊断与 RCPA-T；EM benchmark 与开放集认证；
4. 结果：跨日 75.0% vs CNN 54.2%；RCPA-T K=20 75.0%；EM 下 CFO 为最强失效模式；Prototype open-set AUROC 0.917；
5. 结论与展望。

---

## 章节结构

| 章 | 标题 | 对应成果 | 字数建议 |
|----|------|----------|----------|
| 1 | 绪论 | — | 8–10k |
| 2 | 相关理论与数据集 | OSU LoRa | 10–12k |
| 3 | 带外频谱引导的 RF-HSTU 鲁棒建模 | Paper 1 / IoTJ | 15–18k |
| 4 | 跨接收机失配诊断与 RCPA-T 校准 | Paper 2 | 15–18k |
| 5 | 复杂电磁扰动与开放集未知设备认证 | Chapter 5 | 15–18k |
| 6 | 总结与展望 | — | 4–6k |

---

## 三条创新点主线

```text
创新点一（第 3 章）
  OOB-guided cross-attentive RF-HSTU
  → same-receiver / cross-day 鲁棒闭集识别

创新点二（第 4 章）
  Cross-receiver diagnosis + RCPA-T
  → receiver mismatch 诊断与少样本目标接收机校准

创新点三（第 5 章）
  EM perturbation benchmark + open-set authentication
  → 复杂电磁环境下认证可靠性评估与 embedding-based 拒识
```

---

## 图表规划

- 第 3 章：架构图 1；主结果表 1；消融表 1；部署/距离图 2
- 第 4 章：诊断四联图 1；RCPA shot curve 1；主表 2–3
- 第 5 章：EM 曲线 1；CNN 对比 3；open-set 3；under-EM 3

详见 `THESIS_FIGURE_TABLE_LIST.md`。

---

## 写作原则

1. Claim 与实验一一对应，不夸大「完全解决」；
2. Paper 1 / Paper 2 / 第 5 章边界清晰，不混用协议；
3. EM-CR 写为 preliminary，不作第三创新点核心；
4. 跨接收机 limitation 在第 3 章末引出第 4 章；
5. 第 4 章末引出复杂部署场景下的第 5 章。

---

## 文档索引

| 文件 | 用途 |
|------|------|
| `THESIS_INNOVATION_SUMMARY.md` | 创新点与贡献陈述 |
| `THESIS_EXPERIMENT_MAP.md` | 实验—章节映射 |
| `THESIS_FIGURE_TABLE_LIST.md` | 图表清单 |
| `THESIS_DEFENSE_QA.md` | 答辩问答 |
| `chapters/chapter*.md` | 各章正文草稿 |
