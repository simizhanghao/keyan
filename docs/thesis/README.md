# LoRa RFFI 硕士学位论文写作包

**题目（候选）：** 面向复杂部署环境的 LoRa 设备射频指纹鲁棒认证方法研究

**英文：** Robust LoRa Radio-Frequency Fingerprint Authentication under Complex Deployment Conditions

**代码与实验仓库：** [hanCChan/rf-hstu-lora](https://github.com/hanCChan/rf-hstu-lora)（分支 `thesis-em-openset`，commit `f9dbe1c`）

---

## 文档索引

| 文件 | 说明 |
|------|------|
| [THESIS_OUTLINE.md](THESIS_OUTLINE.md) | 全文提纲、摘要结构、章节字数建议 |
| [THESIS_INNOVATION_SUMMARY.md](THESIS_INNOVATION_SUMMARY.md) | 三项创新点与贡献陈述 |
| [THESIS_EXPERIMENT_MAP.md](THESIS_EXPERIMENT_MAP.md) | 实验—章节—结论—commit 映射表 |
| [THESIS_FIGURE_TABLE_LIST.md](THESIS_FIGURE_TABLE_LIST.md) | 图表清单与来源路径 |
| [THESIS_DEFENSE_QA.md](THESIS_DEFENSE_QA.md) | 答辩 15 题 Q&A |
| [chapters/](chapters/) | 六章正文 Markdown 草稿 |

---

## 三章创新主线

```text
第 3 章：OOB-guided RF-HSTU — same-receiver / cross-day
第 4 章：Cross-receiver diagnosis + RCPA-T
第 5 章：EM benchmark + open-set authentication（非 EM-CR 成功）
```

---

## 章节草稿

| 章 | 文件 |
|----|------|
| 1 绪论 | [chapter1_introduction.md](chapters/chapter1_introduction.md) |
| 2 理论与数据集 | [chapter2_background_dataset.md](chapters/chapter2_background_dataset.md) |
| 3 OOB RF-HSTU | [chapter3_oob_rfhstu.md](chapters/chapter3_oob_rfhstu.md) |
| 4 RCPA-T | [chapter4_cross_receiver_rcpa.md](chapters/chapter4_cross_receiver_rcpa.md) |
| 5 EM + Open-set | [chapter5_em_openset.md](chapters/chapter5_em_openset.md) |
| 6 总结展望 | [chapter6_conclusion.md](chapters/chapter6_conclusion.md) |

---

## 写作原则

1. Claim 与实验一一对应，不夸大「完全解决」；
2. Paper 1 / Paper 2 / 第 5 章协议边界清晰；
3. **EM-CR 写 preliminary，不作第三创新点核心**；
4. 失败实验诚实表述（CFO 失效、EM-CR 未达标）。

---

## 图表资源（在代码仓内）

- 第 3 章：`outputs/paper_ready_v3/final_figures/`
- 第 4 章：`docs/paper2_rcpa/figures/`
- 第 5 章：`docs/thesis_chapter5_em_openset/figures/`

排版时按 `THESIS_FIGURE_TABLE_LIST.md` 嵌入 PDF。

---

## 状态

- 第三创新点实验已收口（commit `f9dbe1c`）
- 不再扩展 EM-CR full training
- 下一步：人工通读主线 → Word/LaTeX 正式排版
