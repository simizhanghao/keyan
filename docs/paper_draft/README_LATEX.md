# IoTJ 中文 LaTeX 论文稿

## 文件

- `IoTJ_中文论文_LoRa_RFFI_Hybrid.tex` — IEEE 双栏中文初稿（IoTJ 逻辑）

## 编译（需 TeX Live + XeLaTeX + ctex）

```bash
cd docs/paper_draft
xelatex IoTJ_中文论文_LoRa_RFFI_Hybrid.tex
xelatex IoTJ_中文论文_LoRa_RFFI_Hybrid.tex   # 第二遍更新引用
```

或使用 latexmk：

```bash
latexmk -xelatex IoTJ_中文论文_LoRa_RFFI_Hybrid.tex
```

## 说明

- 文档类：`IEEEtran` journal 双栏
- 中文：`ctex`（XeLaTeX）
- 附录含英文摘要草稿（IoTJ 正式稿 150–250 词）
- 图 1/2 为占位框，需替换为实际框架图
- 参考文献为占位条目，投稿前替换为真实 BibTeX

## 数据口径（已写入正文）

| 实验 | 目录 |
|------|------|
| Day1–4→Day5 | `outputs/single_gpu_compare_20260614_141808/` |
| LODO | `outputs/lodo_day1to5/` |
| 跨接收机 80ep | `outputs/cross_receiver_norm_confirm_80ep/` |

**注意**：74.6% 为 Day1–4→Day5，非 Day1→Day2。
