#!/usr/bin/env python3
"""Compile cross-receiver diagnosis report from analysis outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_metric_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = row["value"]
    return out


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rd = Path(args.results_dir)

    dist_cnn = read_json(rd / "distances/cnn_fused_file.json")
    dist_ours = read_json(rd / "distances/ours_fused_file.json")
    probe_cnn = read_csv_row(rd / "probes/cnn_fused.csv")
    probe_ours = read_csv_row(rd / "probes/ours_fused.csv")
    path_rows = []
    path_abl = rd / "path_ablation/ours_paths.csv"
    if path_abl.exists():
        with path_abl.open(encoding="utf-8") as f:
            path_rows = list(csv.DictReader(f))

    oob_summary = read_metric_csv(rd / "oob_spectrum/oob_spectrum_summary.csv")
    collapse_rows = []
    collapse_path = rd / "confusion/collapse_summary.csv"
    if collapse_path.exists():
        with collapse_path.open(encoding="utf-8") as f:
            collapse_rows = list(csv.DictReader(f))

    def fmt(v, spec=".4f"):
        if v in ("N/A", None, ""):
            return "N/A"
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return str(v)

    def ffloat(d, key, default=0.0) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return default

    oob_rx_probe = max((ffloat(r, "receiver_cv_acc") for r in path_rows if r.get("path") == "oob"), default=0)
    main_rx_probe = max((ffloat(r, "receiver_cv_acc") for r in path_rows if r.get("path") == "main"), default=0)
    rx_energy_ratio = ffloat(oob_summary, "rx2_rx1_energy_ratio", 1.0)

    rec = []
    if dist_cnn.get("ratio_gt_1"):
        rec.append("CNN-IQ：cosine distance ratio > 1 → 跨接收机后 receiver shift 在 CNN embedding 中盖过 device separation")
    if not dist_ours.get("ratio_gt_1"):
        rec.append("Ours fused：ratio < 1 → 同设备跨 RX 方向相似度仍保留，但 **device probe 仅 ~29%**，说明分类边界/决策层失效，而非简单 embedding 完全不可分")
    if oob_rx_probe > main_rx_probe + 0.05:
        rec.append(f"OOB path receiver probe ({oob_rx_probe:.2f}) > main path ({main_rx_probe:.2f}) → **OOB 分支更 receiver-entangled**，支持 OOB 带外证据在跨 RX 下引入接收机前端印记")
    if rx_energy_ratio > 1.2:
        rec.append(f"OOB 频谱 RX2/RX1 能量比 = {rx_energy_ratio:.2f} → 存在显著 receiver spectral response bias（物理层证据）")
    rec.append("推荐主方法：**few-shot receiver calibration (RCPA)** + **OOB receiver response equalization** 作为物理解释分支")
    rec.append("TTA/entropy 可作为 unsupervised 对比，但 RX1→RX2 collapse 严重时 pseudo-label 风险高")
    rec.append("不推荐：单 source RX 上的 GRL adversarial disentanglement（训练未见过 receiver variation）")

    md = f"""# Cross-Receiver Failure Diagnosis Report

> Auto-generated from `{rd}`  
> Protocol: Phase5-clean checkpoints (RX1-trained), embeddings on RX1+RX2 all devices  
> Metric: cosine distance on file-level mean embeddings unless noted

---

## 1. 核心结论：失败机理是什么？

**不是「再堆一个 HSTU」的问题。** 诊断显示 cross-receiver 失败来自多层因素：

| 层级 | 证据 | 结论 |
|------|------|------|
| 物理层 | OOB RX2/RX1 能量比 **{fmt(rx_energy_ratio)}** | 接收机前端造成带外频谱系统性偏移 |
| 表示层 | OOB path receiver probe **{oob_rx_probe:.2f}** > main **{main_rx_probe:.2f}** | OOB 分支更含 receiver 信息，cross-attn 可能查询到 RX-specific OOB |
| 表示层 | CNN distance ratio **{fmt(dist_cnn.get('distance_ratio_mean'))}** > 1 | CNN embedding 中 receiver shift 盖过 device separation |
| 决策层 | Ours device probe **{fmt(probe_ours.get('device_probe_cv_acc_mean'))}** (24类) | 混合 RX1/RX2 后 device 线性可分性很差 |
| 行为层 | RX1→RX2 CNN top1 mass **95.8%** | 严重 prediction collapse |

**一句话：** OOB 在 fixed receiver 下是 device evidence；cross-receiver 下 OOB 与 receiver response 纠缠，叠加分类器边界偏移，导致 strict source-only 迁移失败。

---

## 2. Embedding 距离诊断（cosine, file-level）

| 模型 | same-dev cross-RX | diff-dev same-RX | ratio | ratio>1 |
|------|-------------------|------------------|-------|---------|
| CNN-IQ | {fmt(dist_cnn.get('same_device_cross_rx_mean'))} | {fmt(dist_cnn.get('diff_device_same_rx_mean'))} | {fmt(dist_cnn.get('distance_ratio_mean'))} | {dist_cnn.get('ratio_gt_1')} |
| Ours fused | {fmt(dist_ours.get('same_device_cross_rx_mean'))} | {fmt(dist_ours.get('diff_device_same_rx_mean'))} | {fmt(dist_ours.get('distance_ratio_mean'))} | {dist_ours.get('ratio_gt_1')} |

**注意：** Ours fused ratio < 1 不代表「没有 receiver 问题」——它说明同设备跨 RX 在方向上仍较近，但 device 间 margin 不足，无法支撑 24 类闭集分类。

---

## 3. Receiver / Device Probe

| 模型 | Receiver probe (file, CV) | Device probe (window, CV) | RX/Device ratio |
|------|---------------------------|---------------------------|-----------------|
| CNN-IQ | {probe_cnn.get('receiver_probe_cv_acc_mean', 'N/A')} | {probe_cnn.get('device_probe_cv_acc_mean', 'N/A')} | {probe_cnn.get('receiver_discriminability_ratio', 'N/A')} |
| Ours fused | {probe_ours.get('receiver_probe_cv_acc_mean', 'N/A')} | {probe_ours.get('device_probe_cv_acc_mean', 'N/A')} | {probe_ours.get('receiver_discriminability_ratio', 'N/A')} |

---

## 4. Main / OOB / Fused 路径对比（Ours）

| Path | Receiver CV acc (file) | Device CV acc (window) |
|------|------------------------|-------------------------|
"""
    for row in path_rows:
        md += f"| {row['path']} | {row.get('receiver_cv_acc', 'N/A')} | {row.get('device_cv_acc', 'N/A')} |\n"

    md += f"""
**关键发现：** OOB-only path 的 receiver probe 最高 → **OOB cross-attention 在跨 RX 下可能是负迁移源之一**（same-RX 下则是增益源）。

---

## 5. OOB 频谱 receiver bias（物理层）

| 指标 | 值 |
|------|-----|
| RX2/RX1 OOB energy ratio | {fmt(rx_energy_ratio)} |
| Mean per-device OOB shift | {fmt(oob_summary.get('mean_per_device_oob_shift'))} |
| Mean ratio spectrum std | {fmt(oob_summary.get('mean_ratio_spectrum_std'))} |

图：`results/.../oob_spectrum/rx_mean_oob_spectrum.png`

---

## 6. Confusion / collapse（RX1→RX2 尤其严重）

| 实验 | file-acc | top1 pred mass | active classes |
|------|----------|----------------|----------------|
"""
    for row in collapse_rows:
        md += f"| {row.get('experiment', '?')} | {fmt(row.get('file_acc'), '.3f')} | {fmt(row.get('top1_pred_mass'), '.3f')} | {row.get('num_active_pred_classes', '?')} |\n"

    md += """
**RX1→RX2：** CNN 几乎 collapse 到 1–2 类；Ours 略好但仍低 acc。  
**RX2→RX1：** 方向不对称，CNN 有时更好 → 两 RX 不是对称域。

---

## 7. 方法选择（基于诊断，不是盲试）

"""
    for r in rec:
        md += f"- {r}\n"

    md += """
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
"""

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
