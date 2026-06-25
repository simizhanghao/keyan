#!/usr/bin/env python3
"""Generate paper figures from final_tables CSVs into outputs/paper_ready_v3/final_figures/."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "paper_ready_v3" / "final_tables"
OUT = ROOT / "outputs" / "paper_ready_v3" / "final_figures"
DOCS_FIG = ROOT / "docs" / "iotj_paper" / "figures"

# IEEE-friendly styling
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def read_csv(name: str) -> list[dict]:
    with (TABLE_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem: str) -> None:
    for d in (OUT, DOCS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf")
        fig.savefig(d / f"{stem}.png")
    plt.close(fig)
    print(f"Wrote {stem}.pdf/png")


def fig2_cross_day_seed_bars() -> None:
    seeds = [0, 1, 2, 3, 4]
    cnn = [62.5, 41.7, 62.5, 33.3, 70.8]
    hybrid = [83.3, 70.8, 70.8, 79.2, 70.8]
    x = range(len(seeds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], hybrid, width=w, label="F Hybrid", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed{s}" for s in seeds])
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cross-Day File-Level Accuracy by Seed")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig2_cross_day_seed_bars")


def fig3_fusion_chirp_ablation() -> None:
    rows = read_csv("table2_fusion_chirp_ablation.csv")
    labels = []
    vals = []
    errs = []
    for r in rows:
        labels.append(f"{r['fusion']}\n{'+chirp' if r['chirp']=='yes' else '-chirp'}")
        vals.append(float(r["file_acc_mean_pct"]))
        errs.append(float(r["file_acc_std_pct"]))
    colors = ["#C44E52", "#C44E52", "#55A868", "#55A868"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.bar(labels, vals, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 90)
    ax.set_title("Fusion/Chirp Ablation (Cross-Day)")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig3_fusion_chirp_ablation")


def fig4_distance_shift() -> None:
    rows = [r for r in read_csv("table3_deployment_shift.csv") if r["shift_type"] == "distance LOCO"]
    rows.sort(key=lambda r: int(r["condition"].split("m")[0]))
    labels = [r["condition"].replace(" held-out", "") for r in rows]
    cnn = [float(r["cnn_file_acc_pct"]) for r in rows]
    hyb = [float(r["hybrid_file_acc_pct"]) for r in rows]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], hyb, width=w, label="F Hybrid", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("File-Acc (%)")
    ax.set_title("Distance LOCO (Held-Out Bin)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig4_distance_shift")


def fig5_cross_receiver_stress() -> None:
    rows = read_csv("table4_cross_receiver_stress.csv")
    dirs = ["RX1→RX2", "RX2→RX1"]
    cnn, hyb, cnn_e, hyb_e = [], [], [], []
    for d in dirs:
        cnn_r = next(r for r in rows if r["direction"] == d and "CNN" in r["model"])
        hyb_r = next(r for r in rows if r["direction"] == d and "Hybrid" in r["model"])
        cnn.append(float(cnn_r["file_acc_mean_pct"]))
        hyb.append(float(hyb_r["file_acc_mean_pct"]))
        cnn_e.append(float(cnn_r["file_acc_std_pct"]))
        hyb_e.append(float(hyb_r["file_acc_std_pct"]))
    x = range(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.bar([i - w / 2 for i in x], cnn, width=w, yerr=cnn_e, capsize=3, label="CNN-IQ", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], hyb, width=w, yerr=hyb_e, capsize=3, label="F Hybrid", color="#DD8452")
    ax.axhline(4.17, color="gray", linestyle="--", linewidth=0.8, label="chance (4.17%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dirs)
    ax.set_ylabel("File-Acc (%)")
    ax.set_title("Cross-Receiver Stress Test")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig5_cross_receiver_stress")


def _box(ax, xy, wh, text, fc="#EAF2FF"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.0, edgecolor="#333333", facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def fig1_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.02, 0.35), (0.12, 0.30), "IQ / FFT /\nAP / OOB\nviews")
    _box(ax, (0.18, 0.55), (0.14, 0.22), "CNN-stem\npatch tokens", "#FFF4E6")
    _box(ax, (0.18, 0.15), (0.14, 0.22), "OOB branch\npatch tokens", "#FFE6E6")
    _box(ax, (0.36, 0.35), (0.16, 0.30), "OOB-guided\ncross-attention\nfusion", "#E8FFE8")
    _box(ax, (0.56, 0.35), (0.14, 0.30), "RF-HSTU\nsequence\nencoder", "#F0E8FF")
    _box(ax, (0.73, 0.35), (0.10, 0.30), "Chirp\nembedding", "#F5F5F5")
    _box(ax, (0.86, 0.35), (0.10, 0.30), "Linear\nclassifier", "#F5F5F5")
    _box(ax, (0.86, 0.05), (0.10, 0.18), "File-level\nmean-logits\nvoting", "#EEEEEE")

    arrows = [
        ((0.14, 0.50), (0.18, 0.66)),
        ((0.14, 0.50), (0.18, 0.26)),
        ((0.32, 0.66), (0.36, 0.50)),
        ((0.32, 0.26), (0.36, 0.42)),
        ((0.52, 0.50), (0.56, 0.50)),
        ((0.70, 0.50), (0.73, 0.50)),
        ((0.83, 0.50), (0.86, 0.50)),
        ((0.91, 0.35), (0.91, 0.23)),
    ]
    for p0, p1 in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                     linewidth=0.9, color="#444444"))

    ax.text(0.5, 0.92, "OOB-Guided Cross-Attentive RF-HSTU Hybrid (draft block diagram)",
            ha="center", va="center", fontsize=10, fontweight="bold")
    save(fig, "fig1_model_architecture")


def main() -> None:
    fig1_model_architecture()
    fig2_cross_day_seed_bars()
    fig3_fusion_chirp_ablation()
    fig4_distance_shift()
    fig5_cross_receiver_stress()
    print(f"All figures written to {OUT} and {DOCS_FIG}")


if __name__ == "__main__":
    main()
