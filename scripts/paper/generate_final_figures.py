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
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

COL_MAIN = "#D6E4F0"
COL_OOB = "#F5D6D6"
COL_FUSE = "#D8ECD8"
COL_SEQ = "#E8DDF5"
COL_OUT = "#EFEFEF"
EDGE = "#333333"
ARROW = "#444444"


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


def _box(ax, xy, wh, text, fc=COL_MAIN, fontsize=7.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8, edgecolor=EDGE, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, p0, p1, style="-|>", color=ARROW, lw=0.85):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    ))


def fig1_model_architecture() -> None:
    """Professional dual-branch architecture diagram (full-width)."""
    fig, ax = plt.subplots(figsize=(10.5, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main RF branch (top)
    ax.text(0.02, 0.88, "Main RF branch", fontsize=8, fontweight="bold", color="#333")
    _box(ax, (0.02, 0.58), (0.10, 0.22), "IQ / FFT /\nAP views", COL_MAIN)
    _box(ax, (0.14, 0.58), (0.10, 0.22), "CNN\nstem", COL_MAIN)
    _box(ax, (0.26, 0.58), (0.11, 0.22), "RF patch\ntokens", COL_MAIN)
    _box(ax, (0.39, 0.58), (0.12, 0.22), "+ pos/chirp\nembedding", COL_OUT)

    # OOB branch (bottom)
    ax.text(0.02, 0.48, "OOB branch", fontsize=8, fontweight="bold", color="#333")
    _box(ax, (0.02, 0.18), (0.10, 0.22), "OOB\nspectral", COL_OOB)
    _box(ax, (0.14, 0.18), (0.10, 0.22), "OOB\ntokenizer", COL_OOB)
    _box(ax, (0.26, 0.18), (0.11, 0.22), "OOB\ntokens", COL_OOB)

    # Cross-attention fusion
    _box(ax, (0.54, 0.38), (0.14, 0.32), "Cross-attention\n(Q, K, V)", COL_FUSE)
    _box(ax, (0.70, 0.38), (0.10, 0.32), "Gated\nresidual", COL_FUSE)

    # Sequence + output
    _box(ax, (0.82, 0.38), (0.10, 0.32), "RF-HSTU\nblocks", COL_SEQ)
    _box(ax, (0.94, 0.50), (0.05, 0.20), "Class-\nifier", COL_OUT)
    _box(ax, (0.94, 0.22), (0.05, 0.20), "Mean-\nlogits", COL_OUT)

    # Main branch arrows
    for x0, x1 in [(0.12, 0.14), (0.24, 0.26), (0.37, 0.39)]:
        _arrow(ax, (x0, 0.69), (x1, 0.69))
    _arrow(ax, (0.51, 0.69), (0.54, 0.60))
    ax.text(0.525, 0.72, "Q", fontsize=7, fontweight="bold", color="#1a5276")

    # OOB branch arrows
    for x0, x1 in [(0.12, 0.14), (0.24, 0.26)]:
        _arrow(ax, (x0, 0.29), (x1, 0.29))
    _arrow(ax, (0.37, 0.29), (0.54, 0.42))
    ax.text(0.44, 0.32, "K, V", fontsize=7, fontweight="bold", color="#922b21")

    # Fusion -> output
    _arrow(ax, (0.68, 0.54), (0.70, 0.54))
    _arrow(ax, (0.80, 0.54), (0.82, 0.54))
    _arrow(ax, (0.92, 0.54), (0.94, 0.60))
    _arrow(ax, (0.965, 0.50), (0.965, 0.42))

    save(fig, "fig1_model_architecture")


def _plot_cross_day(ax):
    seeds = [0, 1, 2, 3, 4]
    cnn = [62.5, 41.7, 62.5, 33.3, 70.8]
    hybrid = [83.3, 70.8, 70.8, 79.2, 70.8]
    x = range(len(seeds))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0", edgecolor="none")
    ax.bar([i + w / 2 for i in x], hybrid, width=w, label="F", color="#DD8452", edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Cross-day seeds", fontsize=9, pad=4)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_fusion(ax):
    rows = read_csv("table2_fusion_chirp_ablation.csv")
    labels, vals, errs = [], [], []
    for r in rows:
        labels.append(f"{r['fusion'][:5]}\n{'+c' if r['chirp'] == 'yes' else '-c'}")
        vals.append(float(r["file_acc_mean_pct"]))
        errs.append(float(r["file_acc_std_pct"]))
    colors = ["#C44E52", "#C44E52", "#55A868", "#55A868"]
    ax.bar(labels, vals, yerr=errs, capsize=2, color=colors, edgecolor="none", width=0.65)
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 90)
    ax.set_title("(b) Fusion/chirp ablation", fontsize=9, pad=4)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_distance(ax):
    rows = [r for r in read_csv("table3_deployment_shift.csv") if r["shift_type"] == "distance LOCO"]
    rows.sort(key=lambda r: int(r["condition"].split("m")[0]))
    labels = [r["condition"].replace(" held-out", "") for r in rows]
    cnn = [float(r["cnn_file_acc_pct"]) for r in rows]
    hyb = [float(r["hybrid_file_acc_pct"]) for r in rows]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0", edgecolor="none")
    ax.bar([i + w / 2 for i in x], hyb, width=w, label="F", color="#DD8452", edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("File-Acc (%)")
    ax.set_title("(c) Distance LOCO", fontsize=9, pad=4)
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_results_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6))
    _plot_cross_day(axes[0])
    _plot_fusion(axes[1])
    _plot_distance(axes[2])
    fig.subplots_adjust(wspace=0.32, left=0.06, right=0.99, top=0.88, bottom=0.18)
    save(fig, "fig_results_summary")


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
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.bar([i - w / 2 for i in x], cnn, width=w, yerr=cnn_e, capsize=2,
           label="CNN-IQ", color="#4C72B0", edgecolor="none")
    ax.bar([i + w / 2 for i in x], hyb, width=w, yerr=hyb_e, capsize=2,
           label="F", color="#DD8452", edgecolor="none")
    ax.axhline(4.17, color="gray", linestyle="--", linewidth=0.7, label="chance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dirs)
    ax.set_ylabel("File-Acc (%)")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig5_cross_receiver_stress")


def main() -> None:
    fig1_model_architecture()
    fig_results_summary()
    fig5_cross_receiver_stress()
    print(f"All figures written to {OUT} and {DOCS_FIG}")


if __name__ == "__main__":
    main()
