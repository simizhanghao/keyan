#!/usr/bin/env python3
"""Generate paper figures from final_tables CSVs into outputs/paper_ready_v3/final_figures/."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "paper_ready_v3" / "final_tables"
OUT = ROOT / "outputs" / "paper_ready_v3" / "final_figures"
DOCS_FIG = ROOT / "docs" / "iotj_paper" / "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
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


def _save(fig, stem: str, *, pad: float = 0.05) -> None:
    for d in (OUT, DOCS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"Wrote {stem}.pdf/png")


def _box(ax, x, y, w, h, text, fc=COL_MAIN, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.9, edgecolor=EDGE, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, p0, p1, style="-|>", color=ARROW, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, shrinkA=3, shrinkB=3,
    ))


def fig1_model_architecture() -> None:
    """Full-width dual-branch architecture with safe right margin."""
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    bw, bh = 1.35, 0.95
    main_y, oob_y = 3.55, 1.05

    # Main RF branch
    ax.text(0.25, 4.75, "Main RF branch", fontsize=10, fontweight="bold", color="#333")
    main_xs = [0.4, 2.15, 3.9, 5.65]
    main_labels = [
        "IQ/FFT/AP\nviews",
        "CNN\nstem",
        "RF patch\ntokens",
        "Pos./chirp\nembedding",
    ]
    for x, lab in zip(main_xs, main_labels):
        _box(ax, x, main_y, bw, bh, lab, COL_MAIN)

    # OOB branch
    ax.text(0.25, 2.25, "OOB branch", fontsize=10, fontweight="bold", color="#333")
    oob_xs = [0.4, 2.15, 3.9]
    oob_labels = ["OOB\nspectrum", "OOB\ntokenizer", "OOB\ntokens"]
    for x, lab in zip(oob_xs, oob_labels):
        _box(ax, x, oob_y, bw, bh, lab, COL_OOB)

    # Fusion and output (right side with margin)
    fuse_xs = [7.35, 9.05, 10.75, 12.35, 14.05]
    fuse_labels = [
        "OOB-guided\ncross-attention",
        "Gated\nresidual",
        "RF-HSTU\nencoder",
        "Device\nclassifier",
        "File-level\nmean logits",
    ]
    fuse_colors = [COL_FUSE, COL_FUSE, COL_SEQ, COL_OUT, COL_OUT]
    fuse_ys = [2.3, 2.3, 2.3, 3.15, 1.45]
    fuse_hs = [1.45, 1.45, 1.45, 0.95, 0.95]
    for x, lab, fc, fy, fh in zip(fuse_xs, fuse_labels, fuse_colors, fuse_ys, fuse_hs):
        _box(ax, x, fy, bw, fh, lab, fc)

    # Main branch horizontal flow
    for i in range(len(main_xs) - 1):
        x0 = main_xs[i] + bw
        x1 = main_xs[i + 1]
        cy = main_y + bh / 2
        _arrow(ax, (x0, cy), (x1, cy))

    # OOB branch horizontal flow
    for i in range(len(oob_xs) - 1):
        x0 = oob_xs[i] + bw
        x1 = oob_xs[i + 1]
        cy = oob_y + bh / 2
        _arrow(ax, (x0, cy), (x1, cy))

    # Q from main tokens to cross-attention
    q_src = (main_xs[-1] + bw, main_y + bh / 2)
    q_dst = (fuse_xs[0], 2.3 + 1.45 / 2 + 0.15)
    _arrow(ax, q_src, q_dst)
    ax.text(6.55, 3.35, "Q", fontsize=9, fontweight="bold", color="#1a5276")

    # K,V from OOB tokens
    kv_src = (oob_xs[-1] + bw, oob_y + bh / 2)
    kv_dst = (fuse_xs[0], 2.3 + 1.45 / 2 - 0.15)
    _arrow(ax, kv_src, kv_dst)
    ax.text(6.55, 1.85, "K, V", fontsize=9, fontweight="bold", color="#922b21")

    # Fusion chain
    chain = [
        ((fuse_xs[0] + bw, 2.3 + 1.45 / 2), (fuse_xs[1], 2.3 + 1.45 / 2)),
        ((fuse_xs[1] + bw, 2.3 + 1.45 / 2), (fuse_xs[2], 2.3 + 1.45 / 2)),
        ((fuse_xs[2] + bw, 2.3 + 1.45 / 2), (fuse_xs[3], 3.15 + 0.95 / 2)),
        ((fuse_xs[3] + bw / 2, 3.15), (fuse_xs[4] + bw / 2, 1.45 + 0.95)),
    ]
    for p0, p1 in chain:
        _arrow(ax, p0, p1)

    _save(fig, "fig1_model_architecture", pad=0.08)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)


def _plot_cross_day(ax):
    seeds = [0, 1, 2, 3, 4]
    cnn = [62.5, 41.7, 62.5, 33.3, 70.8]
    hybrid = [83.3, 70.8, 70.8, 79.2, 70.8]
    x = range(len(seeds))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0", edgecolor="none")
    ax.bar([i + w / 2 for i in x], hybrid, width=w, label="Hybrid", color="#DD8452", edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed{s}" for s in seeds], fontsize=7.5)
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Cross-day seeds", fontsize=10, pad=5)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    _style_axis(ax)


def _plot_fusion(ax):
    rows = read_csv("table2_fusion_chirp_ablation.csv")
    label_map = {
        ("concat", "no"): "Concat\nw/o chirp",
        ("concat", "yes"): "Concat\n+ chirp",
        ("cross-attn", "no"): "X-Attn\nw/o chirp",
        ("cross-attn", "yes"): "X-Attn\n+ chirp",
    }
    labels, vals, errs = [], [], []
    for r in rows:
        key = (r["fusion"], r["chirp"])
        labels.append(label_map.get(key, f"{r['fusion']}\n{r['chirp']}"))
        vals.append(float(r["file_acc_mean_pct"]))
        errs.append(float(r["file_acc_std_pct"]))
    colors = ["#C44E52", "#C44E52", "#55A868", "#55A868"]
    ax.bar(labels, vals, yerr=errs, capsize=2.5, color=colors, edgecolor="none", width=0.62)
    ax.set_ylabel("File-Acc (%)")
    ax.set_ylim(0, 90)
    ax.set_title("(b) Fusion/chirp ablation", fontsize=10, pad=5)
    ax.tick_params(axis="x", labelsize=7.5)
    _style_axis(ax)


def _plot_distance(ax):
    rows = [r for r in read_csv("table3_deployment_shift.csv") if r["shift_type"] == "distance LOCO"]
    rows.sort(key=lambda r: int(r["condition"].split("m")[0]))
    labels = [r["condition"].replace(" held-out", "") for r in rows]
    cnn = [float(r["cnn_file_acc_pct"]) for r in rows]
    hyb = [float(r["hybrid_file_acc_pct"]) for r in rows]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color="#4C72B0", edgecolor="none")
    ax.bar([i + w / 2 for i in x], hyb, width=w, label="Hybrid", color="#DD8452", edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("File-Acc (%)")
    ax.set_title("(c) Distance LOCO", fontsize=10, pad=5)
    ax.legend(frameon=False, fontsize=7.5)
    _style_axis(ax)


def fig_results_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.8))
    _plot_cross_day(axes[0])
    _plot_fusion(axes[1])
    _plot_distance(axes[2])
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.98, top=0.88, bottom=0.22)
    _save(fig, "fig_results_summary", pad=0.06)


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
           label="Hybrid", color="#DD8452", edgecolor="none")
    ax.axhline(4.17, color="gray", linestyle="--", linewidth=0.7, label="chance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dirs)
    ax.set_ylabel("File-Acc (%)")
    ax.legend(frameon=False, fontsize=7)
    _style_axis(ax)
    _save(fig, "fig5_cross_receiver_stress")


def main() -> None:
    fig1_model_architecture()
    fig_results_summary()
    fig5_cross_receiver_stress()
    print(f"All figures written to {OUT} and {DOCS_FIG}")


if __name__ == "__main__":
    main()
