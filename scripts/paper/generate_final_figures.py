#!/usr/bin/env python3
"""Generate result summary figure (Fig.2) for IoTJ paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "paper_ready_v3" / "final_tables"
OUT = ROOT / "outputs" / "paper_ready_v3" / "final_figures"
DOCS_FIG = ROOT / "docs" / "iotj_paper" / "figures"

FS = 8.5
plt.rcParams.update({
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
})

CNN_COLOR = "#3B5B8C"
HYB_COLOR = "#B85C38"
GRID_ALPHA = 0.15


def read_csv(name: str) -> list[dict]:
    with (TABLE_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, stem: str, *, pad: float = 0.03) -> None:
    for d in (OUT, DOCS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"Wrote {stem}.pdf/png")


def _style_axis(ax, *, ylabel: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.4)
    ax.tick_params(length=2.8, width=0.55, direction="out")
    if ylabel:
        ax.set_ylabel("File-Acc (%)")
    else:
        ax.set_yticklabels([])


def _plot_cross_day(ax, *, show_legend: bool = True):
    seeds = [0, 1, 2, 3, 4]
    cnn = [62.5, 41.7, 62.5, 33.3, 70.8]
    hybrid = [83.3, 70.8, 70.8, 79.2, 70.8]
    x = range(len(seeds))
    w = 0.36
    ax.bar([i - w / 2 for i in x], cnn, width=w, label="CNN-IQ", color=CNN_COLOR, edgecolor="none")
    ax.bar([i + w / 2 for i in x], hybrid, width=w, label="Hybrid", color=HYB_COLOR, edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed{s}" for s in seeds])
    ax.set_ylim(0, 100)
    ax.set_title("(a) Cross-day stability", fontsize=9, pad=3)
    if show_legend:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  ncol=2, fontsize=8, handlelength=1.1, columnspacing=0.9)
    _style_axis(ax, ylabel=True)


def _plot_fusion(ax):
    rows = read_csv("table2_fusion_chirp_ablation.csv")
    label_map = {
        ("concat", "no"): "Concat\nno chirp",
        ("concat", "yes"): "Concat\n+ chirp",
        ("cross-attn", "no"): "Cross-attn\nno chirp",
        ("cross-attn", "yes"): "Cross-attn\n+ chirp",
    }
    labels, vals, errs = [], [], []
    for r in rows:
        key = (r["fusion"], r["chirp"])
        labels.append(label_map.get(key, f"{r['fusion']}\n{r['chirp']}"))
        vals.append(float(r["file_acc_mean_pct"]))
        errs.append(float(r["file_acc_std_pct"]))
    colors = ["#A8483C", "#A8483C", "#2F6B4F", "#2F6B4F"]
    ax.bar(labels, vals, yerr=errs, capsize=2.0, color=colors, edgecolor="none", width=0.62)
    ax.set_ylim(0, 90)
    ax.set_title("(b) Fusion/chirp ablation", fontsize=9, pad=3)
    ax.tick_params(axis="x", labelsize=7.2)
    _style_axis(ax, ylabel=False)


def _plot_distance(ax):
    rows = [r for r in read_csv("table3_deployment_shift.csv") if r["shift_type"] == "distance LOCO"]
    rows.sort(key=lambda r: int(r["condition"].split("m")[0]))
    labels = [r["condition"].replace(" held-out", "") for r in rows]
    cnn = [float(r["cnn_file_acc_pct"]) for r in rows]
    hyb = [float(r["hybrid_file_acc_pct"]) for r in rows]
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], cnn, width=w, color=CNN_COLOR, edgecolor="none")
    ax.bar([i + w / 2 for i in x], hyb, width=w, color=HYB_COLOR, edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 40)
    ax.set_title("(c) Distance shift", fontsize=9, pad=3)
    _style_axis(ax, ylabel=False)


def fig_results_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.35))
    _plot_cross_day(axes[0], show_legend=True)
    _plot_fusion(axes[1])
    _plot_distance(axes[2])
    fig.subplots_adjust(wspace=0.22, left=0.09, right=0.99, top=0.78, bottom=0.24)
    _save(fig, "fig_results_summary", pad=0.03)


def main() -> None:
    fig_results_summary()
    print(f"Figures written to {OUT} and {DOCS_FIG}")


if __name__ == "__main__":
    main()
