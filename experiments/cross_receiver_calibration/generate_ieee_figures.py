#!/usr/bin/env python3
"""Generate IEEE-style vector figures for Paper 2 manuscript."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "experiments/cross_receiver_calibration"
FIG = ROOT / "docs/paper2_rcpa/figures"

# Colorblind-friendly palette
C_MAIN = "#4C78A8"
C_OOB = "#F58518"
C_FUSED = "#54A24B"
C_CNN = "#B279A2"
C_RX1 = "#4C78A8"
C_RX2 = "#F58518"
C_GRAY = "#888888"
C_BLACK = "#222222"


def ieee_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )


def bar_labels(ax, bars, fmt="{:.1f}", dy=1.5) -> None:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def load_main_table() -> list[dict]:
    with (CAL / "results/paper2_main/paper2_main_table.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_mean_std(s: str) -> tuple[float, float]:
    parts = s.split("±")
    m = float(parts[0].strip())
    sd = float(parts[1].strip()) if len(parts) > 1 else 0.0
    return m, sd


def fig1_diagnosis_ieee() -> None:
    ieee_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.2))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    # (a) Receiver probe
    ax = axes[0, 0]
    paths = ["Main", "OOB", "Fused"]
    vals = [62.4, 72.7, 50.2]
    colors = [C_MAIN, C_OOB, C_FUSED]
    bars = ax.bar(paths, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    bar_labels(ax, bars)
    ax.set_ylabel("Receiver probe (%)")
    ax.set_ylim(0, 85)
    ax.text(-0.12, 1.02, "(a)", transform=ax.transAxes, fontsize=8.5, fontweight="bold")

    # (b) OOB energy ratio
    ax = axes[0, 1]
    bars = ax.bar([0], [1.44], color=C_OOB, width=0.35, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color=C_GRAY, ls="--", lw=1.0)
    ax.set_xticks([0])
    ax.set_xticklabels(["RX2/RX1"])
    ax.set_ylabel("OOB energy ratio")
    ax.set_ylim(0, 1.8)
    ax.text(0, 1.44 + 0.05, "1.44", ha="center", fontsize=7)
    ax.text(-0.12, 1.02, "(b)", transform=ax.transAxes, fontsize=8.5, fontweight="bold")

    # (c) Distance ratio
    ax = axes[1, 0]
    models = ["CNN-IQ", "Hybrid"]
    ratios = [1.25, 0.22]
    bars = ax.bar(models, ratios, color=[C_CNN, C_FUSED], width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color=C_GRAY, ls="--", lw=1.0)
    ax.set_ylabel("Cross-RX / diff-dev ratio")
    ax.set_ylim(0, 1.45)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.text(-0.12, 1.02, "(c)", transform=ax.transAxes, fontsize=8.5, fontweight="bold")

    # (d) Top-1 mass
    ax = axes[1, 1]
    methods = ["CNN-IQ", "Hybrid"]
    top1 = [95.8, 20.8]
    bars = ax.bar(methods, top1, color=[C_CNN, C_FUSED], width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Top-1 mass (%)")
    ax.set_ylim(0, 105)
    bar_labels(ax, bars, fmt="{:.1f}", dy=2)
    ax.text(-0.12, 1.02, "(d)", transform=ax.transAxes, fontsize=8.5, fontweight="bold")

    out_pdf = FIG / "fig1_diagnosis_summary.pdf"
    out_png = FIG / "fig1_diagnosis_summary.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def fig2_rcpa_shotcurve_ieee() -> None:
    ieee_rcparams()
    main = load_main_table()
    pooled_path = CAL / "results/paper2_main/paper2_rcpa_t_pooled.csv"
    with pooled_path.open(encoding="utf-8") as f:
        pooled = list(csv.DictReader(f))

    ks = [1, 3, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    for direction, color, marker, label in [
        ("rx1_to_rx2", C_RX1, "o", "RX1→RX2"),
        ("rx2_to_rx1", C_RX2, "^", "RX2→RX1"),
    ]:
        ys, stds = [], []
        for k in ks:
            r = next(x for x in main if x["direction"] == direction and "RCPA-T" in x["method"] and x["K"] == str(k))
            m, s = parse_mean_std(r["file_acc"])
            ys.append(m)
            stds.append(s)
        ax.errorbar(
            ks, ys, yerr=stds, marker=marker, color=color, label=label,
            capsize=2, linewidth=1.2, markersize=5, elinewidth=0.8,
        )

    p_ys = [parse_mean_std(next(x for x in pooled if int(x["K"]) == k)["RCPA-T mean±std"])[0] for k in ks]
    ax.plot(ks, p_ys, color=C_BLACK, marker="s", linewidth=1.8, markersize=5, label="Pooled")

    ax.axhline(20.1, color=C_GRAY, ls="--", lw=1.0, label="Source classifier")
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(15, 82)
    ax.set_xlabel("$K$ (labeled windows per device)")
    ax.set_ylabel("File-level accuracy (%)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    out_pdf = FIG / "fig2_rcpa_shotcurve.pdf"
    out_png = FIG / "fig2_rcpa_shotcurve.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_diagnosis_ieee()
    fig2_rcpa_shotcurve_ieee()


if __name__ == "__main__":
    main()
