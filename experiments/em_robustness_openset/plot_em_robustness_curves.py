#!/usr/bin/env python3
"""Plot Chapter 5 EM robustness and open-set figures (IEEE / thesis style)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs/thesis_chapter5_em_openset/figures"

C_MAIN = "#4C78A8"
C_GRAY = "#888888"
C_BLACK = "#222222"
C_BAR = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]


def ieee_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.labelsize": 8.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_baseline_pct(rows: list[dict], ptype: str, clean_val: float) -> float:
    for r in rows:
        if r["perturb_type"] == ptype and abs(float(r["strength"]) - clean_val) < 1e-6:
            return float(r["file_acc"]) * 100
    return 83.33


def plot_robustness_curves(em_dir: Path, out_dir: Path) -> None:
    ieee_rcparams()
    panels = [
        ("AWGN", "awgn_snr_db_sweep.csv", "awgn_snr_db", 100.0, "SNR (dB)", lambda x: x),
        ("CFO", "cfo_norm_sweep.csv", "cfo_norm", 0.0, "CFO norm", lambda x: x),
        ("NBI", "narrowband_sir_db_sweep.csv", "narrowband_sir_db", 30.0, "SIR (dB)", lambda x: x),
        ("Phase noise", "phase_noise_std_sweep.csv", "phase_noise_std", 0.0, "σ", lambda x: x),
        ("IQ imbalance", "iq_amp_db_sweep.csv", "iq_amp_db", 0.0, "Amp (dB)", lambda x: x),
        ("Filter drift", "filter_tilt_norm_sweep.csv", "filter_tilt_norm", 0.0, "Tilt norm", lambda x: x),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6))
    axes_flat = axes.flatten()
    clean_ref = None

    for i, (title, fname, ptype, clean_s, xlab, xf) in enumerate(panels):
        ax = axes_flat[i]
        path = em_dir / fname
        if not path.exists():
            ax.set_visible(False)
            continue
        rows = read_csv(path)
        xs = [xf(float(r["strength"])) for r in rows]
        ys = [float(r["file_acc"]) * 100 for r in rows]
        if clean_ref is None:
            clean_ref = clean_baseline_pct(rows, ptype, clean_s)
        ax.plot(xs, ys, "o-", color=C_MAIN, markersize=3.5, linewidth=1.2)
        ax.axhline(clean_ref, color=C_GRAY, linestyle="--", linewidth=0.9, label="Clean")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("File-level accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Mixed stress bar chart
    ax = axes_flat[6]
    mixed_path = em_dir / "mixed_stress_sweep.csv"
    if mixed_path.exists():
        rows = read_csv(mixed_path)
        labels = [r.get("preset", r["strength"]) for r in rows]
        ys = [float(r["file_acc"]) * 100 for r in rows]
        ax.bar(range(len(labels)), ys, color=C_MAIN, width=0.6, edgecolor=C_BLACK, linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Mixed stress")
        ax.set_ylabel("File-level accuracy (%)")
        if clean_ref:
            ax.axhline(clean_ref, color=C_GRAY, linestyle="--", linewidth=0.9)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.set_visible(False)

    axes_flat[7].set_visible(False)
    fig.subplots_adjust(wspace=0.42, hspace=0.55, left=0.08, right=0.98, top=0.92, bottom=0.14)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig5_1_em_robustness_curves.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def plot_openset(openset_dir: Path, out_dir: Path) -> None:
    ieee_rcparams()
    rows: list[dict] = []
    for seed in range(3):
        p = openset_dir / f"openset_seed{seed}.csv"
        if p.exists():
            for r in read_csv(p):
                rows.append(r)

    scorers = ["msp", "energy", "proto_dist", "mahalanobis"]
    labels = ["MSP", "Energy", "Prototype", "Mahalanobis"]
    metrics = ["auroc", "eer", "known_acc"]
    metric_labels = ["AUROC", "EER", "Known acc (%)"]

    agg: dict[str, dict[str, float]] = {s: {} for s in scorers}
    for s in scorers:
        sub = [r for r in rows if r["scorer"] == s]
        if not sub:
            continue
        agg[s]["auroc"] = np.mean([float(r["auroc"]) for r in sub])
        agg[s]["eer"] = np.mean([float(r["eer"]) for r in sub])
        agg[s]["known_acc"] = np.mean([float(r["known_acc"]) for r in sub]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))
    x = np.arange(len(scorers))
    width = 0.55
    for ax, metric, ylab in zip(axes, metrics, metric_labels):
        vals = [agg[s].get(metric, 0) for s in scorers]
        bars = ax.bar(x, vals, width=width, color=C_BAR[:len(scorers)], edgecolor=C_BLACK, linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05 if metric != "known_acc" else 105)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.98, bottom=0.22, top=0.92)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig5_2_openset_clean.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def plot_stress_ranking(em_dir: Path, out_dir: Path) -> None:
    ieee_rcparams()
    path = em_dir / "em_robustness_by_perturbation.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    families = [r["perturbation_family"] for r in rows]
    avg_rob = [float(r["avg_robust_acc_pct"]) for r in rows]
    drops = [float(r["accuracy_drop_pp"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    x = np.arange(len(families))
    axes[0].barh(x, avg_rob, color=C_MAIN, height=0.55, edgecolor=C_BLACK, linewidth=0.4)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(families)
    axes[0].set_xlabel("Average robust accuracy (%)")
    axes[0].set_xlim(0, 100)
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].barh(x, drops, color="#E45756", height=0.55, edgecolor=C_BLACK, linewidth=0.4)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(families)
    axes[1].set_xlabel("Accuracy drop (pp)")
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    if drops:
        worst = families[int(np.argmax(drops))]
        axes[1].annotate(f"Most destructive: {worst}", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7)

    fig.subplots_adjust(wspace=0.45, left=0.22, right=0.98, top=0.92, bottom=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig5_3_em_stress_ranking.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--em-dir", type=Path, default=ROOT / "experiments/em_robustness_openset/results/em_full_20260628")
    p.add_argument(
        "--openset-dir",
        type=Path,
        default=ROOT / "experiments/em_robustness_openset/results/openset_full_20260628_1123",
    )
    p.add_argument("--out-dir", type=Path, default=FIG_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_robustness_curves(args.em_dir, args.out_dir)
    plot_openset(args.openset_dir, args.out_dir)
    plot_stress_ranking(args.em_dir, args.out_dir)
    print(f"Figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
