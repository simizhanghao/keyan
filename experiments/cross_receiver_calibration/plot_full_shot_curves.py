#!/usr/bin/env python3
"""Plot full-mode shot curves by direction and mean."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def load_curve(csv_path: Path, direction: str | None = None) -> dict[str, list[tuple[int, float, float]]]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    curves: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        if direction and r.get("direction") != direction:
            continue
        method = r["method"]
        k = int(r["shot_k"])
        if k < 0:
            continue
        curves.setdefault(method, []).append((k, float(r["mean_file_acc"]) * 100, float(r["std_file_acc"]) * 100))
    for m in curves:
        curves[m].sort(key=lambda x: x[0])
    return curves


def plot_direction(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    titles = {"rx1_to_rx2": "RX1 → RX2", "rx2_to_rx1": "RX2 → RX1"}
    styles = {
        "RCPA-T": ("o", "C0", "RCPA-T (primary)"),
        "RCPA-B": ("s", "C1", "RCPA-B (ablation)"),
        "RCPA-S": ("^", "C2", "RCPA-S (source proto)"),
    }
    for ax, direction in zip(axes, ["rx1_to_rx2", "rx2_to_rx1"]):
        csv_path = out_dir / f"shot_curve_{direction}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cls_rows = [r for r in rows if r["method"] == "source_classifier"]
        if cls_rows:
            cls_acc = float(cls_rows[0]["mean_file_acc"]) * 100
            ax.axhline(cls_acc, color="gray", ls="--", label=f"Source classifier ({cls_acc:.1f}%)")
        for method, (marker, color, label) in styles.items():
            pts = sorted(
                [(int(r["shot_k"]), float(r["mean_file_acc"]) * 100, float(r["std_file_acc"]) * 100)
                 for r in rows if r["method"] == method and int(r["shot_k"]) >= 0],
                key=lambda x: x[0],
            )
            if not pts:
                continue
            xs, ys, stds = zip(*pts)
            ax.errorbar(xs, ys, yerr=stds, marker=marker, color=color, capsize=3, label=label)
        ax.set_xlabel("K (labeled calibration windows per device)")
        ax.set_ylabel("File-level accuracy (%)")
        ax.set_title(titles[direction])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("RCPA full mode — by direction (mean ± std over seeds & splits)", fontsize=11)
    fig.tight_layout()
    pdf = out_dir / "fig_shot_curve_by_direction.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / "fig_shot_curve_by_direction.png", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf}")


def plot_mean(out_dir: Path) -> None:
    csv_path = out_dir / "shot_curve_mean.csv"
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    cls_rows = [r for r in rows if r["method"] == "source_classifier"]
    if cls_rows:
        cls_acc = float(cls_rows[0]["mean_file_acc"]) * 100
        ax.axhline(cls_acc, color="gray", ls="--", label=f"Source classifier ({cls_acc:.1f}%)")
    for method, color, label in [
        ("RCPA-T", "C0", "RCPA-T (primary)"),
        ("RCPA-B", "C1", "RCPA-B (ablation)"),
    ]:
        pts = sorted(
            [(int(r["shot_k"]), float(r["mean_file_acc"]) * 100, float(r["std_file_acc"]) * 100)
             for r in rows if r["method"] == method and int(r["shot_k"]) >= 0],
            key=lambda x: x[0],
        )
        if not pts:
            continue
        xs, ys, stds = zip(*pts)
        ax.errorbar(xs, ys, yerr=stds, marker="o", color=color, capsize=3, label=label)
    ax.set_xlabel("K (labeled calibration windows per device)")
    ax.set_ylabel("File-level accuracy (%)")
    ax.set_title("RCPA full mode — both directions mean")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf = out_dir / "fig_shot_curve_mean.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / "fig_shot_curve_mean.png", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_direction(out_dir)
    plot_mean(out_dir)


if __name__ == "__main__":
    main()
