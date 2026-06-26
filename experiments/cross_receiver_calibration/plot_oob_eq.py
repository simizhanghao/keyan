#!/usr/bin/env python3
"""Plot OOB-Eq quick probe and shot curves."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_probe(out_dir: Path) -> None:
    rows = load_csv(out_dir / "probe_before_after.csv")
    reprs = sorted(set(r["repr"] for r in rows))
    methods = [m for m in ["mean_shift", "std_alignment", "coral"] if any(r["eq_method"] == m for r in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
        axes,
        ["receiver_probe_acc", "device_probe_acc"],
        ["Receiver probe (lower = less RX entanglement)", "Device probe (higher = more separable)"],
    ):
        x = np.arange(len(reprs))
        width = 0.18
        before = [float(next(r[metric] for r in rows if r["repr"] == rp and r["phase"] == "before")) * 100 for rp in reprs]
        ax.bar(x - width, before, width, label="before", color="gray")
        for i, method in enumerate(methods):
            after = []
            for rp in reprs:
                val = next((r for r in rows if r["repr"] == rp and r["eq_method"] == method and r["phase"] == "after"), None)
                after.append(float(val[metric]) * 100 if val else 0)
            ax.bar(x + i * width, after, width, label=f"after {method}")
        ax.set_xticks(x)
        ax.set_xticklabels(reprs, rotation=15)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("OOB representation equalization — probe before/after (RX1→RX2 quick)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_oob_eq_probe_before_after.pdf")
    fig.savefig(out_dir / "fig_oob_eq_probe_before_after.png", dpi=150)
    plt.close(fig)


def plot_shot_curve(out_dir: Path) -> None:
    summary = load_csv(out_dir / "summary_oob_eq_quick.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for eq_method, color, label in [
        ("none", "gray", "RCPA-T baseline"),
        ("mean_shift", "C0", "mean_shift + RCPA-T"),
        ("std_alignment", "C1", "std_alignment + RCPA-T"),
        ("coral", "C2", "CORAL + RCPA-T"),
    ]:
        pts = sorted(
            [(int(r["shot_k"]), float(r["file_acc"]) * 100)
             for r in summary
             if r["repr"] == "fused" and r["eval_method"] == ("RCPA-T" if eq_method == "none" else "oob_eq_RCPA-T")
             and r["eq_method"] == eq_method and int(r["shot_k"]) > 0],
            key=lambda x: x[0],
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.set_xlabel("K (labeled calibration windows per device)")
    ax.set_ylabel("File-level accuracy (%)")
    ax.set_title("OOB-Eq + RCPA-T vs baseline (fused repr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_oob_eq_shot_curve_quick.pdf")
    fig.savefig(out_dir / "fig_oob_eq_shot_curve_quick.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_probe(out_dir)
    plot_shot_curve(out_dir)
    print(f"Saved figures in {out_dir}")


if __name__ == "__main__":
    main()
