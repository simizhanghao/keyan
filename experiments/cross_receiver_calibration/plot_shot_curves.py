#!/usr/bin/env python3
"""Plot RCPA shot curves from summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--out-png", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cls_row = next(r for r in rows if r["method"] == "source_classifier")
    cls_acc = float(cls_row["file_acc"]) * 100

    curves = {
        "RCPA-S (source proto)": "RCPA-S",
        "RCPA-T (target proto)": "RCPA-T",
        "RCPA-B (alpha=0.5)": "RCPA-B",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(cls_acc, color="gray", ls="--", label=f"Source classifier ({cls_acc:.1f}%)")

    for label, method in curves.items():
        pts = sorted(
            [(int(r["shot_k"]), float(r["file_acc"]) * 100) for r in rows if r["method"] == method],
            key=lambda x: x[0],
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("K (labeled calibration windows per device)")
    ax.set_ylabel("File-level accuracy (%)")
    ax.set_title("RCPA quick mode — RX1→RX2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    if args.out_png:
        fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
