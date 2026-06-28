#!/usr/bin/env python3
"""Plot open-set under EM from summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    conds = sorted({r["condition"] for r in rows})
    x = np.arange(len(conds))
    width = 0.35
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    for metric, fname in [
        ("auroc", "fig_auroc_under_em"),
        ("eer", "fig_eer_under_em"),
        ("known_acc", "fig_known_acc_under_em"),
    ]:
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        for i, scorer in enumerate(("proto_dist", "mahalanobis")):
            vals = []
            for c in conds:
                sub = [r for r in rows if r["condition"] == c and r["scorer"] == scorer]
                vals.append(float(np.mean([float(r[metric]) for r in sub])) if sub else 0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=scorer.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=35, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(args.out_dir / f"{fname}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)
    print(f"Wrote figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
