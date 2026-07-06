#!/usr/bin/env python3
"""Plot file-level confusion matrices for cross-receiver transfer."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", required=True, help="Root with */file_predictions.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-classes", type=int, default=24)
    return p.parse_args()


def load_file_preds(path: Path) -> tuple[list[int], list[int]]:
    labels, preds = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
    return labels, preds


def confusion_matrix(labels: list[int], preds: list[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    return cm


def collapse_stats(cm: np.ndarray) -> dict:
    preds = cm.argmax(axis=0)  # not used
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pred_dist = col_sums / max(col_sums.sum(), 1)
    top_pred_frac = float(col_sums.max() / max(col_sums.sum(), 1))
    top3_pred_frac = float(np.sort(col_sums)[-3:].sum() / max(col_sums.sum(), 1))
    return {
        "top1_pred_mass": top_pred_frac,
        "top3_pred_mass": top3_pred_frac,
        "num_active_pred_classes": int((col_sums > 0).sum()),
        "entropy_pred_distribution": float(-np.sum(pred_dist[pred_dist > 0] * np.log(pred_dist[pred_dist > 0]))),
    }


def plot_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.pred_root)

    collapse_rows = []
    cms = []

    for pred_path in sorted(root.glob("**/file_predictions.csv")):
        rel = pred_path.parent.relative_to(root)
        labels, preds = load_file_preds(pred_path)
        cm = confusion_matrix(labels, preds, args.num_classes)
        cms.append(cm)
        title = str(rel)
        plot_cm(cm, title, out_dir / f"confusion_{rel.as_posix().replace('/', '_')}.png")

        stats = collapse_stats(cm)
        stats["experiment"] = str(rel)
        stats["file_acc"] = float(np.trace(cm) / max(cm.sum(), 1))
        collapse_rows.append(stats)

    # mean confusion across seeds if multiple
    if cms:
        mean_cm = np.mean(np.stack(cms), axis=0)
        plot_cm(mean_cm.astype(int), "Mean confusion (all runs)", out_dir / "confusion_mean.png")

    collapse_path = out_dir / "collapse_summary.csv"
    with collapse_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(collapse_rows[0].keys()) if collapse_rows else [])
        if collapse_rows:
            w.writeheader()
            w.writerows(collapse_rows)

    print(f"Saved confusion plots to {out_dir}")


if __name__ == "__main__":
    main()
