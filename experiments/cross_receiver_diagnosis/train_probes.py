#!/usr/bin/env python3
"""Train linear probes on frozen embeddings: receiver vs device."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--path", default="fused", choices=["main", "oob", "fused"])
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def safe_cv(y: np.ndarray, max_splits: int = 5) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(max_splits, int(counts.min())))


def run_probe(x: np.ndarray, y: np.ndarray) -> dict:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    cv = safe_cv(y)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, xs, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0), scoring="accuracy")
    clf.fit(xs, y)
    return {
        "cv_acc_mean": float(scores.mean()),
        "cv_acc_std": float(scores.std()),
        "train_acc": float(clf.score(xs, y)),
        "cv_folds": cv,
    }


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.emb_dir)

    # Receiver probe: file-level (48 samples, 2 classes)
    file_data = np.load(emb_dir / "file_embeddings.npz", allow_pickle=True)
    z_file = file_data[args.path]
    rx_file = file_data["receivers"]
    rx_metrics = run_probe(z_file, rx_file)

    # Device probe: window-level (enough samples per class)
    win_data = np.load(emb_dir / "window_embeddings.npz", allow_pickle=True)
    z_win = win_data[args.path]
    dev_win = win_data["labels"]
    dev_metrics = run_probe(z_win, dev_win)

    results = {
        "path": args.path,
        "receiver_probe_cv_acc_mean": rx_metrics["cv_acc_mean"],
        "receiver_probe_cv_acc_std": rx_metrics["cv_acc_std"],
        "receiver_probe_train_acc": rx_metrics["train_acc"],
        "device_probe_cv_acc_mean": dev_metrics["cv_acc_mean"],
        "device_probe_cv_acc_std": dev_metrics["cv_acc_std"],
        "device_probe_train_acc": dev_metrics["train_acc"],
        "receiver_discriminability_ratio": rx_metrics["cv_acc_mean"] / max(dev_metrics["cv_acc_mean"], 1e-8),
    }

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
