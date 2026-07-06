#!/usr/bin/env python3
"""Compare main-only / OOB-only / fused path separability for cross-receiver."""
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
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def safe_cv(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(5, int(counts.min())))


def probe_acc(x: np.ndarray, y: np.ndarray) -> float:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    cv = safe_cv(y)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, xs, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0), scoring="accuracy")
    return float(scores.mean())


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.emb_dir)
    file_data = np.load(emb_dir / "file_embeddings.npz", allow_pickle=True)
    win_data = np.load(emb_dir / "window_embeddings.npz", allow_pickle=True)

    rows = []
    for path in ["main", "oob", "fused"]:
        if path not in file_data:
            continue
        rows.append({
            "path": path,
            "receiver_cv_acc": probe_acc(file_data[path], file_data["receivers"]),
            "device_cv_acc": probe_acc(win_data[path], win_data["labels"]),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
