#!/usr/bin/env python3
"""Compute cross-receiver embedding distance diagnostics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--path", default="fused", choices=["main", "oob", "fused"])
    p.add_argument("--level", default="file", choices=["file", "window"])
    p.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def normalize_rows(z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.clip(norms, 1e-8, None)


def pairwise_dist(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        a_n = a / max(np.linalg.norm(a), 1e-8)
        b_n = b / max(np.linalg.norm(b), 1e-8)
        return float(1.0 - np.dot(a_n, b_n))
    return float(np.linalg.norm(a - b))


def main() -> None:
    args = parse_args()
    fname = "file_embeddings.npz" if args.level == "file" else "window_embeddings.npz"
    data = np.load(Path(args.emb_dir) / fname, allow_pickle=True)
    z = data[args.path]
    if args.metric == "cosine":
        z = normalize_rows(z)
    labels = data["labels"]
    receivers = data["receivers"]

    rx_ids = sorted(set(receivers.tolist()))
    devices = sorted(set(labels.tolist()))

    same_dev_cross_rx, diff_dev_same_rx, diff_dev_cross_rx = [], [], []

    for d in devices:
        idx_r1 = np.where((labels == d) & (receivers == rx_ids[0]))[0]
        idx_r2 = np.where((labels == d) & (receivers == rx_ids[1]))[0]
        if len(idx_r1) and len(idx_r2):
            if args.level == "window" and len(idx_r1) > 1:
                c1 = z[idx_r1].mean(axis=0)
                c2 = z[idx_r2].mean(axis=0)
                same_dev_cross_rx.append(pairwise_dist(c1, c2, args.metric))
            else:
                same_dev_cross_rx.append(pairwise_dist(z[idx_r1[0]], z[idx_r2[0]], args.metric))

    for rx in rx_ids:
        idx_rx = np.where(receivers == rx)[0]
        devs_rx = labels[idx_rx]
        for i, j in itertools.combinations(range(len(idx_rx)), 2):
            if devs_rx[i] != devs_rx[j]:
                diff_dev_same_rx.append(pairwise_dist(z[idx_rx[i]], z[idx_rx[j]], args.metric))

    for d1, d2 in itertools.combinations(devices, 2):
        i1 = np.where((labels == d1) & (receivers == rx_ids[0]))[0]
        i2 = np.where((labels == d2) & (receivers == rx_ids[1]))[0]
        if len(i1) and len(i2):
            diff_dev_cross_rx.append(pairwise_dist(z[i1[0]], z[i2[0]], args.metric))

    summary = {
        "path": args.path,
        "level": args.level,
        "metric": args.metric,
        "same_device_cross_rx_mean": float(np.mean(same_dev_cross_rx)),
        "same_device_cross_rx_std": float(np.std(same_dev_cross_rx)),
        "same_device_cross_rx_median": float(np.median(same_dev_cross_rx)),
        "diff_device_same_rx_mean": float(np.mean(diff_dev_same_rx)),
        "diff_device_same_rx_std": float(np.std(diff_dev_same_rx)),
        "diff_device_same_rx_median": float(np.median(diff_dev_same_rx)),
        "diff_device_cross_rx_mean": float(np.mean(diff_dev_cross_rx)),
        "distance_ratio_mean": float(np.mean(same_dev_cross_rx) / max(np.mean(diff_dev_same_rx), 1e-8)),
        "distance_ratio_median": float(np.median(same_dev_cross_rx) / max(np.median(diff_dev_same_rx), 1e-8)),
        "ratio_gt_1": bool(np.mean(same_dev_cross_rx) > np.mean(diff_dev_same_rx)),
        "num_same_device_pairs": len(same_dev_cross_rx),
        "num_diff_device_same_rx_pairs": len(diff_dev_same_rx),
    }

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
