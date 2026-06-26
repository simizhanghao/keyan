#!/usr/bin/env python3
"""Aggregate per-run full-mode CSVs into summary tables."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True, help="Directory with per-run summary.csv files")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def read_all_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for summary in sorted(runs_dir.glob("*/summary.csv")):
        with summary.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def read_alpha(runs_dir: Path) -> list[dict]:
    rows = []
    for ap in sorted(runs_dir.glob("*/alpha_sensitivity.csv")):
        with ap.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def agg_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_summary_full(rows: list[dict], out_path: Path) -> None:
    fields = [
        "method", "direction", "model", "seed", "split_seed",
        "shot_k", "alpha", "distance", "file_acc", "macro_f1",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_shot_curve(rows: list[dict], direction: str, method: str, out_path: Path) -> None:
    """Mean ± std over seeds and split repeats for each K."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["direction"] != direction or r["method"] != method:
            continue
        if int(r["shot_k"]) < 0:
            continue
        buckets[int(r["shot_k"])].append(float(r["file_acc"]))

    fields = ["direction", "method", "shot_k", "mean_file_acc", "std_file_acc", "n_runs"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(buckets.keys()):
            mean, std = agg_stats(buckets[k])
            w.writerow({
                "direction": direction,
                "method": method,
                "shot_k": k,
                "mean_file_acc": mean,
                "std_file_acc": std,
                "n_runs": len(buckets[k]),
            })


def write_shot_curve_mean(rows: list[dict], method: str, out_path: Path) -> None:
    """Average both directions for each K."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["method"] != method:
            continue
        if int(r["shot_k"]) < 0:
            continue
        buckets[int(r["shot_k"])].append(float(r["file_acc"]))

    fields = ["method", "shot_k", "mean_file_acc", "std_file_acc", "n_runs"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(buckets.keys()):
            mean, std = agg_stats(buckets[k])
            w.writerow({
                "method": method,
                "shot_k": k,
                "mean_file_acc": mean,
                "std_file_acc": std,
                "n_runs": len(buckets[k]),
            })


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_all_runs(runs_dir)
    alpha_rows = read_alpha(runs_dir)

    write_summary_full(rows, out_dir / "summary_full.csv")

    for direction in ["rx1_to_rx2", "rx2_to_rx1"]:
        for method in ["source_classifier", "RCPA-S", "RCPA-T", "RCPA-B"]:
            if method == "source_classifier":
                # one row per run at shot_k=-1
                sub = [r for r in rows if r["direction"] == direction and r["method"] == method]
                if not sub:
                    continue
                mean, std = agg_stats([float(r["file_acc"]) for r in sub])
                with (out_dir / f"baseline_{direction}.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["direction", "method", "mean_file_acc", "std_file_acc", "n_runs"])
                    w.writeheader()
                    w.writerow({"direction": direction, "method": method, "mean_file_acc": mean, "std_file_acc": std, "n_runs": len(sub)})
            else:
                pass

        write_shot_curve(rows, direction, "RCPA-T", out_dir / f"shot_curve_{direction}.csv")
        # also include RCPA-S, RCPA-B, classifier baseline in same file
        fields = ["direction", "method", "shot_k", "mean_file_acc", "std_file_acc", "n_runs"]
        with (out_dir / f"shot_curve_{direction}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for method in ["RCPA-S", "RCPA-T", "RCPA-B"]:
                buckets: dict[int, list[float]] = defaultdict(list)
                for r in rows:
                    if r["direction"] != direction or r["method"] != method:
                        continue
                    sk = int(r["shot_k"])
                    if sk < 0:
                        continue
                    buckets[sk].append(float(r["file_acc"]))
                for k in sorted(buckets.keys()):
                    mean, std = agg_stats(buckets[k])
                    w.writerow({"direction": direction, "method": method, "shot_k": k, "mean_file_acc": mean, "std_file_acc": std, "n_runs": len(buckets[k])})
            # classifier baseline as shot_k=0 reference line
            cls_vals = [float(r["file_acc"]) for r in rows if r["direction"] == direction and r["method"] == "source_classifier"]
            if cls_vals:
                mean, std = agg_stats(cls_vals)
                w.writerow({"direction": direction, "method": "source_classifier", "shot_k": -1, "mean_file_acc": mean, "std_file_acc": std, "n_runs": len(cls_vals)})

    write_shot_curve_mean(rows, "RCPA-T", out_dir / "shot_curve_mean.csv")
    # add RCPA-B and classifier to mean file
    with (out_dir / "shot_curve_mean.csv").open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "shot_k", "mean_file_acc", "std_file_acc", "n_runs"])
        for method in ["RCPA-B", "source_classifier"]:
            buckets: dict[int, list[float]] = defaultdict(list)
            for r in rows:
                if r["method"] != method:
                    continue
                sk = int(r["shot_k"])
                if sk < 0 and method != "source_classifier":
                    continue
                buckets[sk if sk >= 0 else -1].append(float(r["file_acc"]))
            for k in sorted(buckets.keys()):
                mean, std = agg_stats(buckets[k])
                w.writerow({"method": method, "shot_k": k, "mean_file_acc": mean, "std_file_acc": std, "n_runs": len(buckets[k])})

    if alpha_rows:
        fields = ["direction", "seed", "split_seed", "shot_k", "alpha", "distance", "file_acc", "macro_f1"]
        with (out_dir / "alpha_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(alpha_rows)

    print(f"Aggregated {len(rows)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
