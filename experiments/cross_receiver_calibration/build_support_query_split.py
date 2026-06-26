#!/usr/bin/env python3
"""Build block-disjoint support/calibration/query split manifest."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfhstu.data import load_manifest
from lib.split_protocol import (
    NUM_WINDOWS,
    ROLE_CALIBRATION,
    ROLE_QUERY,
    ROLE_SOURCE,
    ROLE_SUPPORT,
    assert_disjoint_roles,
    block_of,
    role_for_window,
    window_offset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", default="rx1_to_rx2", choices=["rx1_to_rx2", "rx2_to_rx1"])
    p.add_argument("--manifest", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--root", default=str(ROOT))
    return p.parse_args()


def target_receiver(direction: str) -> int:
    return 2 if direction == "rx1_to_rx2" else 1


def source_receiver(direction: str) -> int:
    return 1 if direction == "rx1_to_rx2" else 2


def main() -> None:
    args = parse_args()
    manifest = args.manifest or f"data/paper/{args.direction}_source_only.csv"
    rows = load_manifest(manifest, root=args.root)

    out_rows: list[dict] = []
    target_rx = target_receiver(args.direction)
    source_rx = source_receiver(args.direction)

    # Source train rows (one file per device on source RX)
    seen_source: set[int] = set()
    for row in rows:
        if row.split != "train":
            continue
        if row.domains["receiver"] != source_rx:
            continue
        if row.device in seen_source:
            continue
        seen_source.add(row.device)
        length = row.path.stat().st_size // np.dtype(np.complex64).itemsize
        for wi in range(NUM_WINDOWS):
            out_rows.append({
                "direction": args.direction,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "device_id": row.device,
                "label": row.label,
                "file_path": str(row.path),
                "window_index": wi,
                "sample_offset": window_offset(wi, length),
                "block": block_of(wi),
                "role": ROLE_SOURCE,
                "shot_k": "",
            })

    # Target receiver rows (test split)
    seen_target: set[int] = set()
    for row in rows:
        if row.split != "test":
            continue
        if row.domains["receiver"] != target_rx:
            continue
        if row.device in seen_target:
            continue
        seen_target.add(row.device)
        length = row.path.stat().st_size // np.dtype(np.complex64).itemsize
        for wi in range(NUM_WINDOWS):
            role = role_for_window(wi, args.split_seed)
            out_rows.append({
                "direction": args.direction,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "device_id": row.device,
                "label": row.label,
                "file_path": str(row.path),
                "window_index": wi,
                "sample_offset": window_offset(wi, length),
                "block": block_of(wi),
                "role": role,
                "shot_k": "",
            })

    assert_disjoint_roles(out_rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "direction", "seed", "split_seed", "device_id", "label", "file_path",
        "window_index", "sample_offset", "block", "role", "shot_k",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    n_cal = sum(1 for r in out_rows if r["role"] == ROLE_CALIBRATION)
    n_sup = sum(1 for r in out_rows if r["role"] == ROLE_SUPPORT)
    n_qry = sum(1 for r in out_rows if r["role"] == ROLE_QUERY)
    n_src = sum(1 for r in out_rows if r["role"] == ROLE_SOURCE)
    print(f"Wrote {out_path}")
    print(f"  source_train={n_src} calibration={n_cal} support={n_sup} query={n_qry}")
    print("  overlap check: PASSED")

    # Write shot_k annotation file for support rows at each K (for audit)
    for k in [0, 1, 3, 5, 10, 20]:
        from lib.split_protocol import sample_k_support_indices
        chosen = set(sample_k_support_indices(k, args.split_seed))
        lo, hi = __import__("lib.split_protocol", fromlist=["support_block_range"]).support_block_range(args.split_seed)
        audit_path = out_path.with_name(f"support_k{k}_split{args.split_seed}_indices.txt")
        with audit_path.open("w", encoding="utf-8") as f:
            f.write(f"# split_seed={args.split_seed} K={k} support window indices [{lo},{hi}): {sorted(chosen)}\n")


if __name__ == "__main__":
    main()
