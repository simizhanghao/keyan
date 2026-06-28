#!/usr/bin/env python3
"""Build open-set manifests: 20 known + 4 unknown devices."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ALL_DEVICE_IDS, pick_unknown_devices, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-manifest", default="data/paper/cross_day_day1to5_source_only.csv")
    p.add_argument("--out-dir", default="experiments/em_robustness_openset/results/openset_splits")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--n-unknown", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / args.base_manifest

    with base_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    meta_splits = {}
    for seed in args.seeds:
        unknown_devices = pick_unknown_devices(seed, args.n_unknown)
        known_devices = [d for d in ALL_DEVICE_IDS if d not in unknown_devices]
        split_rows = []
        for r in rows:
            dev = int(r["device"])
            is_unknown = dev in unknown_devices
            split = r["split"]
            if split == "train":
                if is_unknown:
                    continue
                split_rows.append({**r, "openset_role": "train_known", "is_unknown_device": "0"})
            elif split == "val":
                split_rows.append(
                    {
                        **r,
                        "openset_role": "val_known" if not is_unknown else "val_unknown",
                        "is_unknown_device": "1" if is_unknown else "0",
                    }
                )
            elif split == "test":
                split_rows.append(
                    {
                        **r,
                        "openset_role": "test_known" if not is_unknown else "test_unknown",
                        "is_unknown_device": "1" if is_unknown else "0",
                    }
                )
            else:
                split_rows.append({**r, "openset_role": split, "is_unknown_device": "0"})

        out_csv = out_dir / f"openset_split_seed{seed}.csv"
        fields = list(split_rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(split_rows)

        meta_splits[seed] = {
            "known_devices": known_devices,
            "unknown_devices": unknown_devices,
            "manifest": str(out_csv.relative_to(ROOT)),
            "train_rows": sum(1 for r in split_rows if r["split"] == "train"),
            "val_rows": sum(1 for r in split_rows if r["split"] == "val"),
            "test_rows": sum(1 for r in split_rows if r["split"] == "test"),
        }
        print(f"seed {seed}: unknown={unknown_devices} -> {out_csv}")

    save_json(out_dir / "openset_splits_meta.json", meta_splits)


if __name__ == "__main__":
    main()
