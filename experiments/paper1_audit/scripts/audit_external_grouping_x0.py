#!/usr/bin/env python3
"""X0 grouping/leakage audit on explicitly unsealed source-domain HDF5 files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-train", type=Path, required=True)
    p.add_argument("--seen-test", type=Path, required=True)
    p.add_argument("--drift-train", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    return p.parse_args()


def schema(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        labels = np.asarray(f["label"]).reshape(-1).astype(int)
        return {
            "path": str(path.resolve()),
            "file_attributes": sorted(f.attrs.keys()),
            "datasets": {
                key: {
                    "shape": list(f[key].shape),
                    "dtype": str(f[key].dtype),
                    "attributes": sorted(f[key].attrs.keys()),
                }
                for key in sorted(f.keys())
                if isinstance(f[key], h5py.Dataset)
            },
            "label_counts": {str(k): int(v) for k, v in sorted(Counter(labels).items())},
        }


def row_hashes(path: Path, block: int = 128) -> set[bytes]:
    result: set[bytes] = set()
    with h5py.File(path, "r") as f:
        data = f["data"]
        for start in range(0, data.shape[0], block):
            rows = np.ascontiguousarray(data[start : start + block])
            result.update(hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in rows)
    return result


def main() -> int:
    a = args()
    paths = {
        "source_train": a.source_train,
        "seen_source_test": a.seen_test,
        "drift_day1_train": a.drift_train,
    }
    report = {name: schema(path) for name, path in paths.items()}
    hashes = {name: row_hashes(path) for name, path in paths.items()}
    overlaps = {
        "source_train_vs_seen_test": len(hashes["source_train"] & hashes["seen_source_test"]),
        "source_train_vs_drift_day1_train": len(
            hashes["source_train"] & hashes["drift_day1_train"]
        ),
    }
    payload = {
        "training": False,
        "official_blind_signal_opened": False,
        "files": report,
        "unique_row_hashes": {name: len(values) for name, values in hashes.items()},
        "exact_iq_row_overlaps": overlaps,
        "grouping_verdict": "NO_CAPTURE_OR_SESSION_ID",
    }
    a.out_root.mkdir(parents=True, exist_ok=True)
    (a.out_root / "x0_grouping_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# X0 grouping and exact-overlap audit",
        "",
        "training=false; official_blind_signal_opened=false",
        "",
        "| File role | Data shape | Labels | File attrs | Dataset attrs | Unique IQ rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report.items():
        ds = row["datasets"]
        lines.append(
            f"| {name} | `{ds['data']['shape']}` | {len(row['label_counts'])} | "
            f"{len(row['file_attributes'])} | {sum(len(v['attributes']) for v in ds.values())} | "
            f"{len(hashes[name])} |"
        )
    lines += [
        "",
        "## Exact IQ-row overlap",
        "",
        f"- source train vs seen-source test: **{overlaps['source_train_vs_seen_test']}**",
        f"- source train vs drift day-1 train: **{overlaps['source_train_vs_drift_day1_train']}**",
        "",
        "## Locked interpretation",
        "",
        "- The audited files contain packet rows and device labels but no capture/session IDs or attributes.",
        "- Therefore capture-level independence cannot be asserted or reconstructed; no synthetic grouping key will be invented.",
        "- Later reporting must aggregate over receiver domains/seeds and clearly label packet-level accuracy/macro-F1; capture/session inference is disallowed unless new metadata is supplied.",
        "- Zero exact-row overlap is a duplicate-leakage check only, not proof that packets are statistically independent.",
        "- Only N210_1 source/seen and source-versus-drift-day1 were checked here; the six official blind receiver signals remain sealed until X6.",
        "",
    ]
    (a.out_root / "X0_GROUPING_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"grouping_verdict": payload["grouping_verdict"], **overlaps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
