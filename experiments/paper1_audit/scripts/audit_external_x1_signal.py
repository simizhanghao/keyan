#!/usr/bin/env python3
"""X1 signal-level OOB audit; no classifier training and no blind RX access.

Inputs are extracted source/train HDF5 files and optional receiver-drift train
files.  The audit computes packet-level OOB/IB magnitude ratios in dB using the
X0-frozen physical mask, then reports receiver/device/day effects.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}


def receiver_type(rx: str) -> str:
    if rx.startswith("b200_mini"):
        return "b200_mini"
    return re.sub(r"_\d+$", "", rx)


def metadata(path: Path) -> tuple[str, str]:
    name = path.name
    drift = re.match(r"(?P<rx>[a-z0-9_]+?)_day(?P<day>\d+)_wireless_(?:train|test)\.h5$", name)
    if drift is not None:
        rx, day = drift.group("rx"), f"day{drift.group('day')}"
    else:
        m = re.match(r"(?P<rx>[a-z0-9_]+?)(?:_train|_test)?\.h5$", name)
        if m is None:
            raise ValueError(f"unrecognized HDF5 filename: {path}")
        rx, day = m.group("rx"), "train"
    if rx in BLIND:
        raise RuntimeError(f"blind receiver access forbidden in X1: {rx}")
    return rx, day


def ratio_features(path: Path, fs: float, bw: float, chunk: int) -> dict[str, np.ndarray]:
    rx, day = metadata(path)
    with h5py.File(path, "r") as f:
        data = f["data"]
        labels = np.asarray(f["label"]).reshape(-1).astype(int)
        if data.ndim != 2 or data.shape[1] % 2:
            raise ValueError(f"{path}: expected [N, 2*T] data, got {data.shape}")
        n, width = data.shape
        t = width // 2
        freq = np.fft.fftshift(np.fft.fftfreq(t, d=1.0 / fs))
        ib = np.abs(freq) <= bw / 2.0
        oob = ~ib
        values: list[np.ndarray] = []
        for start in range(0, n, chunk):
            raw = np.asarray(data[start : start + chunk], dtype=np.float32)
            iq = raw[:, :t] + 1j * raw[:, t:]
            spec = np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)
            power = np.abs(spec) ** 2
            ib_rms = np.sqrt(power[:, ib].mean(axis=1))
            oob_rms = np.sqrt(power[:, oob].mean(axis=1))
            values.append(20.0 * np.log10((oob_rms + 1e-12) / (ib_rms + 1e-12)))
    return {"receiver": np.repeat(rx, n), "receiver_type": np.repeat(receiver_type(rx), n),
            "day": np.repeat(day, n), "device": labels, "ratio_db": np.concatenate(values)}


def summarize(rows: list[dict[str, np.ndarray]]) -> dict:
    keys = ["receiver", "receiver_type", "day", "device"]
    out: dict = {"files": len(rows), "packets": int(sum(x["ratio_db"].size for x in rows)), "groups": {}}
    all_ratio = np.concatenate([x["ratio_db"] for x in rows])
    out["overall"] = {"mean_db": float(all_ratio.mean()), "std_db": float(all_ratio.std()),
                       "min_db": float(all_ratio.min()), "max_db": float(all_ratio.max())}
    for key in keys:
        vals = {}
        for row in rows:
            for group, ratio in zip(row[key], row["ratio_db"]):
                vals.setdefault(str(group), []).append(float(ratio))
        out["groups"][key] = {g: {"n": len(v), "mean_db": float(np.mean(v)), "std_db": float(np.std(v))}
                               for g, v in sorted(vals.items())}
    # Between-group variance of packet means is descriptive, not an inferential
    # mixed-effects estimate; packets lack capture/session identifiers.
    for key in keys:
        means = np.asarray([v["mean_db"] for v in out["groups"][key].values()])
        out.setdefault("between_group_std_db", {})[key] = float(means.std()) if means.size else 0.0
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--drift-root", type=Path)
    p.add_argument("--include-drift-test", action="store_true",
                   help="Include non-blind receiver-drift test-day files; still no classifier is trained.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fs-hz", type=float, default=1_000_000.0)
    p.add_argument("--bw-hz", type=float, default=125_000.0)
    p.add_argument("--chunk", type=int, default=128)
    args = p.parse_args()
    paths = sorted(args.source_root.glob("*_train.h5"))
    if args.drift_root:
        paths += sorted(args.drift_root.rglob("*_wireless_train.h5"))
        if args.include_drift_test:
            paths += sorted(args.drift_root.rglob("*_wireless_test.h5"))
    if not paths:
        raise SystemExit("no source/train HDF5 files found")
    rows = [ratio_features(path, args.fs_hz, args.bw_hz, args.chunk) for path in paths]
    payload = {"protocol": {"fs_hz": args.fs_hz, "bw_hz": args.bw_hz,
                             "mask": "inband |f|<=BW/2; OOB complement", "training": False,
                             "blind_opened": False, "capture_metadata": False},
               "files": [str(x.resolve()) for x in paths], "summary": summarize(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"files": len(paths), "packets": payload["summary"]["packets"],
                      "out": str(args.out), "between_group_std_db": payload["summary"]["between_group_std_db"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
