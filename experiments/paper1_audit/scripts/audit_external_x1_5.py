#!/usr/bin/env python3
"""X1.5 publication audit on non-blind external receiver data.

Aggregates packets to receiver x device x file cells before uncertainty
summaries. No classifier training and no official blind receiver access.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}


def meta(path: Path) -> tuple[str, str]:
    m = re.match(r"(.+?)_day(\d+)_wireless_(?:train|test)\.h5$", path.name)
    if m:
        rx, day = m.group(1), f"day{m.group(2)}"
    else:
        m = re.match(r"(.+?)_(?:train|test)\.h5$", path.name)
        if not m:
            raise ValueError(path)
        rx, day = m.group(1), "train"
    if rx in BLIND:
        raise RuntimeError(f"blind receiver forbidden: {rx}")
    return rx, day


def cells(path: Path, fs: float, bw: float, chunk: int) -> list[dict]:
    rx, day = meta(path)
    with h5py.File(path, "r") as f:
        data = f["data"]
        label = np.asarray(f["label"]).reshape(-1).astype(int)
        snr = np.asarray(f["SNR"]).reshape(-1).astype(float)
        cfo = np.asarray(f["CFO"]).reshape(-1).astype(float)
        n, width = data.shape
        t = width // 2
        freq = np.fft.fftshift(np.fft.fftfreq(t, d=1.0 / fs))
        ib, oob = np.abs(freq) <= bw / 2.0, np.abs(freq) > bw / 2.0
        sums: dict[int, list[list[float]]] = {}
        for start in range(0, n, chunk):
            raw = np.asarray(data[start:start + chunk], dtype=np.float32)
            iq = raw[:, :t] + 1j * raw[:, t:]
            power = np.abs(np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)) ** 2
            ratio = 20 * np.log10((np.sqrt(power[:, oob].mean(1)) + 1e-12) /
                                  (np.sqrt(power[:, ib].mean(1)) + 1e-12))
            for lab, r, s, c in zip(label[start:start + chunk], ratio, snr[start:start + chunk], cfo[start:start + chunk]):
                sums.setdefault(int(lab), []).append([float(r), float(s), float(c)])
    out = []
    for device, vals in sorted(sums.items()):
        a = np.asarray(vals)
        out.append({"receiver": rx, "day": day, "file": path.name, "device": device,
                    "n": int(len(a)), "ratio_db": float(a[:, 0].mean()),
                    "snr": float(a[:, 1].mean()), "cfo": float(a[:, 2].mean())})
    return out


def std_group(rows: list[dict], key: str) -> float:
    means = {}
    for r in rows:
        means.setdefault(str(r[key]), []).append(r["ratio_db"])
    return float(np.std([np.mean(v) for v in means.values()]))


def bootstrap(rows: list[dict], key: str, reps: int, seed: int) -> dict:
    groups = sorted({str(r[key]) for r in rows})
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        vals = []
        for g in sampled:
            vals.extend(r["ratio_db"] for r in rows if str(r[key]) == g)
        estimates.append(float(np.mean(vals)))
    q = np.quantile(estimates, [0.025, 0.975])
    return {"clusters": len(groups), "mean_db": float(np.mean(estimates)),
            "ci95_db": [float(q[0]), float(q[1])], "unit": key}


def residual_receiver_effect(rows: list[dict]) -> dict:
    # Cell-level OLS: ratio ~ receiver + device + day + SNR + CFO.
    y = np.asarray([r["ratio_db"] for r in rows])
    receivers = sorted({r["receiver"] for r in rows})[1:]
    devices = sorted({r["device"] for r in rows})[1:]
    days = sorted({r["day"] for r in rows})[1:]
    x = [np.ones(len(rows))]
    for g in receivers: x.append(np.asarray([r["receiver"] == g for r in rows], float))
    for g in devices: x.append(np.asarray([r["device"] == g for r in rows], float))
    for g in days: x.append(np.asarray([r["day"] == g for r in rows], float))
    for k in ("snr", "cfo"):
        v = np.asarray([r[k] for r in rows]); x.append((v - v.mean()) / (v.std() + 1e-12))
    design = np.column_stack(x)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred_no_rx = design @ beta
    # receiver contribution is the fitted categorical receiver component.
    rx_start = 1
    rx_end = 1 + len(receivers)
    rx_component = design[:, rx_start:rx_end] @ beta[rx_start:rx_end]
    return {"cell_rmse_db": float(np.sqrt(np.mean((y - pred_no_rx) ** 2))),
            "receiver_component_std_db": float(np.std(rx_component)),
            "snr_beta_db_per_sd": float(beta[-2]), "cfo_beta_db_per_sd": float(beta[-1]),
            "covariates": ["receiver", "device", "day", "snr_z", "cfo_z"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--drift-root", type=Path, required=True)
    p.add_argument("--include-drift-test", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fs-hz", type=float, default=1e6)
    p.add_argument("--bw-hz", type=float, default=125e3)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    paths = sorted(args.source_root.glob("*_train.h5"))
    paths += sorted(args.drift_root.rglob("*_wireless_train.h5"))
    if args.include_drift_test: paths += sorted(args.drift_root.rglob("*_wireless_test.h5"))
    rows = [cell for path in paths for cell in cells(path, args.fs_hz, args.bw_hz, 128)]
    payload = {"protocol": {"training": False, "blind_opened": False, "cluster_unit": "receiver x device x file",
                             "mask": "|f|<=BW/2", "fs_hz": args.fs_hz, "bw_hz": args.bw_hz},
               "files": [str(x.resolve()) for x in paths], "cells": len(rows),
               "group_mean_std_db": {k: std_group(rows, k) for k in ("receiver", "device", "day")},
               "cluster_bootstrap": {k: bootstrap(rows, k, args.bootstrap, 20260823) for k in ("receiver", "device", "day")},
               "per_device_receiver": {str(d): std_group([r for r in rows if r["device"] == d], "receiver")
                                        for d in sorted({r["device"] for r in rows})},
               "per_receiver_day": {rx: std_group([r for r in rows if r["receiver"] == rx], "day")
                                    for rx in sorted({r["receiver"] for r in rows})},
               "snr_cfo_sensitivity": residual_receiver_effect(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cells": len(rows), "group_mean_std_db": payload["group_mean_std_db"],
                      "snr_cfo_sensitivity": payload["snr_cfo_sensitivity"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
