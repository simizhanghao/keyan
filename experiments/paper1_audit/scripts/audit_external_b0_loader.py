#!/usr/bin/env python3
"""B0 pipeline sanity check for the released receiver-agnostic loader.

No training and no blind receiver access. Uses a NumPy STFT equivalent to the
released SciPy implementation so the audit remains lightweight.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}


def receiver(path: Path) -> str:
    rx = re.sub(r"_(?:train|test)\.h5$", "", path.name)
    if rx in BLIND:
        raise RuntimeError(f"blind receiver forbidden: {rx}")
    return rx


def spectrogram(iq: np.ndarray, win: int = 128, crop: float = 0.3) -> np.ndarray:
    # boxcar, 50% overlap, no padding/boundary, two-sided FFT, as released.
    step = win // 2
    frames = np.stack([iq[:, i : i + win] for i in range(0, iq.shape[1] - win + 1, step)], axis=1)
    spec = np.fft.fftshift(np.fft.fft(frames, axis=-1), axes=-1)
    spec = spec + 1e-12
    dspec = spec[:, 1:, :] / spec[:, :-1, :]
    amp = np.log10(np.abs(dspec) ** 2).transpose(0, 2, 1)
    lo, hi = int(np.floor(amp.shape[1] * crop)), int(np.ceil(amp.shape[1] * (1 - crop)))
    return amp[:, lo:hi, :]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--packets-per-file", type=int, default=8)
    args = p.parse_args()
    paths = sorted(args.source_root.glob("*_train.h5"))
    rows = []
    for path in paths:
        rx = receiver(path)
        with h5py.File(path, "r") as f:
            keys = sorted(f.keys())
            data = np.asarray(f["data"][: args.packets_per_file])
            labels = np.asarray(f["label"][: args.packets_per_file]).reshape(-1).astype(int)
            snr = np.asarray(f["SNR"][: args.packets_per_file]).reshape(-1)
            cfo = np.asarray(f["CFO"][: args.packets_per_file]).reshape(-1)
        t = data.shape[1] // 2
        iq = data[:, :t] + 1j * data[:, t:]
        rms = np.sqrt(np.mean(np.abs(iq) ** 2, axis=1))
        normalized = iq / rms[:, None]
        out = spectrogram(normalized)
        rows.append({"receiver": rx, "keys": keys, "shape": list(data.shape),
                     "labels": sorted(set(labels.tolist())), "label_min": int(labels.min()),
                     "label_max": int(labels.max()), "snr_finite": bool(np.isfinite(snr).all()),
                     "cfo_finite": bool(np.isfinite(cfo).all()), "iq_finite": bool(np.isfinite(iq).all()),
                     "normalization_rms_mean": float(rms.mean()), "spectrogram_shape": list(out.shape),
                     "spectrogram_finite": bool(np.isfinite(out).all())})
    payload = {"protocol": {"training": False, "blind_opened": False, "loader": "released LoadDataset + ChannelIndSpectrogram", "win_len": 128, "overlap": 64, "crop_ratio": 0.3, "classes": 10}, "files": rows,
               "all_pass": bool(rows and all(x["iq_finite"] and x["spectrogram_finite"] and x["snr_finite"] and x["cfo_finite"] and x["spectrogram_shape"][1:] == [52, 126] for x in rows))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"files": len(rows), "all_pass": payload["all_pass"], "shape": rows[0]["spectrogram_shape"] if rows else None}, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
