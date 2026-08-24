#!/usr/bin/env python3
"""Build the deterministic X6 neutral OOB vector from development receivers only."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--packets-per-receiver", type=int, default=256)
    a = p.parse_args()
    paths = sorted(a.development_root.glob("*_train.h5"))
    names = {q.stem.removesuffix("_train") for q in paths}
    if len(paths) != 14 or names & BLIND:
        raise SystemExit("expected exactly 14 development receivers and no blind receiver")
    mask = np.abs(np.fft.fftshift(np.fft.fftfreq(8192, d=1e-6))) > 62500
    receiver_means = []
    for path in paths:
        with h5py.File(path, "r") as f:
            raw = np.asarray(f["data"][:a.packets_per_receiver], dtype=np.float32)
        half = raw.shape[1] // 2
        z = raw[:, :half] + 1j * raw[:, half:]
        z /= np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)) + 1e-8
        receiver_means.append(np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1))[:, mask].mean(0))
    neutral = np.mean(receiver_means, axis=0).astype(np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, neutral)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    meta = {
        "source": "14 development receivers only", "receiver_weighting": "equal",
        "packets_per_receiver": a.packets_per_receiver, "vector": str(a.out.resolve()),
        "shape": list(neutral.shape), "dtype": str(neutral.dtype), "sha256": digest,
        "phase_convention": "real nonnegative mean magnitude (zero phase)", "blind_opened": False,
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
