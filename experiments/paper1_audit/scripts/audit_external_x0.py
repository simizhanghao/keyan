#!/usr/bin/env python3
"""X0 audit for the Zhang/Shen multi-receiver LoRa archives.

This script never opens signal values from the official test receivers.  It
uses ZIP metadata to audit the official split and only parses explicitly
extracted source/train HDF5 samples for schema and OOB usability.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


EXPECTED_ARCHIVES = {
    "multiple_receiver_train.zip",
    "multiple_receiver_test.zip",
    "receiver_drift_dataset.zip",
}
EXPECTED_BLIND_RX = {
    "b200_2",
    "b200_mini_2",
    "b210_2",
    "n210_2",
    "n210_3",
    "pluto_2",
}
REQUIRED_KEYS = {"data", "label", "SNR", "CFO"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-training external multi-RX X0 audit")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--sample-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--fs-hz", type=float, default=1_000_000.0)
    p.add_argument("--bw-hz", type=float, default=125_000.0)
    p.add_argument("--samples-per-device", type=int, default=10)
    return p.parse_args()


def receiver_id(filename: str, suffix: str) -> str:
    name = Path(filename).name
    if not name.endswith(suffix):
        raise ValueError(name)
    return name[: -len(suffix)]


def receiver_type(rx: str) -> str:
    if rx.startswith("b200_mini"):
        return "b200_mini"
    return re.sub(r"_\d+$", "", rx)


def archive_inventory(data_root: Path) -> tuple[dict, dict]:
    archives: dict[str, dict] = {}
    members: dict[str, list[str]] = {}
    for path in sorted(data_root.glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
        members[path.name] = [i.filename for i in infos]
        archives[path.name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "entries": len(infos),
            "h5_entries": sum(i.filename.lower().endswith(".h5") for i in infos),
            "uncompressed_bytes": sum(i.file_size for i in infos),
            "compressed_bytes": sum(i.compress_size for i in infos),
            "encrypted_entries": sum(bool(i.flag_bits & 0x1) for i in infos),
        }
    return archives, members


def split_audit(members: dict[str, list[str]]) -> dict:
    train_names = [n for n in members.get("multiple_receiver_train.zip", []) if n.endswith("_train.h5")]
    test_names = [
        n
        for n in members.get("multiple_receiver_test.zip", [])
        if n.endswith("_test.h5") and not Path(n).name.startswith("Location_")
    ]
    location_names = [
        n for n in members.get("multiple_receiver_test.zip", []) if Path(n).name.startswith("Location_")
    ]
    train_rx = {receiver_id(n, "_train.h5") for n in train_names}
    test_rx = {receiver_id(n, "_test.h5") for n in test_names}
    seen_test_rx = train_rx & test_rx
    blind_test_rx = test_rx - train_rx
    return {
        "train_receivers": sorted(train_rx),
        "train_receiver_types": sorted({receiver_type(r) for r in train_rx}),
        "base_test_receivers": sorted(test_rx),
        "seen_test_receivers": sorted(seen_test_rx),
        "blind_test_receivers": sorted(blind_test_rx),
        "expected_blind_receivers": sorted(EXPECTED_BLIND_RX),
        "blind_matches_expected": blind_test_rx == EXPECTED_BLIND_RX,
        "location_test_files": sorted(location_names),
        "drift_files": sorted(n for n in members.get("receiver_drift_dataset.zip", []) if n.endswith(".h5")),
    }


def choose_indices(labels: np.ndarray, per_device: int) -> np.ndarray:
    selected: list[int] = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if positions.size <= per_device:
            selected.extend(positions.tolist())
        else:
            selected.extend(positions[np.linspace(0, positions.size - 1, per_device, dtype=int)].tolist())
    return np.asarray(sorted(selected), dtype=int)


def h5_audit(path: Path, fs_hz: float, bw_hz: float, per_device: int) -> dict:
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        schema = {
            key: {"shape": list(f[key].shape), "dtype": str(f[key].dtype)}
            for key in sorted(keys)
            if isinstance(f[key], h5py.Dataset)
        }
        labels = np.asarray(f["label"]).reshape(-1).astype(int)
        snr = np.asarray(f["SNR"]).reshape(-1)
        cfo = np.asarray(f["CFO"]).reshape(-1)
        data = f["data"]
        if data.ndim != 2 or data.shape[1] % 2:
            raise ValueError(f"Unexpected IQ storage shape: {data.shape}")
        complex_len = data.shape[1] // 2
        idx = choose_indices(labels, per_device)
        raw = data[idx]

    iq = raw[:, :complex_len] + 1j * raw[:, complex_len:]
    spec = np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)
    power = np.abs(spec) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(complex_len, d=1.0 / fs_hz))
    inband = np.abs(freq) <= bw_hz / 2.0
    oob = ~inband
    ib_rms = np.sqrt(power[:, inband].mean(axis=1))
    oob_rms = np.sqrt(power[:, oob].mean(axis=1))
    ratio_db = 20.0 * np.log10((oob_rms + 1e-12) / (ib_rms + 1e-12))
    edge = np.abs(freq) >= 0.45 * fs_hz
    edge_vs_oob_db = 10.0 * np.log10(
        (power[:, edge].mean() + 1e-30) / (power[:, oob].mean() + 1e-30)
    )
    label_counts = Counter(labels.tolist())
    return {
        "path": str(path.resolve()),
        "receiver": receiver_id(path.name, "_train.h5"),
        "receiver_type": receiver_type(receiver_id(path.name, "_train.h5")),
        "size_bytes": path.stat().st_size,
        "keys": sorted(keys),
        "required_keys_present": REQUIRED_KEYS <= keys,
        "schema": schema,
        "num_packets": int(data.shape[0]),
        "stored_width": int(data.shape[1]),
        "complex_samples": int(complex_len),
        "labels": sorted(label_counts),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "snr_min": float(np.min(snr)),
        "snr_max": float(np.max(snr)),
        "cfo_min": float(np.min(cfo)),
        "cfo_max": float(np.max(cfo)),
        "sampled_packets": int(idx.size),
        "inband_bins": int(inband.sum()),
        "oob_bins": int(oob.sum()),
        "inband_energy_fraction": float(power[:, inband].sum() / power.sum()),
        "oob_ratio_db_mean": float(np.mean(ratio_db)),
        "oob_ratio_db_std": float(np.std(ratio_db)),
        "oob_ratio_db_min": float(np.min(ratio_db)),
        "oob_ratio_db_max": float(np.max(ratio_db)),
        "oob_nonzero_fraction": float(np.mean(power[:, oob] > 0)),
        "edge_vs_oob_db": float(edge_vs_oob_db),
        "finite": bool(np.isfinite(raw).all() and np.isfinite(power).all()),
    }


def markdown(payload: dict) -> str:
    split = payload["official_split"]
    lines = [
        "# External multi-receiver X0 audit",
        "",
        f"verdict={payload['verdict']} training=false gpu=false official_test_signal_opened=false",
        "",
        "## Archive inventory",
        "",
        "| Archive | Entries | HDF5 | Compressed GB | Uncompressed GB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["archives"].items():
        lines.append(
            f"| `{name}` | {row['entries']} | {row['h5_entries']} | "
            f"{row['compressed_bytes']/1e9:.2f} | {row['uncompressed_bytes']/1e9:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Official split from archive metadata",
            "",
            f"- source/train receivers ({len(split['train_receivers'])}): `{', '.join(split['train_receivers'])}`",
            f"- seen test receivers ({len(split['seen_test_receivers'])}): `{', '.join(split['seen_test_receivers'])}`",
            f"- blind test receivers ({len(split['blind_test_receivers'])}): `{', '.join(split['blind_test_receivers'])}`",
            f"- expected 14/6 mapping matched: **{split['blind_matches_expected']}**",
            f"- location files: {len(split['location_test_files'])}; drift HDF5: {len(split['drift_files'])}",
            "",
            "The six blind receiver signal files were not extracted or opened. Archive filenames and sizes only were inspected.",
            "",
            "## Source/train HDF5 samples",
            "",
            "| RX | Type | Shape | DUTs | Packets | OOB/IB RMS dB | IB energy | OOB nonzero |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["source_h5_samples"]:
        shape = row["schema"]["data"]["shape"]
        lines.append(
            f"| {row['receiver']} | {row['receiver_type']} | {shape} | {len(row['labels'])} | "
            f"{row['num_packets']} | {row['oob_ratio_db_mean']:.2f}±{row['oob_ratio_db_std']:.2f} | "
            f"{100*row['inband_energy_fraction']:.2f}% | {100*row['oob_nonzero_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Locked interpretation",
            "",
            "- IQ reconstruction: first half real, second half imaginary (author loader convention).",
            f"- candidate physical mask was fixed before this audit: Fs={payload['fs_hz']:.0f} Hz, BW={payload['bw_hz']:.0f} Hz, in-band `|f| <= BW/2`, OOB otherwise.",
            "- HDF5 has no sampling-rate attribute; Fs/BW remain paper/code claims and must be cited as such.",
            "- `OOB_OK` here means full complex IQ and nonzero full-band OOB across all sampled source receiver types; it does not claim stable transmitter identity.",
            "- Official blind receiver signals remain sealed until X6.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    archives, members = archive_inventory(args.data_root)
    missing = EXPECTED_ARCHIVES - set(archives)
    split = split_audit(members)
    samples = [
        h5_audit(path, args.fs_hz, args.bw_hz, args.samples_per_device)
        for path in sorted(args.sample_root.glob("*_train.h5"))
    ]
    sampled_types = {row["receiver_type"] for row in samples}
    expected_types = set(split["train_receiver_types"])
    oob_ok = bool(samples) and sampled_types == expected_types and all(
        row["required_keys_present"]
        and row["complex_samples"] == 8192
        and row["finite"]
        and row["oob_nonzero_fraction"] > 0.99
        for row in samples
    )
    verdict = "OOB_OK" if not missing and split["blind_matches_expected"] and oob_ok else "OOB_UNKNOWN"
    payload = {
        "training": False,
        "gpu": False,
        "official_test_signal_opened": False,
        "fs_hz": args.fs_hz,
        "bw_hz": args.bw_hz,
        "missing_archives": sorted(missing),
        "archives": archives,
        "official_split": split,
        "sampled_source_types": sorted(sampled_types),
        "expected_source_types": sorted(expected_types),
        "source_h5_samples": samples,
        "verdict": verdict,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "x0_external_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_root / "x0_external_audit.md").write_text(markdown(payload) + "\n")
    (args.out_root / "OFFICIAL_SPLIT_AUDIT.md").write_text(
        "# Official 14/6 receiver split audit\n\n"
        f"matched={split['blind_matches_expected']} official_test_signal_opened=false\n\n"
        f"Source/train (14): `{', '.join(split['train_receivers'])}`\n\n"
        f"Official blind test (6): `{', '.join(split['blind_test_receivers'])}`\n\n"
        "The six blind HDF5 signal values are sealed until X6. Their ZIP metadata may be used only for integrity and split mapping.\n"
    )
    print(markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
