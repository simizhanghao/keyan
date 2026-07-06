#!/usr/bin/env python3
"""OOB spectrum receiver profile analysis for cross-receiver diagnosis."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx"
RECEIVERS = ["RX1", "RX2"]
EXCLUDED = {9}
WINDOW = 8192
SAMPLE_RATE = 1_000_000.0
LORA_BW = 125_000.0
SAMPLES_PER_FILE = 64
DEVICE_RE = re.compile(r"Device(\d+)_IQ\.dat$")


def label_from_raw(raw: int) -> int:
    return raw - sum(1 for ex in EXCLUDED if ex < raw)


def oob_spectrum(iq: np.ndarray) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft(iq))
    psd = np.abs(spec) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / SAMPLE_RATE))
    oob = np.abs(freq) > (LORA_BW / 2.0)
    return psd[oob]


def file_mean_oob(path: Path) -> np.ndarray:
    mm = np.memmap(path, dtype=np.complex64, mode="r")
    length = mm.shape[0]
    max_off = max(0, length - WINDOW)
    stride = max(1, max_off // max(1, SAMPLES_PER_FILE - 1))
    specs = []
    for i in range(SAMPLES_PER_FILE):
        off = min(i * stride, max_off)
        win = np.asarray(mm[off:off + WINDOW])
        if len(win) == WINDOW:
            specs.append(oob_spectrum(win))
    return np.mean(specs, axis=0) if specs else np.zeros(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_oob: dict[int, dict[str, np.ndarray]] = {}
    rx_specs: dict[str, list[np.ndarray]] = {"RX1": [], "RX2": []}

    for rx in RECEIVERS:
        for path in sorted((SUBSET / rx).glob("Device*_IQ.dat")):
            m = DEVICE_RE.search(path.name)
            if not m:
                continue
            raw = int(m.group(1))
            if raw in EXCLUDED:
                continue
            label = label_from_raw(raw)
            spec = file_mean_oob(path)
            device_oob.setdefault(label, {})[rx] = spec
            rx_specs[rx].append(spec)

    min_bins = min(len(s) for specs in rx_specs.values() for s in specs)
    mean_rx1 = np.mean([s[:min_bins] for s in rx_specs["RX1"]], axis=0)
    mean_rx2 = np.mean([s[:min_bins] for s in rx_specs["RX2"]], axis=0)
    ratio = mean_rx2 / (mean_rx1 + 1e-12)

    freq = np.fft.fftshift(np.fft.fftfreq(WINDOW, d=1.0 / SAMPLE_RATE))
    oob_freq = freq[np.abs(freq) > (LORA_BW / 2.0)][:min_bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(oob_freq / 1e3, 10 * np.log10(mean_rx1 + 1e-12), label="RX1 mean OOB")
    ax.plot(oob_freq / 1e3, 10 * np.log10(mean_rx2 + 1e-12), label="RX2 mean OOB")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("OOB PSD (dB)")
    ax.set_title("Mean OOB spectrum by receiver (averaged over devices)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rx_mean_oob_spectrum.png", dpi=150)
    fig.savefig(out_dir / "rx_mean_oob_spectrum.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(oob_freq / 1e3, ratio)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("RX2/RX1 ratio")
    ax.set_title("Receiver OOB spectral ratio (RX2/RX1)")
    fig.tight_layout()
    fig.savefig(out_dir / "rx_oob_ratio_spectrum.png", dpi=150)
    plt.close(fig)

    devices_sorted = sorted(device_oob.keys())
    shifts = []
    for d in devices_sorted:
        if "RX1" in device_oob[d] and "RX2" in device_oob[d]:
            s1 = device_oob[d]["RX1"][:min_bins]
            s2 = device_oob[d]["RX2"][:min_bins]
            shifts.append(float(np.linalg.norm(s1 - s2) / (np.linalg.norm(s1) + 1e-12)))
        else:
            shifts.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    im = ax.imshow([shifts], aspect="auto", cmap="viridis")
    ax.set_yticks([0])
    ax.set_yticklabels(["OOB shift"])
    ax.set_xticks(range(len(devices_sorted)))
    ax.set_xticklabels([str(d) for d in devices_sorted], rotation=90, fontsize=7)
    ax.set_title("Per-device normalized OOB shift (RX1 vs RX2)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "per_device_oob_shift_heatmap.png", dpi=150)
    plt.close(fig)

    with (out_dir / "oob_spectrum_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mean_rx1_oob_energy", float(np.mean(mean_rx1))])
        w.writerow(["mean_rx2_oob_energy", float(np.mean(mean_rx2))])
        w.writerow(["rx2_rx1_energy_ratio", float(np.mean(mean_rx2) / (np.mean(mean_rx1) + 1e-12))])
        w.writerow(["mean_ratio_spectrum_std", float(np.std(ratio))])
        w.writerow(["mean_per_device_oob_shift", float(np.nanmean(shifts))])

    print(f"Saved OOB spectrum analysis to {out_dir}")


if __name__ == "__main__":
    main()
