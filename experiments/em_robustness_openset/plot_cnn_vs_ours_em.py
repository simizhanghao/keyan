#!/usr/bin/env python3
"""Plot CNN-IQ vs RF-HSTU EM robustness comparison for Chapter 5."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SWEEPS = {
    "AWGN": "awgn_snr_db_sweep.csv",
    "CFO": "cfo_norm_sweep.csv",
    "Narrowband": "narrowband_sir_db_sweep.csv",
}


def read_sweep(d: Path, fname: str) -> list[dict]:
    path = d / fname
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ieee_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
        }
    )


def plot_family(title: str, cnn_rows: list[dict], ours_rows: list[dict], out: Path) -> None:
    x_cnn = [float(r["strength"]) for r in cnn_rows]
    y_cnn = [float(r["file_acc"]) * 100 for r in cnn_rows]
    x_o = [float(r["strength"]) for r in ours_rows]
    y_o = [float(r["file_acc"]) * 100 for r in ours_rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(x_o, y_o, "o-", color="#4C78A8", label="RF-HSTU (Ours)", linewidth=1.2, markersize=4)
    ax.plot(x_cnn, y_cnn, "s--", color="#F58518", label="CNN-IQ", linewidth=1.2, markersize=4)
    ax.set_title(title)
    ax.set_ylabel("File-level accuracy (%)")
    ax.set_xlabel(cnn_rows[0]["perturb_type"].replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-dir", type=Path, required=True)
    p.add_argument("--ours-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    ieee_rcparams()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for title, fname in SWEEPS.items():
        cnn = read_sweep(args.cnn_dir, fname)
        ours = read_sweep(args.ours_dir, fname)
        plot_family(title, cnn, ours, args.out_dir / f"fig5_cnn_vs_ours_{title.lower().replace(' ', '_')}")
    print(f"Wrote comparison figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
