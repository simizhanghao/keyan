#!/usr/bin/env python3
"""Summarize CNN-IQ EM curves and compare to Ours."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from summarize_phase_b_d import (  # noqa: E402
    EMCR_MODERATE,
    EMCR_FORBIDDEN,
    load_all_sweeps,
    summarize_by_perturbation,
    SWEEP_FILES,
)


def read_sweeps_from_dir(d: Path, prefix: str = "") -> list[dict]:
    rows = []
    for label, fname in SWEEP_FILES.items():
        path = d / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "model": prefix,
                        "perturbation_family": label,
                        "perturb_type": r["perturb_type"],
                        "strength": float(r["strength"]),
                        "file_acc_pct": round(float(r["file_acc"]) * 100, 2),
                    }
                )
    mixed = d / "mixed_stress_sweep.csv"
    if mixed.exists():
        with mixed.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "model": prefix,
                        "perturbation_family": "Mixed stress",
                        "perturb_type": r.get("preset", "mixed"),
                        "strength": r.get("preset", ""),
                        "file_acc_pct": round(float(r["file_acc"]) * 100, 2),
                    }
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-dir", type=Path, required=True)
    p.add_argument("--ours-dir", type=Path, required=True)
    args = p.parse_args()

    cnn_rows = read_sweeps_from_dir(args.cnn_dir, "CNN-IQ")
    ours_rows = read_sweeps_from_dir(args.ours_dir, "Ours")

    summary_path = args.cnn_dir / "cnn_em_robustness_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cnn_rows[0].keys()) if cnn_rows else ["model"])
        w.writeheader()
        w.writerows(cnn_rows)

    # by perturbation for CNN
    cnn_internal = []
    for r in cnn_rows:
        cnn_internal.append(
            {
                "perturbation_family": r["perturbation_family"],
                "perturb_type": r["perturb_type"],
                "strength": r["strength"] if isinstance(r["strength"], (int, float)) else 0,
                "file_acc": r["file_acc_pct"] / 100,
                "is_clean": float(r["strength"]) == 0 or float(r["strength"]) >= 100 if isinstance(r["strength"], (int, float)) else False,
            }
        )
    by_cnn = summarize_by_perturbation(cnn_internal)
    by_path = args.cnn_dir / "cnn_em_robustness_by_perturbation.csv"
    with by_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_cnn[0].keys()))
        w.writeheader()
        w.writerows(by_cnn)

    # comparison at key points
    key_points = [
        ("AWGN 30 dB", "awgn_snr_db", 30.0),
        ("CFO 0.003", "cfo_norm", 0.003),
        ("NBI 10 dB", "narrowband_sir_db", 10.0),
        ("Clean", "awgn_snr_db", 100.0),
    ]
    lines = [
        "# CNN-IQ EM Robustness Report",
        "",
        f"**CNN dir:** `{args.cnn_dir}`",
        f"**Ours dir:** `{args.ours_dir}`",
        "",
        "## Key comparison (file-acc %)",
        "",
        "| Condition | CNN-IQ | Ours | Δ (Ours-CNN) |",
        "|-----------|--------|------|--------------|",
    ]
    for name, ptype, s in key_points:
        cnn_v = next((r["file_acc_pct"] for r in cnn_rows if r["perturb_type"] == ptype and float(r["strength"]) == s), None)
        ours_v = next((r["file_acc_pct"] for r in ours_rows if r["perturb_type"] == ptype and float(r["strength"]) == s), None)
        if cnn_v is not None and ours_v is not None:
            lines.append(f"| {name} | {cnn_v} | {ours_v} | {ours_v - cnn_v:+.1f} |")

    lines.extend(["", "## CNN ranking", ""])
    for s in by_cnn:
        lines.append(f"- {s['perturbation_family']}: drop {s['accuracy_drop_pp']} pp")

    (args.cnn_dir / "CNN_EM_ROBUSTNESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
