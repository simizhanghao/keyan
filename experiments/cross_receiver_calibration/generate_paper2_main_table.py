#!/usr/bin/env python3
"""Generate paper-ready main table for Paper 2 from full RCPA results."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-full", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def agg(vals: list[float]) -> str:
    if not vals:
        return "N/A"
    a = np.array(vals)
    return f"{a.mean()*100:.1f} ± {a.std(ddof=0)*100:.1f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.summary_full).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Main table: both directions, mean over 3 seeds × 3 splits
    table_rows = []
    for direction in ["rx1_to_rx2", "rx2_to_rx1"]:
        sub = [r for r in rows if r["direction"] == direction]
        cls_vals = [float(r["file_acc"]) for r in sub if r["method"] == "source_classifier"]
        table_rows.append({
            "direction": direction,
            "method": "Source classifier",
            "K": "—",
            "file_acc": agg(cls_vals),
            "n_runs": len(cls_vals),
        })
        for method, label in [("RCPA-S", "Source prototype (RCPA-S)"), ("RCPA-T", "RCPA-T (primary)"), ("RCPA-B", "RCPA-B (ablation)")]:
            for k in [0, 1, 3, 5, 10, 20]:
                vals = [float(r["file_acc"]) for r in sub if r["method"] == method and int(r["shot_k"]) == k]
                if not vals:
                    continue
                table_rows.append({
                    "direction": direction,
                    "method": label,
                    "K": k,
                    "file_acc": agg(vals),
                    "n_runs": len(vals),
                })

    # Pooled mean across directions for RCPA-T
    pooled = []
    for k in [1, 3, 5, 10, 20]:
        vals = [float(r["file_acc"]) for r in rows if r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
        cls_all = [float(r["file_acc"]) for r in rows if r["method"] == "source_classifier"]
        delta = np.mean(vals) - np.mean(cls_all) if vals and cls_all else float("nan")
        pooled.append({"K": k, "RCPA-T mean±std": agg(vals), "Δ vs classifier (pp)": f"{delta*100:+.1f}"})

    main_csv = out_dir / "paper2_main_table.csv"
    with main_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "method", "K", "file_acc", "n_runs"])
        w.writeheader()
        w.writerows(table_rows)

    pooled_csv = out_dir / "paper2_rcpa_t_pooled.csv"
    with pooled_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "RCPA-T mean±std", "Δ vs classifier (pp)"])
        w.writeheader()
        w.writerows(pooled)

    md = f"""# Paper 2 — Main Results Table (Frozen)

> **Primary method: RCPA-T** (target-receiver K-shot prototype calibration)
> Backbone: frozen RF-HSTU (`F_cross_attn_chirp_plain`), **not a new architecture**
> Aggregated over 3 seeds × 3 block split repeats per direction

## Source classifier baseline

| Direction | File-Acc (mean ± std) |
|-----------|----------------------|
| RX1→RX2 | {next(r['file_acc'] for r in table_rows if r['direction']=='rx1_to_rx2' and r['method']=='Source classifier')} |
| RX2→RX1 | {next(r['file_acc'] for r in table_rows if r['direction']=='rx2_to_rx1' and r['method']=='Source classifier')} |

## RCPA-T (primary) — pooled both directions

| K | File-Acc | Δ vs classifier |
|---|----------|-----------------|
"""
    for p in pooled:
        md += f"| {p['K']} | {p['RCPA-T mean±std']}% | {p['Δ vs classifier (pp)']} pp |\n"

    md += """
## Notes for manuscript

1. K = labeled **calibration windows per device**, not K files.
2. Calibration / support / query blocks are disjoint.
3. RCPA-T is post-hoc calibration on frozen RF-HSTU; not comparable to IoTJ source-only cross-day protocol.
4. RCPA-B included as ablation only (source-target blend often harmful).
5. OOB representation equalization and TTA are supplementary / negative baselines.

## CSV outputs

- `paper2_main_table.csv` — full direction × method × K table
- `paper2_rcpa_t_pooled.csv` — pooled RCPA-T summary
"""
    (out_dir / "PAPER2_MAIN_TABLE.md").write_text(md, encoding="utf-8")
    print(f"Wrote {main_csv}, {pooled_csv}, {out_dir / 'PAPER2_MAIN_TABLE.md'}")


if __name__ == "__main__":
    main()
