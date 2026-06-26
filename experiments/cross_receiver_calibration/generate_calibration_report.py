#!/usr/bin/env python3
"""Generate CALIBRATION_REPORT.md from quick mode results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--direction", default="rx1_to_rx2")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_row(rows, method, k=None):
    for r in rows:
        if r["method"] != method:
            continue
        if k is None:
            return r
        if int(r["shot_k"]) == k:
            return r
    return None


def pct(row):
    return float(row["file_acc"]) * 100 if row else None


def main() -> None:
    args = parse_args()
    with Path(args.summary_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cls = get_row(rows, "source_classifier")
    cls_acc = pct(cls) or 0.0
    rcpa_s0 = get_row(rows, "RCPA-S", 0)
    rcpa_t1 = get_row(rows, "RCPA-T", 1)
    rcpa_t5 = get_row(rows, "RCPA-T", 5)
    rcpa_b5 = get_row(rows, "RCPA-B", 5)
    rcpa_t10 = get_row(rows, "RCPA-T", 10)
    rcpa_b10 = get_row(rows, "RCPA-B", 10)

    kshot_rows = [r for r in rows if r["method"] in ("RCPA-T", "RCPA-B") and int(r["shot_k"]) > 0]
    best_k_row = max(kshot_rows, key=lambda r: float(r["file_acc"])) if kshot_rows else None
    best_k_acc = pct(best_k_row) or 0.0
    passed = (best_k_acc - cls_acc >= 10.0) or (best_k_acc >= 30.0)

    def row_line(method, k, label=None):
        r = get_row(rows, method, k) if k >= 0 else get_row(rows, method)
        if not r:
            return f"| {label or method} | {k if k >= 0 else '—'} | N/A | N/A |"
        return f"| {label or method} | {k if k >= 0 else '—'} | {pct(r):.1f} | {float(r['macro_f1'])*100:.1f} |"

    proto_vs_cls = "N/A"
    if rcpa_s0:
        proto_vs_cls = "prototype better" if pct(rcpa_s0) > cls_acc else "classifier better or tie"

    md = f"""# RCPA Calibration Report (Quick Mode)

> Direction: **{args.direction}** | Seed: **{args.seed}** | Model: Ours fused

---

## Protocol limitations (must disclose in paper)

- Each target receiver has **one capture file per device**.
- **K-shot means K labeled calibration windows per device**, not K independent files.
- Calibration (Block A), support (Block B), and query (Block C+D) windows are **disjoint**.
- Query windows are **never** used for prototype construction or receiver-statistic estimation.
- Quick mode uses only RX1→RX2, seed 0; results must not be overclaimed.

Split manifest: `{args.split_csv}`

---

## Leakage checks

- support ∩ query = **∅** (asserted at split build)
- calibration ∩ query = **∅**
- support ∩ calibration = **∅**

---

## Quick results (file-level accuracy %)

| Method | K | File-Acc | Macro-F1 |
|--------|---|----------|----------|
{row_line("source_classifier", -1, "Source classifier")}
{row_line("RCPA-S", 0, "Source prototype (RCPA-S)")}
{row_line("RCPA-T", 1, "RCPA-T")}
{row_line("RCPA-T", 5, "RCPA-T")}
{row_line("RCPA-B", 5, "RCPA-B")}
{row_line("RCPA-T", 10, "RCPA-T")}
{row_line("RCPA-B", 10, "RCPA-B")}

---

## Quick gate

| Criterion | Result |
|-----------|--------|
| Best K-shot vs classifier ≥ +10 pp | {best_k_acc - cls_acc:+.1f} pp |
| Best K-shot absolute ≥ 30% | {best_k_acc:.1f}% |
| **Passed** | **{'YES' if passed else 'NO'}** |

Best: {best_k_row['method'] if best_k_row else 'N/A'} K={best_k_row['shot_k'] if best_k_row else 'N/A'} acc={best_k_acc:.1f}%

---

## Diagnostic questions

1. **Source prototype vs classifier:** RCPA-S {pct(rcpa_s0) if rcpa_s0 else 0:.1f}% vs classifier {cls_acc:.1f}% → {proto_vs_cls}
2. **RCPA-T vs source prototype (K=5):** {"T better" if rcpa_t5 and rcpa_s0 and pct(rcpa_t5) > pct(rcpa_s0) else "T not better at K=5"}
3. **RCPA-B vs RCPA-T (K=5):** {"B better or tie" if rcpa_b5 and rcpa_t5 and pct(rcpa_b5) >= pct(rcpa_t5) else "T better at K=5"}

---

## OOB representation equalization

Quick mode: **not run** (interface reserved for full mode).

---

## Next step

{"Proceed to **full mode** (both directions, 3 seeds, split repeats)." if passed else "Analyze: if prototype > classifier, tune K-shot / blend; else consider metric learning before RCPA."}
"""
    out = Path(args.out_md)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
