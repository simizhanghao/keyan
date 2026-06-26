#!/usr/bin/env python3
"""Generate TTA negative baseline report."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def get_row(rows, method):
    for r in rows:
        if r["method"] == method:
            return r
    return None


def pct(v):
    return float(v) * 100 if v not in ("", None) else float("nan")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = list(csv.DictReader((out_dir / "summary_tta_negative.csv").open(encoding="utf-8")))

    cls = get_row(rows, "source_classifier")
    ent = get_row(rows, "entropy_min_tta")
    pseudo = get_row(rows, "pseudo_proto_tta")
    rcpa5 = get_row(rows, "RCPA-T_K5")

    md = f"""# TTA Negative Baseline Report

> **Unsupervised adaptation on target calibration Block A only** (64 unlabeled windows/device).
> Query Block C+D never used for adaptation. No labeled support.

Direction: **rx1_to_rx2** | Seed: **0** | Split: **0**

---

## Results (file-level accuracy)

| Method | File-Acc | Macro-F1 | Top-1 file mass | # classes predicted |
|--------|----------|----------|-----------------|---------------------|
| Source classifier | {pct(cls['file_acc']):.1f}% | {pct(cls['macro_f1']):.1f}% | {pct(cls.get('top1_file_mass','')):.1f}% | {cls.get('num_file_classes_predicted','')} |
| Entropy-min TTA | {pct(ent['file_acc']):.1f}% | {pct(ent['macro_f1']):.1f}% | {pct(ent.get('top1_file_mass','')):.1f}% | {ent.get('num_file_classes_predicted','')} |
| Pseudo-proto TTA | {pct(pseudo['file_acc']):.1f}% | {pct(pseudo['macro_f1']):.1f}% | {pct(pseudo.get('top1_file_mass','')):.1f}% | {pseudo.get('num_file_classes_predicted','')} |
| **RCPA-T K=5** (labeled) | {pct(rcpa5['file_acc']) if rcpa5 else 0:.1f}% | {pct(rcpa5['macro_f1']) if rcpa5 else 0:.1f}% | — | — |

---

## Interpretation

1. **Unsupervised TTA does not replace labeled calibration.** RCPA-T uses K labeled target windows; TTA uses unlabeled cal pool only.
2. **Collapse risk:** High top-1 file mass indicates prediction collapse under cross-RX shift (cf. diagnosis CNN 95.8%).
3. **Paper claim supported:** "When labeled calibration windows are unavailable, unsupervised TTA is insufficient; minimal labeled RCPA-T is the practical deployment mode."

---

## Positioning

- TTA baselines are **negative results**, not competitors to RCPA-T.
- Do **not** compare TTA directly to IoTJ source-only cross-day numbers.
"""
    (out_dir / "TTA_NEGATIVE_BASELINE_REPORT.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'TTA_NEGATIVE_BASELINE_REPORT.md'}")


if __name__ == "__main__":
    main()
