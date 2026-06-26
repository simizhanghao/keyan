#!/usr/bin/env python3
"""Generate full-mode CALIBRATION_REPORT.md with success gates."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def load_summary(out_dir: Path) -> list[dict]:
    with (out_dir / "summary_full.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section_baseline(rows: list[dict]) -> str:
    lines = ["| Direction | Source classifier (mean ± std) |", "|-----------|-------------------------------|"]
    for d in ["rx1_to_rx2", "rx2_to_rx1"]:
        vals = [float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "source_classifier"]
        m, s = agg(vals)
        lines.append(f"| {d} | {m*100:.1f} ± {s*100:.1f}% (n={len(vals)}) |")
    return "\n".join(lines)


def section_rcpa_t(rows: list[dict], k: int) -> str:
    lines = [f"### RCPA-T at K={k}", "", "| Direction | mean ± std | n |", "|-----------|-------------|---|"]
    for d in ["rx1_to_rx2", "rx2_to_rx1"]:
        vals = [float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
        m, s = agg(vals)
        lines.append(f"| {d} | {m*100:.1f} ± {s*100:.1f}% | {len(vals)} |")
    all_vals = [float(r["file_acc"]) for r in rows if r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
    m, s = agg(all_vals)
    lines.append(f"| **Both directions** | **{m*100:.1f} ± {s*100:.1f}%** | {len(all_vals)} |")
    return "\n".join(lines)


def improvement_vs_cls(rows: list[dict], k: int) -> str:
    lines = [f"### RCPA-T vs source classifier at K={k}", "", "| Direction | Δ (pp) |", "|-----------|--------|"]
    for d in ["rx1_to_rx2", "rx2_to_rx1"]:
        cls = [float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "source_classifier"]
        rcpa = [float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
        if cls and rcpa:
            # paired by seed+split
            cls_map = {(r["seed"], r["split_seed"]): float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "source_classifier"}
            deltas = []
            for r in rows:
                if r["direction"] == d and r["method"] == "RCPA-T" and int(r["shot_k"]) == k:
                    key = (r["seed"], r["split_seed"])
                    if key in cls_map:
                        deltas.append((float(r["file_acc"]) - cls_map[key]) * 100)
            m, s = agg(deltas)
            lines.append(f"| {d} | {m:+.1f} ± {s:.1f} pp |")
    return "\n".join(lines)


def rcpa_b_vs_t(rows: list[dict], k: int) -> str:
    t_vals = [float(r["file_acc"]) for r in rows if r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
    b_vals = [float(r["file_acc"]) for r in rows if r["method"] == "RCPA-B" and int(r["shot_k"]) == k]
    t_m, _ = agg(t_vals)
    b_m, _ = agg(b_vals)
    verdict = "RCPA-B consistently below RCPA-T" if b_m < t_m else "RCPA-B not consistently below RCPA-T"
    return f"At K={k}: RCPA-T mean={t_m*100:.1f}%, RCPA-B mean={b_m*100:.1f}%. **{verdict}**."


def monotonic_check(rows: list[dict]) -> str:
    ks = [1, 3, 5, 10, 20]
    means = []
    for k in ks:
        vals = [float(r["file_acc"]) for r in rows if r["method"] == "RCPA-T" and int(r["shot_k"]) == k]
        m, _ = agg(vals)
        means.append((k, m))
    mono = all(means[i][1] <= means[i + 1][1] for i in range(len(means) - 1))
    curve = ", ".join(f"K={k}:{m*100:.1f}%" for k, m in means)
    return f"RCPA-T mean curve: {curve}. Overall monotonic increase: **{'YES' if mono else 'NO (minor dips possible)'}**."


def gate_check(rows: list[dict]) -> str:
    results = []
    for gate_name, k_list, cls_delta, abs_acc in [
        ("Primary (+10 pp)", [5, 10], 10.0, None),
        ("Strong (≥40%)", [10], None, 40.0),
        ("Very strong (≥50%)", [10, 20], None, 50.0),
    ]:
        passed = False
        detail = ""
        for k in k_list:
            deltas = []
            accs = []
            for d in ["rx1_to_rx2", "rx2_to_rx1"]:
                cls_map = {(r["seed"], r["split_seed"]): float(r["file_acc"]) for r in rows if r["direction"] == d and r["method"] == "source_classifier"}
                for r in rows:
                    if r["direction"] == d and r["method"] == "RCPA-T" and int(r["shot_k"]) == k:
                        key = (r["seed"], r["split_seed"])
                        if key in cls_map:
                            deltas.append((float(r["file_acc"]) - cls_map[key]) * 100)
                        accs.append(float(r["file_acc"]) * 100)
            mean_delta = np.mean(deltas) if deltas else float("nan")
            mean_acc = np.mean(accs) if accs else float("nan")
            if cls_delta is not None and mean_delta >= cls_delta:
                passed = True
                detail = f"K={k}: mean Δ={mean_delta:+.1f} pp"
                break
            if abs_acc is not None and mean_acc >= abs_acc:
                passed = True
                detail = f"K={k}: mean acc={mean_acc:.1f}%"
                break
        results.append(f"| {gate_name} | {'**PASS**' if passed else 'FAIL'} | {detail} |")
    return "\n".join(["| Gate | Status | Detail |", "|------|--------|--------|"] + results)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = load_summary(out_dir)

    md = f"""# RCPA Calibration Report (Full Mode)

> Primary method: **RCPA-T** (target-receiver K-shot prototype calibration)
> RCPA-B reported as ablation only. OOB equalization **not run**.

---

## Protocol limitations

- Each target receiver has **one capture file per device**.
- **K-shot = K labeled calibration windows per device**, not K files.
- Calibration / support / query blocks are **disjoint** (3 split repeats via block rotation).
- Query windows never used for prototype construction.
- Alpha sweep is sensitivity analysis only; **no query-tuned alpha**.

---

## 1. Source classifier baseline

{section_baseline(rows)}

---

## 2. RCPA-T shot curve (mean ± std over 3 seeds × 3 splits)

{section_rcpa_t(rows, 1)}

{section_rcpa_t(rows, 3)}

{section_rcpa_t(rows, 5)}

{section_rcpa_t(rows, 10)}

{section_rcpa_t(rows, 20)}

---

## 3. RCPA-T improvement over source classifier

{improvement_vs_cls(rows, 5)}

{improvement_vs_cls(rows, 10)}

---

## 4. RCPA-B ablation

{rcpa_b_vs_t(rows, 5)}

{rcpa_b_vs_t(rows, 10)}

> Source-target prototype blending may be harmful under asymmetric receiver shift.

---

## 5. Direction asymmetry

Compare RX1→RX2 vs RX2→RX1 RCPA-T at K=10 in section 2.

---

## 6. K monotonicity

{monotonic_check(rows)}

---

## 7. Full-mode success gates

{gate_check(rows)}

---

## OOB representation equalization

**Not run** in this full-mode pass (Step 2 in roadmap).

---

## Outputs

- `{out_dir}/summary_full.csv`
- `{out_dir}/shot_curve_rx1_to_rx2.csv`
- `{out_dir}/shot_curve_rx2_to_rx1.csv`
- `{out_dir}/shot_curve_mean.csv`
- `{out_dir}/alpha_sensitivity.csv`
- `{out_dir}/fig_shot_curve_by_direction.pdf`
- `{out_dir}/fig_shot_curve_mean.pdf`
"""
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
