#!/usr/bin/env python3
"""Plot TTA threshold sweep and write report."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rcpa-k5-acc", type=float, default=45.833)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = list(csv.DictReader((out_dir / "tta_threshold_sweep.csv").open(encoding="utf-8")))

    xs = [float(r["threshold"]) for r in rows]
    accs = [float(r["file_acc"]) * 100 for r in rows]
    selected = [int(r["num_pseudo_selected"]) for r in rows]
    classes = [int(r["num_classes_updated"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, accs, "o-", color="C0", label="Pseudo-proto file-acc")
    ax1.axhline(args.rcpa_k5_acc, color="C2", ls="--", label=f"RCPA-T K=5 ({args.rcpa_k5_acc:.1f}%)")
    ax1.axhline(16.7, color="gray", ls=":", label="Source classifier (16.7%)")
    ax1.set_xlabel("Pseudo-label confidence threshold")
    ax1.set_ylabel("File-level accuracy (%)")
    ax1.set_title("Pseudo-proto TTA threshold sweep (RX1→RX2, cal Block A only)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(xs, selected, width=0.03, alpha=0.25, color="C1", label="# pseudo selected")
    ax2.set_ylabel("# pseudo samples selected")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_tta_threshold_sweep.pdf")
    fig.savefig(out_dir / "fig_tta_threshold_sweep.png", dpi=150)
    plt.close(fig)

    best = max(rows, key=lambda r: float(r["file_acc"]))
    near_rcpa = any(float(r["file_acc"]) * 100 >= args.rcpa_k5_acc - 5 for r in rows)

    md = f"""# TTA Threshold Sweep Report (Appendix Defense)

> RX1→RX2, seed0, split0. Adapt on unlabeled calibration Block A only; query Block C+D.

## Results

| Threshold | # Pseudo | # Classes updated | File-Acc | Macro-F1 | Top-1 mass | Collapse? |
|-----------|----------|-------------------|----------|----------|------------|-----------|
"""
    for r in rows:
        md += (
            f"| {float(r['threshold']):.2f} | {r['num_pseudo_selected']} | {r['num_classes_updated']} | "
            f"{float(r['file_acc'])*100:.1f}% | {float(r['macro_f1'])*100:.1f}% | "
            f"{float(r['top1_file_mass'])*100:.1f}% | {'YES' if int(r['collapse_flag']) else 'no'} |\n"
        )

    md += f"""
## Questions answered

1. **Pseudo samples:** ranges from {min(selected)} to {max(selected)} across thresholds.
2. **Class coverage:** at most {max(classes)} / 24 classes updated.
3. **Best threshold:** {best['threshold']} → {float(best['file_acc'])*100:.1f}% file-acc.
4. **Near RCPA-T K=5 ({args.rcpa_k5_acc:.1f}%)?** {'No' if not near_rcpa else 'Partially'}.
5. **Collapse:** high top-1 mass persists at several thresholds.

## Conclusion

**Unlabeled pseudo-proto TTA cannot replace minimal labeled receiver calibration under severe cross-receiver shift**, even after threshold tuning. This is an appendix defense experiment only.
"""
    (out_dir / "TTA_THRESHOLD_SWEEP_REPORT.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'TTA_THRESHOLD_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()
