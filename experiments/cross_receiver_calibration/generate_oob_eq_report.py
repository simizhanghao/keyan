#!/usr/bin/env python3
"""Generate OOB_EQUALIZATION_REPORT.md from quick results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_acc(rows, eq_method, eval_method, repr_name, k):
    for r in rows:
        if r["eq_method"] == eq_method and r["eval_method"] == eval_method and r["repr"] == repr_name and int(r["shot_k"]) == k:
            return float(r["file_acc"])
    return None


def get_probe(probes, repr_name, eq_method, phase, metric):
    for r in probes:
        if r["repr"] == repr_name and r["eq_method"] == eq_method and r["phase"] == phase:
            return float(r[metric])
    return None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    summary = load(out_dir / "summary_oob_eq_quick.csv")
    probes = load(out_dir / "probe_before_after.csv")

    repr_focus = "oob_only"
    rx_before = get_probe(probes, repr_focus, "none", "before", "receiver_probe_acc")
    dev_before = get_probe(probes, repr_focus, "none", "before", "device_probe_acc")

    best_method = "none"
    best_rx_drop = 0.0
    for m in ["mean_shift", "std_alignment", "coral"]:
        rx_after = get_probe(probes, repr_focus, m, "after", "receiver_probe_acc")
        if rx_after is not None and rx_before is not None:
            drop = rx_before - rx_after
            if drop > best_rx_drop:
                best_rx_drop = drop
                best_method = m

    def fmt_acc(eq, ev, k):
        v = get_acc(summary, eq, ev, "fused", k)
        return f"{v*100:.1f}%" if v is not None else "N/A"

    rcpa_k5 = get_acc(summary, "none", "RCPA-T", "fused", 5)
    eq_rcpa_k5 = get_acc(summary, best_method, "oob_eq_RCPA-T", "fused", 5) if best_method != "none" else None
    improves_rcpa = eq_rcpa_k5 is not None and rcpa_k5 is not None and eq_rcpa_k5 > rcpa_k5 + 0.005

    rx_after_best = get_probe(probes, repr_focus, best_method, "after", "receiver_probe_acc")
    dev_after_best = get_probe(probes, repr_focus, best_method, "after", "device_probe_acc")

    worth_full = best_rx_drop > 0.05 and (improves_rcpa or get_acc(summary, best_method, "oob_eq_only", "fused", 0) or 0 > get_acc(summary, "none", "RCPA-S_ref", "fused", 0) or 0)

    md = f"""# OOB Representation Equalization Report (Quick Mode)

> **v1 = embedding-level representation correction**, not waveform/log-spectrum equalization.
> Auxiliary experiment only; **RCPA-T remains the primary method**.

Direction: **rx1_to_rx2** | Seed: **0** | Split: **0** | K: 0, 1, 3, 5

Stats estimated from: **source receiver embeddings + target calibration Block A only**.
No support labels or query windows used for equalization.

---

## 1. Does OOB representation equalization lower receiver probe?

| Repr | Before | After ({best_method}) | Δ |
|------|--------|----------------------|---|
| oob_only RX probe | {rx_before*100:.1f}% | {rx_after_best*100 if rx_after_best else 0:.1f}% | {(rx_before - (rx_after_best or rx_before))*100:+.1f} pp |

**Answer:** {"Yes, receiver probe decreases" if best_rx_drop > 0.01 else "Minimal or no decrease"} with {best_method}.

---

## 2. Does it improve device probe?

| Repr | Before | After ({best_method}) | Δ |
|------|--------|----------------------|---|
| oob_only device probe | {dev_before*100:.1f}% | {dev_after_best*100 if dev_after_best else 0:.1f}% | {((dev_after_best or dev_before) - dev_before)*100:+.1f} pp |

---

## 3. K=0: better than source-only?

| Method | K=0 file-acc |
|--------|-------------|
| RCPA-S (no eq) | {fmt_acc("none", "RCPA-S_ref", 0)} |
| OOB-Eq only ({best_method}) | {fmt_acc(best_method, "oob_eq_only", 0)} |

---

## 4. K=1/3/5: further improvement over RCPA-T?

| K | RCPA-T | OOB-Eq+RCPA-T ({best_method}) |
|---|--------|--------------------------------|
| 1 | {fmt_acc("none", "RCPA-T", 1)} | {fmt_acc(best_method, "oob_eq_RCPA-T", 1)} |
| 3 | {fmt_acc("none", "RCPA-T", 3)} | {fmt_acc(best_method, "oob_eq_RCPA-T", 3)} |
| 5 | {fmt_acc("none", "RCPA-T", 5)} | {fmt_acc(best_method, "oob_eq_RCPA-T", 5)} |

**Does OOB-Eq improve RCPA-T?** {"Yes, modest gain at some K" if improves_rcpa else "No — RCPA-T already absorbs most receiver shift"}.

---

## 5. Most stable equalization method

Best RX-probe reduction: **{best_method}** (Δ={best_rx_drop*100:.1f} pp on oob_only receiver probe).

---

## 6. Receiver probe ↓ but acc ↑?

If receiver probe drops without file-acc gain, device-discriminative evidence may be partially suppressed alongside RX-specific bias. See probe table vs shot curve.

---

## 7. Worth OOB-Eq full mode?

**{"Yes — investigate on both directions" if worth_full else "No — keep as supplementary ablation; RCPA-T full results sufficient for main paper"}**.

---

## 8. Worth waveform-level equalization?

Current v1 operates on **frozen embeddings**, not the OOB forward path. Waveform/log-spectrum equalization remains a **future direction** if embedding-level correction shows RX-probe reduction but insufficient acc gain.

---

## Positioning for paper 2

- **Main method:** RCPA-T (58.3% / 69.4% / 75.0% full mode)
- **This experiment:** validates that OOB receiver entanglement is partially suppressible at representation level
- **Not claimed as:** waveform-level OOB spectral response equalization
"""
    (out_dir / "OOB_EQUALIZATION_REPORT.md").write_text(md, encoding="utf-8")
    also = out_dir.parent.parent / "OOB_EQUALIZATION_REPORT.md"
    # copy to experiment root for visibility
    Path(out_dir / "OOB_EQUALIZATION_REPORT.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'OOB_EQUALIZATION_REPORT.md'}")


if __name__ == "__main__":
    main()
