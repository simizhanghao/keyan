# TTA Negative Baseline Report

> **Unsupervised adaptation on target calibration Block A only** (64 unlabeled windows/device).
> Query Block C+D never used for adaptation. No labeled support.

Direction: **rx1_to_rx2** | Seed: **0** | Split: **0**

---

## Results (file-level accuracy)

| Method | File-Acc | Macro-F1 | Top-1 file mass | # classes predicted |
|--------|----------|----------|-----------------|---------------------|
| Source classifier | 16.7% | 11.8% | 16.7% | 13 |
| Entropy-min TTA | 20.8% | 17.4% | 16.7% | 14 |
| Pseudo-proto TTA | 8.3% | 5.6% | 25.0% | 12 |
| **RCPA-T K=5** (labeled) | 45.8% | 36.8% | — | — |

---

## Interpretation

1. **Unsupervised TTA does not replace labeled calibration.** RCPA-T uses K labeled target windows; TTA uses unlabeled cal pool only.
2. **Collapse risk:** High top-1 file mass indicates prediction collapse under cross-RX shift (cf. diagnosis CNN 95.8%).
3. **Paper claim supported:** "When labeled calibration windows are unavailable, unsupervised TTA is insufficient; minimal labeled RCPA-T is the practical deployment mode."

---

## Positioning

- TTA baselines are **negative results**, not competitors to RCPA-T.
- Do **not** compare TTA directly to IoTJ source-only cross-day numbers.
