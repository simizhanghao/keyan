# TTA Threshold Sweep Report (Appendix Defense)

> RX1→RX2, seed0, split0. Adapt on unlabeled calibration Block A only; query Block C+D.

## Results

| Threshold | # Pseudo | # Classes updated | File-Acc | Macro-F1 | Top-1 mass | Collapse? |
|-----------|----------|-------------------|----------|----------|------------|-----------|
| 0.50 | 231 | 10 | 16.7% | 13.9% | 29.2% | no |
| 0.70 | 70 | 5 | 8.3% | 5.6% | 29.2% | no |
| 0.80 | 27 | 2 | 8.3% | 5.6% | 25.0% | no |
| 0.90 | 3 | 1 | 8.3% | 5.6% | 25.0% | no |
| 0.95 | 1 | 1 | 8.3% | 5.6% | 25.0% | no |

## Questions answered

1. **Pseudo samples:** ranges from 1 to 231 across thresholds.
2. **Class coverage:** at most 10 / 24 classes updated.
3. **Best threshold:** 0.5 → 16.7% file-acc.
4. **Near RCPA-T K=5 (45.8%)?** No.
5. **Collapse:** high top-1 mass persists at several thresholds.

## Conclusion

**Unlabeled pseudo-proto TTA cannot replace minimal labeled receiver calibration under severe cross-receiver shift**, even after threshold tuning. This is an appendix defense experiment only.
