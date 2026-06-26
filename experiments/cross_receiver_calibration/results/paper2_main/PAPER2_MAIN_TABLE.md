# Paper 2 — Main Results Table (Frozen)

> **Primary method: RCPA-T** (target-receiver K-shot prototype calibration)
> Backbone: frozen RF-HSTU (`F_cross_attn_chirp_plain`), **not a new architecture**
> Aggregated over 3 seeds × 3 block split repeats per direction

## Source classifier baseline

| Direction | File-Acc (mean ± std) |
|-----------|----------------------|
| RX1→RX2 | 19.4 ± 3.4 |
| RX2→RX1 | 20.8 ± 8.8 |

## RCPA-T (primary) — pooled both directions

| K | File-Acc | Δ vs classifier |
|---|----------|-----------------|
| 1 | 33.1 ± 11.2% | +13.0 pp |
| 3 | 49.8 ± 11.4% | +29.6 pp |
| 5 | 58.3 ± 9.0% | +38.2 pp |
| 10 | 69.4 ± 9.7% | +49.3 pp |
| 20 | 75.0 ± 8.0% | +54.9 pp |

## Notes for manuscript

1. K = labeled **calibration windows per device**, not K files.
2. Calibration / support / query blocks are disjoint.
3. RCPA-T is post-hoc calibration on frozen RF-HSTU; not comparable to IoTJ source-only cross-day protocol.
4. RCPA-B included as ablation only (source-target blend often harmful).
5. OOB representation equalization and TTA are supplementary / negative baselines.

## CSV outputs

- `paper2_main_table.csv` — full direction × method × K table
- `paper2_rcpa_t_pooled.csv` — pooled RCPA-T summary
