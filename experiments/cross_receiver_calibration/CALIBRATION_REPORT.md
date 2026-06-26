# RCPA Calibration Report (Full Mode)

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

| Direction | Source classifier (mean ± std) |
|-----------|-------------------------------|
| rx1_to_rx2 | 19.4 ± 3.4% (n=9) |
| rx2_to_rx1 | 20.8 ± 8.8% (n=9) |

---

## 2. RCPA-T shot curve (mean ± std over 3 seeds × 3 splits)

### RCPA-T at K=1

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 36.1 ± 11.9% | 9 |
| rx2_to_rx1 | 30.1 ± 9.4% | 9 |
| **Both directions** | **33.1 ± 11.2%** | 18 |

### RCPA-T at K=3

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 49.1 ± 12.5% | 9 |
| rx2_to_rx1 | 50.5 ± 10.1% | 9 |
| **Both directions** | **49.8 ± 11.4%** | 18 |

### RCPA-T at K=5

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 57.4 ± 9.4% | 9 |
| rx2_to_rx1 | 59.3 ± 8.5% | 9 |
| **Both directions** | **58.3 ± 9.0%** | 18 |

### RCPA-T at K=10

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 67.6 ± 10.0% | 9 |
| rx2_to_rx1 | 71.3 ± 9.1% | 9 |
| **Both directions** | **69.4 ± 9.7%** | 18 |

### RCPA-T at K=20

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 77.3 ± 7.1% | 9 |
| rx2_to_rx1 | 72.7 ± 8.1% | 9 |
| **Both directions** | **75.0 ± 8.0%** | 18 |

---

## 3. RCPA-T improvement over source classifier

### RCPA-T vs source classifier at K=5

| Direction | Δ (pp) |
|-----------|--------|
| rx1_to_rx2 | +38.0 ± 7.7 pp |
| rx2_to_rx1 | +38.4 ± 15.4 pp |

### RCPA-T vs source classifier at K=10

| Direction | Δ (pp) |
|-----------|--------|
| rx1_to_rx2 | +48.1 ± 9.4 pp |
| rx2_to_rx1 | +50.5 ± 15.8 pp |

---

## 4. RCPA-B ablation

At K=5: RCPA-T mean=58.3%, RCPA-B mean=36.1%. **RCPA-B consistently below RCPA-T**.

At K=10: RCPA-T mean=69.4%, RCPA-B mean=37.0%. **RCPA-B consistently below RCPA-T**.

> Source-target prototype blending may be harmful under asymmetric receiver shift.

---

## 5. Direction asymmetry

Compare RX1→RX2 vs RX2→RX1 RCPA-T at K=10 in section 2.

---

## 6. K monotonicity

RCPA-T mean curve: K=1:33.1%, K=3:49.8%, K=5:58.3%, K=10:69.4%, K=20:75.0%. Overall monotonic increase: **YES**.

---

## 7. Full-mode success gates

| Gate | Status | Detail |
|------|--------|--------|
| Primary (+10 pp) | **PASS** | K=5: mean Δ=+38.2 pp |
| Strong (≥40%) | **PASS** | K=10: mean acc=69.4% |
| Very strong (≥50%) | **PASS** | K=10: mean acc=69.4% |

---

## OOB representation equalization

**Not run** in this full-mode pass (Step 2 in roadmap).

---

## Outputs

- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/summary_full.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/shot_curve_rx1_to_rx2.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/shot_curve_rx2_to_rx1.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/shot_curve_mean.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/alpha_sensitivity.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/fig_shot_curve_by_direction.pdf`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260626_1720/fig_shot_curve_mean.pdf`
