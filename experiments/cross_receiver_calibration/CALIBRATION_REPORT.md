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
| rx1_to_rx2 | 19.4 ± 3.9% (n=3) |
| rx2_to_rx1 | nan ± nan% (n=0) |

---

## 2. RCPA-T shot curve (mean ± std over 3 seeds × 3 splits)

### RCPA-T at K=1

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 29.2 ± 6.8% | 3 |
| rx2_to_rx1 | nan ± nan% | 0 |
| **Both directions** | **29.2 ± 6.8%** | 3 |

### RCPA-T at K=3

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 40.3 ± 9.8% | 3 |
| rx2_to_rx1 | nan ± nan% | 0 |
| **Both directions** | **40.3 ± 9.8%** | 3 |

### RCPA-T at K=5

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 48.6 ± 7.1% | 3 |
| rx2_to_rx1 | nan ± nan% | 0 |
| **Both directions** | **48.6 ± 7.1%** | 3 |

### RCPA-T at K=10

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 58.3 ± 3.4% | 3 |
| rx2_to_rx1 | nan ± nan% | 0 |
| **Both directions** | **58.3 ± 3.4%** | 3 |

### RCPA-T at K=20

| Direction | mean ± std | n |
|-----------|-------------|---|
| rx1_to_rx2 | 72.2 ± 2.0% | 3 |
| rx2_to_rx1 | nan ± nan% | 0 |
| **Both directions** | **72.2 ± 2.0%** | 3 |

---

## 3. RCPA-T improvement over source classifier

### RCPA-T vs source classifier at K=5

| Direction | Δ (pp) |
|-----------|--------|
| rx1_to_rx2 | +29.2 ± 3.4 pp |

### RCPA-T vs source classifier at K=10

| Direction | Δ (pp) |
|-----------|--------|
| rx1_to_rx2 | +38.9 ± 2.0 pp |

---

## 4. RCPA-B ablation

At K=5: RCPA-T mean=48.6%, RCPA-B mean=31.9%. **RCPA-B consistently below RCPA-T**.

At K=10: RCPA-T mean=58.3%, RCPA-B mean=33.3%. **RCPA-B consistently below RCPA-T**.

> Source-target prototype blending may be harmful under asymmetric receiver shift.

---

## 5. Direction asymmetry

Compare RX1→RX2 vs RX2→RX1 RCPA-T at K=10 in section 2.

---

## 6. K monotonicity

RCPA-T mean curve: K=1:29.2%, K=3:40.3%, K=5:48.6%, K=10:58.3%, K=20:72.2%. Overall monotonic increase: **YES**.

---

## 7. Full-mode success gates

| Gate | Status | Detail |
|------|--------|--------|
| Primary (+10 pp) | **PASS** | K=5: mean Δ=+29.2 pp |
| Strong (≥40%) | **PASS** | K=10: mean acc=58.3% |
| Very strong (≥50%) | **PASS** | K=10: mean acc=58.3% |

---

## OOB representation equalization

**Not run** in this full-mode pass (Step 2 in roadmap).

---

## Outputs

- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/summary_full.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/shot_curve_rx1_to_rx2.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/shot_curve_rx2_to_rx1.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/shot_curve_mean.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/alpha_sensitivity.csv`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/fig_shot_curve_by_direction.pdf`
- `/data1/hcc/llm4RF/experiments/cross_receiver_calibration/results/full_20260824/fig_shot_curve_mean.pdf`
