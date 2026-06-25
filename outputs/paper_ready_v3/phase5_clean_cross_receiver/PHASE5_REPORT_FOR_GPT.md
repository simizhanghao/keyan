# Phase5-Clean + Step1.5 Complete Report (for GPT)

Generated after **12/12 jobs succeeded** (`logs/phase5_clean_20260625_113046.log`, ~11:30–11:38, total ~8 min wall time).

---

## 0. Protocol (paper_ready_v3)

| Item | Cross-day Step1 | Phase5-clean |
|------|-----------------|--------------|
| Manifest | `cross_day_day1to5_source_only.csv` | `rx1_to_rx2 / rx2_to_rx1_source_only.csv` |
| Val split | Source domain (Day4) | **Source RX only** |
| Test split | Day5 | **Target RX** |
| F oob_norm | zscore | **ratio** |
| batch | 128 | 128 |
| label_smoothing | 0 | 0 |
| seeds | 5 | 3 |
| eval args | Full architecture in eval_cmd | Full architecture in eval_cmd |

Chance level (24 classes): **4.17%** file-acc.

---

## 1. Step1.5 Cross-day Statistics (DONE, no GPU)

**Path:** `outputs/paper_ready_v3/step1_phase7_clean/statistics/`

| Metric | F_cross_attn_chirp_plain | A_cnn_iq |
|--------|--------------------------|----------|
| File-Acc (5 seeds) | **75.0 ± 5.3%** | 54.2 ± 14.2% |
| Per-seed File-Acc | 83.3, 70.8, 70.8, 79.2, 70.8 | 62.5, 41.7, 62.5, 33.3, 70.8 |
| Paired gain F−A | **+20.8 pp mean** | — |
| Seed wins | **4 F / 1 tie / 0 A** | — |
| Pooled bootstrap 95% CI (F−A) | **[+9.2, +32.5] pp** | — |

**best.pt vs last.pt (F, eval-only):** mean Δ(last−best) = **−2.5 pp** (std 9.4 pp); seed0 outlier −20.8 pp, seeds 1–4 stable.

**D/H collapse:** D seeds 1,2,4 and H seeds 0,2 show single-class collapse (mode_frac=1.0).

---

## 2. Phase5-Clean Cross-Receiver Results (12/12 OK)

**Path:** `outputs/paper_ready_v3/phase5_clean_cross_receiver/outputs/`

### 2.1 Per-run File-Acc / File-Macro-F1

| Model | Direction | seed | File-Acc | File-F1 |
|-------|-----------|------|----------|---------|
| CNN-IQ | RX1→RX2 | 0 | 4.2% | 0.3% |
| CNN-IQ | RX1→RX2 | 1 | 4.2% | 0.4% |
| CNN-IQ | RX1→RX2 | 2 | 4.2% | 0.4% |
| **F (ratio)** | RX1→RX2 | 0 | **20.8%** | 13.9% |
| **F (ratio)** | RX1→RX2 | 1 | **12.5%** | 11.1% |
| **F (ratio)** | RX1→RX2 | 2 | **20.8%** | 12.5% |
| CNN-IQ | RX2→RX1 | 0 | 41.7% | 32.2% |
| CNN-IQ | RX2→RX1 | 1 | 4.2% | 2.8% |
| CNN-IQ | RX2→RX1 | 2 | 25.0% | 18.3% |
| **F (ratio)** | RX2→RX1 | 0 | **25.0%** | 21.5% |
| **F (ratio)** | RX2→RX1 | 1 | **12.5%** | 6.7% |
| **F (ratio)** | RX2→RX1 | 2 | **8.3%** | 4.4% |

### 2.2 Direction summary (mean ± std, 3 seeds)

| Model | RX1→RX2 File-Acc | RX2→RX1 File-Acc |
|-------|------------------|------------------|
| CNN-IQ | 4.2 ± 0.0% | 23.6 ± 15.3% |
| F (ratio) | **18.1 ± 3.9%** | 15.3 ± 7.1% |

Two-direction average File-Acc: CNN **13.9%**, F **16.7%** (both >> chance but << cross-day).

### 2.3 Paired F vs CNN (per direction × seed)

| Direction | seed | CNN | F | Δ (F−CNN) | Winner |
|-----------|------|-----|---|-----------|--------|
| RX1→RX2 | 0 | 4.2% | 20.8% | +16.7 pp | **F** |
| RX1→RX2 | 1 | 4.2% | 12.5% | +8.3 pp | **F** |
| RX1→RX2 | 2 | 4.2% | 20.8% | +16.7 pp | **F** |
| RX2→RX1 | 0 | 41.7% | 25.0% | −16.7 pp | **CNN** |
| RX2→RX1 | 1 | 4.2% | 12.5% | +8.3 pp | **F** |
| RX2→RX1 | 2 | 25.0% | 8.3% | −16.7 pp | **CNN** |

**Overall paired: F wins 4 / CNN wins 2 / tie 0** (6 direction-seed pairs).

**Key asymmetry:** RX1→RX2 F consistently beats CNN (hard direction); RX2→RX1 is unstable/high-variance for both models (CNN seed0=41.7%, F seed0=25%; F seed2=8.3%).

### 2.4 Sanity vs old Phase5 seed=0 (same strict source-only manifests)

| Run | RX1→RX2 CNN | RX1→RX2 F | RX2→RX1 CNN | RX2→RX1 F |
|-----|-------------|-----------|-------------|-----------|
| Old Phase5 seed0 | 4.2% | 16.7% | 16.7% | 25.0% |
| **Phase5-clean seed0** | 4.2% | **20.8%** | **41.7%** | 25.0% |

Phase5-clean F on RX1→RX2 slightly higher (+4.1 pp); CNN RX2→RX1 much higher in clean run (41.7% vs 16.7%) — **high seed variance**, not protocol bug.

---

## 3. Same-RX Upper Bound (for Limitation table)

From prior Phase5 upper-bound runs (same receivers, **not** re-run in phase5_clean):

| Setting | CNN File-Acc | F Hybrid File-Acc |
|---------|--------------|-------------------|
| RX1 train/test (UB) | 16.7% | **100.0%** |
| RX2 train/test (UB) | 54.2% | 66.7% |

Interpretation: models **can learn** on same RX; cross-RX source-only transfer collapses → **receiver domain gap**, not fundamental non-separability.

---

## 4. Paper claim boundaries (recommended)

### CAN claim
- Cross-day: F stable 75.0±5.3% vs CNN 54.2±14.2% (strong, Step1.5 bootstrap supported).
- Cross-receiver stress test: F shows **modest improvement on hard direction RX1→RX2** (18.1% vs 4.2% mean).
- OOB cross-attentive RF-HSTU helps **some** deployment shifts (Phase4, separate table).

### CANNOT claim
- Receiver-invariant / cross-receiver robust RFFI.
- F uniformly beats CNN on cross-receiver (RX2→RX1: CNN wins 2/3 seeds).
- Solving receiver calibration mismatch.

### Limitation wording (safe)
> Strict source-only cross-receiver transfer remains close to chance for both CNN and the proposed model, although the hybrid shows limited gains on RX1→RX2. Same-receiver upper bounds reach 66–100% file accuracy, indicating that receiver-induced calibration mismatch—not lack of device separability—is the dominant bottleneck.

---

## 5. Next steps (GPT priority)

1. ~~Step1.5~~ ✅  
2. ~~Phase5-clean 12 jobs~~ ✅  
3. **Step1b:** F_no_chirp + D_concat_chirp (3 seeds each)  
4. **Step2 recipe** (F only, R1–R4)  
5. P2 optional: F+bn_adapt seed0 diagnostic if needed for Discussion

---

## 6. Artifacts

```
outputs/paper_ready_v3/step1_phase7_clean/statistics/STEP1_STAT_REPORT.md
outputs/paper_ready_v3/phase5_clean_cross_receiver/outputs/**/metrics.json
outputs/paper_ready_v3/phase5_clean_cross_receiver/step0_audit/jobs_preview.tsv
logs/phase5_clean_20260625_113046.log
```
