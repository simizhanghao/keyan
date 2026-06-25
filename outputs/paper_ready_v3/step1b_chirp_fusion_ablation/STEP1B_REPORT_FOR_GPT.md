# Step1b Chirp/Fusion Ablation Report (for GPT)

**Status:** 6/6 jobs succeeded (`logs/step1b_20260625_115240.log`, ~8–10 min wall time).

**Protocol:** Same as Step1 cross-day clean (Day1–3 train / Day4 val / Day5 test, zscore, batch=128, CE, val_acc, 80 epochs).

**Path:** `outputs/paper_ready_v3/step1b_chirp_fusion_ablation/outputs/`

---

## 1. Main comparison table (File-Acc / File-Macro-F1)

| Model | Fusion | Chirp | n seeds | File-Acc | File-F1 | Per-seed File-Acc |
|-------|--------|-------|---------|----------|---------|-------------------|
| **F (Step1)** | cross-attn | ✅ | 5 | **75.0 ± 5.3%** | 67.9 ± 6.8% | 83.3, 70.8, 70.8, 79.2, 70.8 |
| **F_no_chirp (Step1b)** | cross-attn | ❌ | 3 | **75.0 ± 3.4%** | 68.1 ± 4.0% | 79.2, 75.0, 70.8 |
| D (Step1) | concat | ❌ | 5 | 18.3 ± 18.6% | 12.3 ± 15.7% | 29.2, 4.2, 4.2, 50.0, 4.2 |
| D_chirp (Step1b) | concat | ✅ | 3 | **9.7 ± 2.0%** | 3.1 ± 1.8% | 8.3, 12.5, 8.3 |
| A CNN-IQ (Step1 ref) | — | — | 5 | 54.2 ± 14.2% | 45.6 ± 14.8% | 62.5, 41.7, 62.5, 33.3, 70.8 |

---

## 2. Ablation interpretation → **Case A (confirmed)**

GPT's Case A:

> F_no_chirp still strong, D_concat_chirp still weak → **cross-attention is key, chirp is auxiliary.**

Evidence:

### 2.1 Chirp effect on cross-attention F (seeds 0–2)

| seed | F + chirp | F − chirp | Δ |
|------|-----------|-----------|---|
| 0 | 83.3% | 79.2% | +4.2 pp |
| 1 | 70.8% | 75.0% | −4.2 pp |
| 2 | 70.8% | 70.8% | 0.0 pp |

**Mean F+chirp vs F−chirp (3 seeds): identical 75.0% File-Acc.** Chirp does not explain F's advantage; removing chirp does not collapse performance.

### 2.2 Chirp effect on concat D (seeds 0–2)

| seed | D − chirp | D + chirp | Δ |
|------|-----------|-----------|---|
| 0 | 29.2% | 8.3% | −20.8 pp |
| 1 | 4.2% | 12.5% | +8.3 pp |
| 2 | 4.2% | 8.3% | +4.2 pp |

Adding chirp to concat **does not rescue** D. Mean drops from 18.3% (5 seeds) to 9.7% (3 seeds). D remains unstable / near chance.

### 2.3 Fusion effect (cross-attn vs concat)

**Without chirp (F_no_chirp vs D_no_chirp):**

| seed | F_no_chirp | D_no_chirp | Δ |
|------|------------|------------|---|
| 0 | 79.2% | 29.2% | +50.0 pp |
| 1 | 75.0% | 4.2% | +70.8 pp |
| 2 | 70.8% | 4.2% | +66.7 pp |

**With chirp (F+chirp vs D+chirp):**

| seed | F+chirp | D+chirp | Δ |
|------|---------|---------|---|
| 0 | 83.3% | 8.3% | +75.0 pp |
| 1 | 70.8% | 12.5% | +58.3 pp |
| 2 | 70.8% | 8.3% | +62.5 pp |

Cross-attention dominates regardless of chirp. Gap is **+50–75 pp**, not noise.

---

## 3. Prediction collapse diagnostic

| Model | seed | unique_preds | mode_frac | Notes |
|-------|------|--------------|-----------|-------|
| F_no_chirp | 0,1,2 | 18–19 | 0.08–0.12 | Healthy, diverse |
| D_chirp | 0 | 2 | 0.71 | Collapse toward class 22 |
| D_chirp | 1 | 4 | 0.46 | Partial collapse |
| D_chirp | 2 | 2 | 0.83 | Strong collapse toward class 9 |

F_no_chirp behaves like full F (healthy predictions). D_chirp behaves like Step1 D (collapse).

---

## 4. Paper writing implications

### Can claim
- **OOB cross-attentive fusion** is the primary stability mechanism for cross-day LoRa RFFI.
- Chirp embedding is a **compatible auxiliary cue** but not required for the main gain.
- Simple concat OOB fusion is **unstable** even with chirp; not suitable as main method.

### Recommended ablation table (2×2)

|  | No chirp | + chirp |
|--|----------|---------|
| **concat OOB** | D: 18.3±18.6% | D_chirp: 9.7±2.0% |
| **cross-attn OOB** | F_no_chirp: **75.0±3.4%** | F: **75.0±5.3%** |

### Method naming (unchanged)
`F_cross_attn_chirp_plain` remains the paper method; ablation supports emphasizing **cross-attention OOB fusion** over chirp or concat.

Safe sentence:

> Removing chirp embedding from the cross-attentive hybrid does not degrade cross-day performance (75.0±3.4% vs 75.0±5.3%), whereas adding chirp to concat OOB fusion fails to restore stability (9.7±2.0% vs 18.3±18.6%).

---

## 5. Full evidence chain status (post Step1b)

| Track | Status | Role |
|-------|--------|------|
| Cross-day Step1 + Step1.5 | ✅ Strong | Main table |
| Step1b chirp/fusion | ✅ **Case A** | Ablation table |
| Phase5-clean cross-receiver | ✅ Done | Limitation |
| Phase4 deployment | ✅ Retained | Extension table |
| Step2 recipe | ⏳ Next | Optional polish |

---

## 6. Next step recommendation

Proceed to **Step2 recipe** (F only, R1–R4, 3 seeds each) only if macro-F1 / class balance needs improvement. Step1b suggests **structure ablation is complete**; recipe tuning is incremental, not blocking for paper framing.

Do **not** expand Step1b to 5 seeds unless reviewer asks — 3 seeds already show a clear 75% vs ~10% separation.

---

## 7. Artifacts

```
outputs/paper_ready_v3/step1b_chirp_fusion_ablation/outputs/*/seed_*/metrics.json
outputs/paper_ready_v3/step1b_chirp_fusion_ablation/step0_audit/jobs_preview.tsv
logs/step1b_20260625_115240.log
```
