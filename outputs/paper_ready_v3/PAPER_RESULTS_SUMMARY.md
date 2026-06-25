# Paper Results Summary (paper_ready_v3)

> Generated for IoTJ paper writing. Git branch target: `paper-ready-v3`.
> Main model frozen — do not reopen architecture exploration without explicit reason.

---

## 1. Final Main Model

| Item | Value |
|------|-------|
| **Model ID** | `F_cross_attn_chirp_plain` |
| **Architecture** | CNN-stem + RF-HSTU + OOB-guided cross-attention + chirp embedding |
| **Core mechanism** | **Cross-attentive OOB fusion** (Step1b Case A confirmed) |
| **Chirp role** | Auxiliary LoRa-structure prior — **not** primary gain source |
| **Checkpoint metric** | val acc (CE, no label smoothing) |

---

## 2. Final Protocols

### Primary protocol — Cross-day (Table I)

| Item | Value |
|------|-------|
| Name | Step1 paper_ready_v3 cross-day |
| Manifest | `data/paper/cross_day_day1to5_source_only.csv` |
| Split | train=Day1–3, val=Day4, test=Day5 |
| OOB norm | **zscore** |
| Batch / LR / Epochs | 128 / 3e-3 / 80 |
| Seeds | 5 (models A, D, F, H); B uses 3 seeds |
| Report | `outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md` |
| Statistics | `outputs/paper_ready_v3/step1_phase7_clean/statistics/STEP1_STAT_REPORT.md` |

### Mechanism ablation — Fusion/chirp 2×2 (Table II)

| Item | Value |
|------|-------|
| Protocol | Same as Step1 cross-day |
| Models | D (concat ± chirp), F (cross-attn ± chirp) |
| Report | `outputs/paper_ready_v3/step1b_chirp_fusion_ablation/STEP1B_REPORT_FOR_GPT.md` |

### Extension — Deployment shift (Table III)

| Item | Value |
|------|-------|
| Phase | Phase4 deployment LOCO |
| Note | Uses **ratio** OOB norm and batch 256 — differs from Step1; footnote in paper |
| Source CSV | `outputs/paper_ready_v2/deployment_shift_fixed_summary.csv` |
| Final table | `outputs/paper_ready_v3/final_tables/table3_deployment_shift.csv` |

### Limitation — Cross-receiver stress test (Table IV)

| Item | Value |
|------|-------|
| Name | Phase5-clean **strict source-only** cross-receiver |
| Manifests | `data/paper/rx1_to_rx2_source_only.csv`, `rx2_to_rx1_source_only.csv` |
| Val | Source RX only; Test = target RX |
| F OOB norm | **ratio** (not zscore) |
| Seeds | 3 |
| Report | `outputs/paper_ready_v3/phase5_clean_cross_receiver/PHASE5_REPORT_FOR_GPT.md` |
| Chance level | 4.17% file-acc (24 classes) |

### Edge deployment (Table V)

| Item | Value |
|------|-------|
| Phase | Phase6 edge benchmark |
| Source | `outputs/paper_ready/edge_deployment_summary.csv` |
| CNN | 47.7K params, ~1.12 ms @ bs1 |
| Hybrid | 1.16M params, ~2.45 ms @ bs1 |

---

## 3. Results INCLUDED in Paper

| Table | Content | Key numbers |
|-------|---------|-------------|
| **Table I** | Cross-day main results | F **75.0±5.3%** vs CNN **54.2±14.2%** File-Acc; +20.8 pp; bootstrap CI **[+9.2, +32.5] pp**; 4 win / 1 tie / 0 loss |
| **Table II** | Fusion/chirp 2×2 ablation | cross-attn ~75% regardless of chirp; concat 9.7–18.3% (collapse) |
| **Table III** | Deployment shift (config / location / distance LOCO) | Hybrid wins distance 10/15/20m, indoor room/office; CNN wins outdoor; both struggle on config |
| **Table IV** | Cross-receiver stress test | RX1→RX2: CNN 4.2% vs F 18.1%; RX2→RX1: CNN 23.6% vs F 15.3% — **limitation, not main win** |
| **Table V** | Edge params/latency | Hybrid deployable but heavier than CNN |

**Diagnostic (optional appendix, not main table):**
- `B_linear_no_oob`: 66.7±3.4% File-Acc — shows RF-HSTU alone is strong but not the final model
- Step1.5 McNemar / best vs last.pt — appendix material

---

## 4. Results EXCLUDED from Paper

| Item | Reason |
|------|--------|
| Old target-val cross-receiver (`docs/cross_receiver_findings.md`, RX2→RX1 CNN ~58%) | Not strict source-only; **DEPRECATED** |
| LODO five-fold as primary result | Superseded by Step1 Day1–3/4/5 protocol |
| Phase3 single-seed concat 87.5% | Pilot only; unstable under multi-seed Step1 |
| M7 full robust 0% | Failed experiment |
| `H_gated_chirp_plain` as main method | Unstable (19.2±24.2%); ablation negative example only |
| SupCon / CORAL+IM / MixStyle / focal / OOB-dropout | Not part of frozen main line |
| Old single-run cross-day (74.6% window, no multi-seed) | Replaced by Step1 5-seed statistics |
| Same-RX upper bound (100% F on RX1) | Discussion/limitation context only, not a main claim |

---

## 5. Table → CSV Paths

| Paper table | CSV path |
|-------------|----------|
| Table I — Cross-day main | `outputs/paper_ready_v3/final_tables/table1_cross_day_main.csv` |
| Table II — Fusion/chirp ablation | `outputs/paper_ready_v3/final_tables/table2_fusion_chirp_ablation.csv` |
| Table III — Deployment shift | `outputs/paper_ready_v3/final_tables/table3_deployment_shift.csv` |
| Table IV — Cross-receiver stress | `outputs/paper_ready_v3/final_tables/table4_cross_receiver_stress.csv` |
| Table V — Edge deployment | `outputs/paper_ready_v3/final_tables/table5_edge_deployment.csv` |

Regenerate tables: `python3 scripts/paper/generate_final_paper_tables.py`

---

## 6. Figure Plan (final_figures/ — to be generated)

| Figure | Description | Data source |
|--------|-------------|-------------|
| **Fig. 1** | Model architecture (CNN-stem → RF-HSTU → OOB cross-attn → classifier; chirp embedding; file voting) | Method section / draw.io |
| **Fig. 2** | Cross-day per-seed File-Acc bars (CNN vs F, seeds 0–4) | Step1 per-seed table in STEP1_REPORT |
| **Fig. 3** | Fusion/chirp 2×2 ablation bar chart | table2 CSV |
| **Fig. 4** | Distance shift (5/10/15/20 m, CNN vs Hybrid) | table3 distance rows |
| **Fig. 5** | Cross-receiver stress test (RX1→RX2, RX2→RX1) | table4 CSV |

Output directory (planned): `outputs/paper_ready_v3/final_figures/`

---

## 7. Claim Boundaries (must follow)

### CAN claim
- Cross-attentive OOB fusion is the key mechanism for stable cross-day generalization
- F significantly improves cross-day File-Acc over CNN-IQ (with bootstrap CI)
- Hybrid shows promising gains under **some** deployment shifts (distance, indoor location)
- Cross-receiver is a **stress test** revealing receiver calibration mismatch

### CANNOT claim
- First to use OOB for LoRa RFFI
- Chirp embedding is the main innovation
- Receiver-invariant / cross-receiver robust
- Gated or concat fusion as viable alternatives
- Universal deployment robustness

See also: `docs/method_positioning.md`, `docs/literature_notes.md`

---

## 8. Key Report Files for GPT / Co-author

```
docs/paper_reading_notes.md
docs/literature_notes.md
docs/method_positioning.md
docs/paper_draft/IoTJ_中文论文_LoRa_RFFI_Hybrid.tex   (outdated — use this summary for numbers)
outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md
outputs/paper_ready_v3/step1b_chirp_fusion_ablation/STEP1B_REPORT_FOR_GPT.md
outputs/paper_ready_v3/phase5_clean_cross_receiver/PHASE5_REPORT_FOR_GPT.md
outputs/paper_ready_v3/PAPER_RESULTS_SUMMARY.md   (this file)
outputs/paper_ready_v3/final_tables/*.csv
```

---

## 9. Optional Non-blocking Experiments (not required for first draft)

- **Step2 recipe** (F only, R1–R4 × 3 seeds): may update training recipe if Macro-F1 improves
- **final_figures/** generation scripts
- **references.bib** from paper_reading_notes (15–25 entries)

---

## 10. Paper Title (frozen narrative)

**English:** OOB-Guided Cross-Attentive RF-HSTU Hybrid Modeling for Robust LoRa Device Authentication

**中文:** 面向鲁棒 LoRa 设备认证的 OOB 引导交叉注意力 RF-HSTU 混合建模方法
