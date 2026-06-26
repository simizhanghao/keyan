# Paper 2 Draft Overlap Self-Check

> Generated during manuscript v0 writing (2026-06-26).  
> Compare against Paper 1: `docs/iotj_paper/main.tex` and `experiments/cross_receiver_calibration/PAPER2_OVERLAP_AUDIT.md`.

---

## Checklist

| Item | Status | Notes |
|------|--------|-------|
| Paper 1 abstract sentences reused verbatim | **PASS** | Paper 2 abstract rewritten; focuses on cross-RX diagnosis + RCPA, not architecture/cross-day |
| Paper 1 introduction paragraphs reused | **PASS** | New `\IEEEPARstart` opening; cross-RX/OOB entanglement framing; no RF-HSTU architecture contribution list |
| Paper 1 architecture figure reused | **PASS** | No TikZ architecture figure; only `fig1_diagnosis_summary` and `fig2_rcpa_shotcurve` |
| Paper 1 cross-day main table in Paper 2 main results | **PASS** | Main tables are cross-RX baselines + RCPA shot curve + ablation only |
| RF-HSTU described beyond necessary background | **PASS** | One frozen-backbone paragraph + citation; no CNN stem / HSTU block / fusion equations duplicated |
| Paper 2 clearly states not a new-backbone paper | **PASS** | Stated in abstract, intro, method, discussion |
| Paper 1 "source-only cross-day robustness" as Paper 2 primary claim | **PASS** | Paper 2 explicitly requires labeled target windows; cross-day out of scope |
| Paper 1 conclusion/architecture superiority wording | **PASS** | Conclusion discusses diagnosis + calibration only |

---

## Side-by-side claim boundary

| Topic | Paper 1 IoTJ | Paper 2 RCPA (this draft) |
|-------|--------------|---------------------------|
| Primary contribution | OOB cross-attention RF-HSTU architecture | OOB entanglement diagnosis + RCPA-T calibration |
| Main evaluation | Cross-day Day1–5 source-only | Cross-receiver RX1↔RX2 with K-window calibration |
| OOB role | Fusion mechanism for robustness | Dual role: device evidence vs receiver entanglement |
| Cross-receiver | Limitation row (~20%) | Main problem + solution |
| Hero figure | Architecture TikZ | Diagnosis 4-panel + RCPA shot curve |
| Deployment claim | Source-only gateway auth | Source-only / unlabeled TTA / K-shot calibration modes |

---

## Residual overlap risks

| Risk | Severity | Mitigation in draft |
|------|----------|---------------------|
| Same OSU LoRa dataset | Medium | Cite Paper 1; emphasize Diff_Receivers subset + block-disjoint K-window protocol |
| Shared keyword "OOB", "LoRa RFFI" | Low | Expected; disclosure + distinct title/abstract |
| Citation to prior backbone work | Low | `paper1_placeholder` used consistently; update after Paper 1 decision |
| Intro paragraph 1 IoT/RFFI background similarity | Low | Reworded; shorter; points to cross-RX not cross-day |
| Related work dataset paragraph | Medium | Similar citations to Paper 1 but rewritten for cross-RX focus; acceptable with disclosure |

---

## Phrases deliberately avoided

- "first cross-receiver RFFI"
- "new RF-HSTU architecture"
- "receiver-independent solved"
- "fully robust cross-receiver RFFI"
- "source-only receiver calibration"

---

## Manual review still needed before submission

1. Run diff against Paper 1 intro/abstract sentence-by-sentence after author names added.
2. Verify `refs.bib` placeholders (`ahmed2024`, `cekic2020`) before submission.
3. Select cover letter Version A or B per Paper 1 status.
4. Human read of Discussion limitation paragraphs for tone/claim discipline.

---

## Verdict

**Overlap self-check: PASS with Medium residual dataset/backbone citation risk.**

Manageable with cover letter disclosure and Paper 1 citation after decision. No blocking verbatim reuse detected in v0 draft.
