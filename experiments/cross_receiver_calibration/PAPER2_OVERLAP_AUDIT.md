# Paper 2 Overlap Audit

> **Purpose:** Manage redundant-publication / self-overlap risk between Paper 1 (IoTJ RF-HSTU) and Paper 2 (RCPA cross-receiver calibration).
> **Status:** Preparation only — do not submit Paper 2 until Paper 1 decision.

---

## Table A: Contribution boundary

| Dimension | Paper 1 IoTJ | Paper 2 RCPA | Overlap risk | Mitigation |
|-----------|--------------|--------------|--------------|------------|
| Problem setting | Same-receiver cross-day / deployment shift | Cross-receiver transfer (RX1↔RX2) | Medium | Different shift type; explicit protocol distinction |
| Dataset | OSU LoRa 24-class, Day1–5 protocol | Same corpus, Diff_Receivers cross-RX split | **High** | Cite Paper 1; new split manifest; block-disjoint K-window protocol |
| Backbone | **RF-HSTU architecture contribution** | **Frozen RF-HSTU checkpoint** | **High** | Paper 2: zero backbone change; 1–2 paragraph background only |
| Method | OOB cross-attention fusion, training | Diagnosis + RCPA-T post-hoc calibration | Low | Entirely different method section |
| Experiments | Cross-day, fusion ablation, deployment | Cross-RX RCPA curve, TTA negative, OOB-Eq aux | Low | No cross-day main table in Paper 2 |
| Figures | Architecture TikZ, cross-day bars | Diagnosis 4-panel, RCPA shot curve | Low | **Do not reuse** Paper 1 architecture figure |
| Tables | Table I cross-day, Table IV cross-RX limitation | RCPA shot curve, baseline, ablation | Medium | Cross-RX in Paper 1 is **limitation row only**; Paper 2 expands with calibration |
| Claims | Source-only robust hybrid model | Diagnosis-first receiver calibration | Low | Rewrite all claims; no "better RF-HSTU" |
| Title keywords | OOB, cross-attention, RF-HSTU, cross-day | OOB entanglement, cross-receiver, RCPA, calibration | Medium | Distinct title; shared "OOB" and "LoRa RFFI" OK with disclosure |
| Abstract wording | Architecture + cross-day gains | Diagnosis + K-window calibration | **High** if copied | **Rewrite from scratch** |
| Introduction wording | IoT auth + model proposal | Cross-RX failure + calibration need | Medium | New intro; cite Paper 1 as prior backbone work |
| Method description | Full RF-HSTU pipeline | Frozen embeddings + RCPA protocol | Low | No architecture subsections duplicated |
| Discussion/limitation | Cross-RX as open limitation | Cross-RX as **main problem solved** | Medium | Paper 2 cites Paper 1 limitation as motivation |

---

## Table B: Prohibited reuse checklist

Paper 2 **must NOT** reuse:

- [ ] Paper 1 abstract sentences (verbatim or lightly edited)
- [ ] Paper 1 introduction paragraphs on model design
- [ ] Paper 1 RF-HSTU architecture figure (`fig1_architecture_tikz.tex`)
- [ ] Paper 1 cross-day main results as Paper 2 main results
- [ ] Paper 1 Method section network structure (CNN stem, HSTU blocks, fusion equations) beyond 1-paragraph summary
- [ ] Paper 1 claim framing: "source-only cross-day robustness" as **primary** contribution
- [ ] Paper 1 Table I / cross-day bar charts as Paper 2 hero results
- [ ] Paper 1 conclusion wording on architecture superiority

---

## Table C: Permitted brief reuse

Paper 2 **may** briefly state:

- [x] RF-HSTU is used as a **frozen backbone** from prior/submitted work
- [x] OOB cross-attention model from Paper 1 provides the feature extractor
- [x] Paper 1 reported cross-receiver as an open limitation (~18% RX1→RX2)
- [x] Paper 2 proposes **diagnosis-first calibration**, not a new backbone
- [x] Same 24-class LoRa dataset family (with new split protocol documented)

---

## Final risk rating

**Overall: Medium**

- **Medium if submitted before Paper 1 decision** — editors/reviewers may flag companion-paper relationship; requires cover letter disclosure and distinct writing.
- **Manageable after Paper 1 decision** — if accepted, cite clearly; if rejected for cross-RX gap, reassess merge vs split (see `POST_DECISION_PLAN.md`).

**Human decision needed before submission:**
1. Confirm Paper 1 submission status for cover letter Version A vs B.
2. Confirm target venue (IoTJ companion vs TIFS/TWC).
3. Legal/institutional policy on overlapping datasets from same lab.
