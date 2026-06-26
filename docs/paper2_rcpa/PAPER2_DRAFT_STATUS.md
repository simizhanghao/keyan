# Paper 2 Manuscript Draft Status (v1)

> **Branch:** `thesis-rffi-extension`  
> **Date:** 2026-06-26  
> **Goal:** Submission-ready draft for advisor review — **do not submit** until Paper 1 IoTJ decision.

---

## Version history

| Version | Milestone |
|---------|-----------|
| v0 | Full manuscript skeleton + frozen RCPA/TTA/OOB-Eq results |
| **v1** | Same-protocol SOTA-style baselines integrated into Results/Discussion/Related Work |

---

## Compile status

```bash
cd docs/paper2_rcpa
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Overleaf pack: `docs/paper2_rcpa/paper2_overleaf_pack_v0.zip` (re-zip after v1 edits if uploading)

| Item | Status |
|------|--------|
| `main.tex` | ✅ Complete |
| Tables 1–3 + **SOTA baseline table** | ✅ |
| Figures | ✅ diagnosis + shot curve |
| BibTeX | ✅ 38 entries; see `REFERENCE_AUDIT.md` |

---

## Section completion (v1)

| Section | Status | v1 changes |
|---------|--------|------------|
| Abstract | ✅ | Unchanged; claims remain non-SOTA |
| Introduction | ✅ | — |
| Related Work | ✅ | Expanded to 38 refs; apples-to-apples disclaimer |
| Diagnosis | ✅ | — |
| RCPA Method | ✅ | — |
| Experiments | ✅ | — |
| **Results** | ✅ | **New § Same-protocol baseline comparison + Table** |
| **Discussion** | ✅ | **New § RCPA-T vs K-shot parametric calibration** |
| Conclusion | ✅ | — |

---

## Frozen results + new baselines

| Claim | Source |
|-------|--------|
| RCPA-T K=5/10 pooled | `full_20260626_1720` (commit `e33120a`) |
| Linear probe K=5/10 | `sota_style_baselines_20260626_1819` |
| Head FT, CORAL, mean-shift | same |
| TTA / threshold sweep | `d26ce8d` auxiliary |

**Key v1 narrative (safe):**
- RCPA-T **competitive** with K-shot linear probe (K=5: 58.3% vs 59.0%; within std)
- RCPA-T **higher and more stable** at K=10 (69.4% vs 65.7%; std 9.7 vs 11.2)
- Head FT below RCPA-T; unsupervised methods far below K-shot
- **Not** claiming comprehensive SOTA superiority

---

## Supporting documents

| Document | Path |
|----------|------|
| SOTA baseline report | `experiments/cross_receiver_calibration/SOTA_STYLE_BASELINE_REPORT.md` |
| Reference audit | `docs/paper2_rcpa/REFERENCE_AUDIT.md` |
| Overlap audit | `experiments/cross_receiver_calibration/PAPER2_OVERLAP_AUDIT.md` |
| Overlap self-check | `docs/paper2_rcpa/PAPER2_DRAFT_OVERLAP_CHECK.md` |
| Cover letter draft | `docs/paper2_rcpa/COVER_LETTER_DISCLOSURE_DRAFT.md` |
| Post-decision plan | `docs/paper2_rcpa/POST_DECISION_PLAN.md` |
| Paper 1 revision plan | `docs/iotj_paper/REVISION_RISK_PLAN.md` |

---

## Remaining gaps (pre-submission)

| Gap | Priority |
|-----|----------|
| Author block / affiliations | High |
| Update `paper1_placeholder` after IoTJ decision | High |
| Re-zip Overleaf pack with v1 tex | Medium |
| Appendix LaTeX for TTA threshold table | Low |
| English polish | Medium |

---

## Experiments explicitly NOT run

- cross-day RCPA sanity
- OOB-Eq full mode
- Full SCRFFI / adversarial receiver-agnostic reproduction
- New backbone / new main method

---

## Strategy

```text
Paper 2 v1 complete → advisor review → wait for Paper 1 IoTJ decision → POST_DECISION_PLAN.md
```

**One-line verdict:** Same-protocol baselines integrated; manuscript v1 ready for advisor review. No further main experiments needed before Paper 1 decision.
