# Paper 2 Visual and Review Polish Report

**Date:** 2026-06-26  
**Branch:** `thesis-rffi-extension`  
**Scope:** Visual redraw + internal-term cleanup (no new experiments, no RCPA number changes)

---

## 1. Fig. 1 — Redrawn

| Item | Status |
|------|--------|
| IEEE vector PDF | **Done** — `figures/fig1_diagnosis_summary.pdf` |
| PNG 300 dpi preview | **Done** — `figures/fig1_diagnosis_summary.png` |
| `figure*` double-column | **Done** — `sections/03_diagnosis.tex` |
| No in-figure title | **Done** |
| Colorblind-friendly palette | **Done** (#4C78A8, #F58518, #54A24B, #B279A2) |
| Bar value labels | **Done** |
| Formal caption | **Done** |

**Generator:** `experiments/cross_receiver_calibration/generate_ieee_figures.py`

---

## 2. Fig. 2 — Redrawn

| Item | Status |
|------|--------|
| IEEE vector PDF | **Done** — `figures/fig2_rcpa_shotcurve.pdf` |
| Discrete K ticks {1,3,5,10,20} | **Done** |
| No internal title | **Done** |
| Source classifier gray dashed | **Done** |
| Pooled / direction markers | **Done** |
| Moved to Results only | **Done** — removed from `04_method_rcpa.tex`, added after RCPA main results in `06_results.tex` |

---

## 3. Tables

| Item | Status |
|------|--------|
| Split Table V | **Done** |
| `table5_unlabeled_baselines.tex` | Source classifier, mean-shift, CORAL (no TTA) |
| `table6_kshot_baselines.tex` | Linear probe, head FT, RCPA-T at K=1,5,10 |
| Table I (baseline) | **Done** — TTA columns removed |
| Table III caption | **Done** — removed `full-mode` |
| Legacy `table_sota_style_baselines.tex` | **Cleaned** — no script names / TTA; points to split tables |
| `generate_sota_baseline_report.py` | **Updated** — writes split tables |

---

## 4. Internal Terms Removed

| Term | Action |
|------|--------|
| `quick run` | Removed from main text; → `controlled diagnostic run` in appendix |
| `full-mode` | → `main evaluation` or removed |
| `frozen full-mode results` | **Removed** from figure caption |
| `Phase5-clean` | → `source-receiver training checkpoints` |
| `same_protocol_baselines` / script names | **Removed** from table captions |
| `Venue and narrative fit` | **Deleted** entire subsection |
| `related submitted work` | → `a related submitted manuscript` (consistent) |
| `paper1_placeholder` in PDF body | **Not shown** — BibTeX key only; `@unpublished` in refs.bib |

---

## 5. Venue Narrative

**Deleted** — `sections/07_discussion.tex` subsection H removed.

---

## 6. TTA Quick-Only Handling

| Item | Status |
|------|--------|
| TTA removed from Table I | **Done** |
| TTA removed from main baseline tables | **Done** |
| Appendix diagnostic table | **Done** — `sections/09_appendix.tex`, `tab:tta_diagnostic` |
| Results text | Points to appendix for TTA; not mixed with full aggregate baselines |

---

## 7. Author / Placeholder

| Item | Status |
|------|--------|
| `Author Names TBD` | **Fixed** — Chengcheng Han (first), Ziyang Wang (corresponding) |
| `Affiliations TBD` | **Fixed** — Beihang University; NSFC Grant 62301018 |
| Empty journal on Paper 1 cite | **Fixed** — `@unpublished` in refs.bib |

---

## 8. Compile Status

| Item | Status |
|------|--------|
| Local `pdflatex` | **Not available** on this machine |
| Overleaf compile | **Required** — `pdflatex → bibtex → pdflatex ×2` |
| Expected errors | 0 (after Overleaf compile) |
| BibTeX warnings | 0 expected (`paper1_placeholder` is `@unpublished`) |

**Overleaf pack:** `paper2_overleaf_pack_v1.zip` (rebuilt)

---

## 9. Page Count

Not measured locally (no LaTeX). Recompile on Overleaf and note delta vs. prior PDF (~9 pages).

---

## 10. Remaining Reviewer Risks

1. **Author/affiliation TBD** — must fill before external review.
2. **Paper 1 disclosure** — `a related submitted manuscript`; update after IoTJ decision.
3. **Single indoor RX1/RX2 setup** — stated in limitations.
4. **RCPA-T requires labeled target windows** — prominent in abstract/limitations.
5. **Linear probe parity at K=5** — honestly reported; not claimed as universal SOTA.
6. **Fig. 1 still 2×2 panels** — improved styling; consider Table-only variant if mentor prefers.

---

## Summary Block

```text
Paper2 visual/review polish completed:
Fig1: Redrawn IEEE figure* (7.16"×3.2"), colorblind palette, value labels
Fig2: Redrawn single-column shot curve, discrete K, moved to Results
Tables: Split into unlabeled (Table V-A) + K-shot (Table V-B); Table I cleaned
Internal terms removed: quick run, full-mode, frozen full-mode, Phase5-clean, script names, Venue section
TTA handling: Moved to Appendix diagnostic table; removed from main comparison tables
Venue narrative: Deleted
Compile status: Local pdflatex unavailable; Overleaf zip rebuilt
Warnings: Verify on Overleaf after compile
Page count: Recompile on Overleaf
Remaining reviewer risks: Author TBD, Paper1 status, single setup, K-window labels, linear-probe parity
```
