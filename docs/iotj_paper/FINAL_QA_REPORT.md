# IoTJ Final QA Report

Date: 2026-06-23  
Branch: `paper-ready-v3`  
Commit audited: `c727a77` (Polish IoTJ manuscript terminology and final layout)

## Automated Source Audit

| Check | Result |
|-------|--------|
| Hard placeholders | **PASS** — no hits in tex/sections/tables/figures; only benign comment in `refs.bib` line 1 (`paper-ready-v3`, not rendered in PDF) |
| Forbidden claims | **PASS** — no forbidden numeric claims (87.5/58.3/74.6/60.7/LODO/target-val); `receiver-invariant` / `receiver-independent` appear only in negation or future-work context |
| Model label `F` | **PASS** — no `F` column names, `F cross-attn`, `M7`, or `H_gated_chirp_plain` in tables/sections |
| `F_cross_attn_chirp_plain` | **PASS** — appears once in Model Variants (`sections/05_experiments.tex`) as `\texttt{...}` log name |
| Old figure labels | **PASS** — only `fig:architecture` and `fig:results_summary` referenced; no `fig:cross_receiver_stress` or legacy fig labels |
| `\balance` / balance package | **PASS** — removed from `main.tex` |
| Validators | **PASS** — `validate_citations.py`, `validate_latex_structure.py`, `audit_iotj_consistency.py`, `audit_table_layout.py`, `audit_layout_visual.py`, `git diff --check` |

## Reference Counts

- `refs.bib` entries: **31**
- Unique `\cite{}` keys in tex: **26**
- All cited keys: `READ` or `ABSTRACT_CHECKED` in `reference_candidates.csv`

## Overleaf Compile Log (user to confirm on recompile)

- Errors: *(pending user Overleaf recompile — expected 0)*
- Warnings: *(pending — balance warning expected gone)*
- Overfull hbox: *(pending user log)*
- Underfull hbox: *(pending user log)*
- Page count: **11** (per user PDF v12; no source changes this round)
- References: **26** (expected in PDF bibliography)

## PDF Manual Checklist (user v12 confirmed)

| Item | Status |
|------|--------|
| Page 1: title / authors / affiliation / funding | OK |
| Page 1: abstract / index terms | OK |
| Pages 2–3: Related Work density | OK (26 refs) |
| Page 5: Fig.1 not clipped, readable | OK (acceptable, not redraw) |
| Pages 6–8: Tables I–V, `Ours` labels | OK |
| Page 9: Fig.2 legend, Table VI `Ours`, no cross-receiver fig | OK |
| Page 10: Limitations conservative | OK |
| Page 11: Data Availability before References, 26 refs, no internal paths | OK |

## Component Verdict

| Component | Verdict |
|-----------|---------|
| Fig.1 | Acceptable — do not redraw |
| Fig.2 | Acceptable — do not redraw |
| Tables | Acceptable — `Ours` labels correct |
| Data availability | Acceptable — no draft paths |
| Claim boundary | Conservative — cross-receiver framed as stress test / limitation |

## Final Decision

**PASS — ready for submission-prep phase.**

No source edits required this round. Do not modify figures, tables, experiment numbers, or references unless a hard error is found during Overleaf recompile.

Next steps: cover letter, contribution summary, ORCID / affiliation confirmation, repository release policy, open/traditional access choice.
