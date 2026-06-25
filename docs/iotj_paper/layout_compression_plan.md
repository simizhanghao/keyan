# Layout Compression Plan

## Current status

- First compiled PDF: **~11 pages** (IEEEtran journal, double column).
- IoTJ over-length page charges may apply beyond journal page budget; treat **≤10 pages** as a practical target before adding more references.
- Adding 8–12 references may add **~0.3–0.6 pages** to bibliography alone.

## Possible compression actions (priority order)

### 1. Appendix / reproducibility (low risk)

- Move Appendix A "Reproducibility" to a **short Data and Code Availability** footnote or one sentence before References.
- Keep GitHub URL if coauthors approve; remove branch name from main text.

### 2. Tables (medium risk — verify readability)

| Table | Action |
|-------|--------|
| Table III deployment-shift (`table3`) | Wrap in `\footnotesize` or `\scriptsize` inside table only; consider moving full table to appendix, keep distance numbers in text + Fig. 4 |
| Table V edge + Table IV cross-RX | Edge benchmark is small — could merge into Discussion as prose + one compact table |
| `table_hyperparams.tex` | Not currently `\input`; either add to appendix or omit |

### 3. Figures (medium risk)

- **Fig. 1** architecture: required; replace matplotlib draft with compact vector figure when ready.
- **Fig. 2–3** (cross-day + ablation): both important; if needed, combine into 2-panel figure with shared legend.
- **Fig. 4** distance shift: keep (clearest deployment story).
- **Fig. 5** cross-receiver: keep for limitation narrative; could move to appendix if page-critical.

### 4. Vertical whitespace (low risk)

- Use `[!t]` figures (already used).
- Reduce `\abovecaptionskip` / `\belowcaptionskip` locally if needed.
- Avoid `\clearpage` before bibliography.

### 5. Text (last resort — do NOT cut claims)

- Do **not** delete limitation / cross-receiver / protocol-boundary sentences.
- Minor trims OK in Related Work if redundant sentences exist after reference expansion.
- Conclusion is already compact.

## Suggested target after reference expansion

| Scenario | Action |
|----------|--------|
| PDF ≤ 10 pages after refs | Submit as-is |
| PDF 11–12 pages | Apply items 1–4 above |
| PDF > 12 pages | Move deployment table + cross-RX figure to appendix |

## Do not compress

- Cross-day main results table (Table I)
- Fusion/chirp ablation table (Table II) — core mechanism evidence
- Protocol variants table in §V Experiments
- Limitation paragraphs in Discussion / Results cross-RX subsection
