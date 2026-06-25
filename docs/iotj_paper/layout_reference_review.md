# IoTJ Layout Reference Review

## Reference material checked

- IEEE IoTJ author guidelines.
- IEEEtran journal paper template.
- Publicly accessible LoRa/RFFI or RF fingerprinting papers, where downloadable.

## Observed style requirements

1. Architecture figure
- Should be vector-style, clean, and readable at IEEE two-column scale.
- Should not look like a debugging block diagram.
- Main data path and auxiliary branch should be visually separated.
- Cross-attention should explicitly show Q from main branch and K/V from auxiliary branch.
- The final classifier/output should be inside the canvas with safe margin.
- Captions explain the flow; the figure itself should not contain too much prose.

2. Results figures
- Multi-panel summary figures are acceptable, but labels must be self-explanatory.
- Avoid cryptic x-axis labels such as "conca -c".
- Avoid oversized legends and crowded gridlines.
- Use consistent y-axis units and panel labels.

3. Tables
- Do not enlarge small tables just to fill a column.
- Use booktabs-style rules.
- Use short but interpretable column names.
- Use max-width guards only to shrink wide tables, not to enlarge narrow tables.

4. Appendix / reproducibility
- A raw appendix with a URL and output path is visually poor.
- For initial submission, prefer a short Data and Code Availability paragraph or a footnote.
- Do not leave branch/path text as a large appendix block unless the journal explicitly asks for it.

## Current manuscript actions

- Redesign Fig.1 as a professional full-width architecture figure.
- Redesign Fig.2 as a clean three-panel summary with readable labels.
- Remove Appendix A Reproducibility and replace with a compact Data and Code Availability statement before References or in a footnote.
- Recompile and inspect pages containing Fig.1, Fig.2, and References.
