# IoTJ Layout Reference Review

## What to imitate

1. Core architecture figure
- Should be a clean pipeline diagram, usually one full-width or one-column figure.
- It should show modules, inputs, feature flows, and decision output.
- Avoid toy-looking boxes and oversized text.

2. Result figures
- Do not scatter too many small figures.
- Prefer one multi-panel summary figure if several plots support the same experimental story.
- Each panel should have compact labels and consistent font sizes.

3. Tables
- Tables should not be scaled up to fill a column.
- Use IEEE-style thin rules or booktabs.
- Use short headers and consistent units.
- Use `adjustbox` max width or native tabular layout; do not use `resizebox{0.98\columnwidth}` blindly.

4. Page budget
- Avoid pushing appendices or code URLs into the main page count if not necessary.
- Keep limitations text, because it protects claim boundaries.
- Prefer combining figures over deleting important limitation text.

## Current manuscript problems (pre-refactor)

- Table I/III/IV were visually too large because resizebox scales them up.
- Fig.2/Fig.3/Fig.4 were scattered and made Page 8 look like an experiment report.
- Fig.1 was too simple and should be replaced by a professional network architecture figure.
- References are still thin, but reference expansion should happen only after layout is stable.

## Refactor actions (this commit)

- Replace `resizebox{0.98\columnwidth}` with native tabular or `adjustbox{max width=\columnwidth}` (shrink-only).
- Redesign Fig.1 as dual-branch architecture with Q/K/V cross-attention and gated residual.
- Merge Fig.2/3/4 into `fig_results_summary.pdf` (three-panel figure*).
- Keep Fig.5 single-column at 0.92\columnwidth.

## Reference URLs checked

- IoTJ author guidelines: https://ieee-iotj.org/guidelines-for-authors/
- IEEEtran journal template (Overleaf): IEEE journal paper template based on IEEEtran.

Note: IEEE Xplore PDF downloads require institutional access; layout observations follow IEEE two-column journal conventions and IoTJ author guidelines.
