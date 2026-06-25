# IoTJ / IEEE Layout Reference Review

## Checked references

- IEEE IoTJ author guideline page.
- IEEEtran journal layout style.
- Shen et al., LoRa spectrogram CNN RFFI paper (arXiv:2101.01668, INFOCOM 2021).
- Shen et al., channel-robust LoRa RFFI paper (arXiv:2107.02867, TIFS 2022).

Rendered locally: `style_refs/rendered/shen2021_lora_spectrogram_cnn-*.png` and `shen2022_channel_robust_lora-*.png` (pages 1--8).

## Observations from Shen 2021 (spectrogram-CNN)

1. **Signal preprocessing first.** Fig. 1 uses compact (a)/(b) subpanels for time-domain preamble and spectrogram, with short axis labels and no decorative grid.
2. **Task chain in text, not in one oversized diagram.** The paper explains CFO compensation and hybrid classification in prose + equations; figures show data representations rather than full end-to-end boxes.
3. **Caption carries interpretation.** Subfigure labels are minimal; the caption states what each panel shows.

## Observations from Shen 2022 (channel-robust RFFI)

1. **System-level Fig. 6 (RFF extractor).** Vertical ResNet-style stack with rounded boxes, layer specs inside boxes (`7x7 conv, 32`), residual skip arrows, and a clear output head (`512 dense`, `L2 normalization`, `RFF`). This reads as a *model*, not a script flowchart.
2. **Grouped experimental figures.** Channel effects (Fig. 3--5) use consistent axis units and multi-panel layout; tables (Table I) use simple two-column booktabs-style rules without forced scaling.
3. **Training vs. inference separation.** Model architecture figure focuses on the extractor; enrollment/database and rogue detection are described in text—avoid cramming every system block into one figure.

## Architecture figure lessons (for our manuscript)

1. A publishable model figure should communicate a system/model mechanism, not just a chain of boxes.
2. Branches should be visually grouped, e.g., main RF branch and auxiliary/OOB branch.
3. Key intermediate representations should be labeled with compact tensor notation, such as RF tokens \(T_m\in\mathbb{R}^{P\times D}\) and OOB tokens \(T_o\in\mathbb{R}^{P\times D}\) with \(P{=}32, D{=}64\).
4. The paper's novelty should be visually obvious. Here it is not "CNN + HSTU" alone, but OOB-guided cross-attention plus gated residual injection.
5. Captions should explain the figure; the figure itself should contain compact labels only.
6. Avoid script-looking plots, oversized boxes, and default matplotlib styling.
7. **Use TikZ / native LaTeX vector figures** so fonts match IEEEtran body text.

## Current manuscript fixes

- Replace matplotlib Fig.1 with a TikZ architecture figure embedded in LaTeX (`figures/fig1_architecture_tikz.tex`).
- Add token-shape labels, Q/K/V, gated-residual formula inset, RF-HSTU lightweight gated mixing, and \(K\)-window mean-logits voting.
- Keep Fig.2 as a summary result figure, but reduce default matplotlib look (serif fonts, lighter grid, readable panel labels).
- Remove Appendix A Reproducibility; use a compact Data and Code Availability paragraph before References.
- Do not commit downloaded style reference PDFs.
