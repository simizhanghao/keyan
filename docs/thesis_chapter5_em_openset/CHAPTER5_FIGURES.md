# Chapter 5 Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig. 5-1 | `figures/fig5_1_em_robustness_curves.pdf` | EM robustness curves (AWGN, CFO, NBI, phase, IQ, filter, mixed) |
| Fig. 5-2 | `figures/fig5_2_openset_clean.pdf` | Open-set AUROC / EER / known accuracy by scorer |
| Fig. 5-3 | `figures/fig5_3_em_stress_ranking.pdf` | Perturbation ranking: avg robust acc & accuracy drop |

PNG previews: same basename with `.png`.

**Generator:** `experiments/em_robustness_openset/plot_em_robustness_curves.py`  
**Plot Python:** `/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python` (matplotlib compatible)

**Style:** IEEE / thesis — no internal/smoke labels; y-axis `File-level accuracy (%)`; clean baseline dashed gray.

## Planned (post EM-CR)

- Fig. 5-4: EM-CR vs clean-trained robustness
- Fig. 5-5: Open-set under EM stress
