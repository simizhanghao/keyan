# X3-A OOB-scale audit

Protocol: frozen X2 checkpoints; scales `0.5, 0.7071, 1, 1.4142, 2`; no
training; official blind receivers sealed. Primary statistic is the mean over
two seeds for each held-out receiver. Degradation is clean accuracy minus the
lowest perturbed accuracy.

| held-out receiver | B1 degradation (pp) | C' degradation (pp) |
|---|---:|---:|
| rtl_2 | 46.79 | 16.54 |
| rtl_5 | 55.63 | 38.68 |
| b200_1 | 34.09 | 59.57 |
| b200_mini_1 | 47.78 | 64.87 |
| b210_1 | 49.38 | 53.62 |
| pluto_1 | 37.11 | 75.42 |
| **mean** | **45.13** | **51.45** |
| **median** | **47.28** | **56.60** |

Gate B (`>=5 pp` in at least 4/6 folds) passes for both backbones: 6/6 each.
The effect is therefore not confined to one receiver family. Source-validation
audits also pass (B1 mean 49.01 pp; C' mean 78.50 pp). These are mechanism
audits, not authentication results.

Raw JSON curves:
`x3_scale/heldout/{fold}/{model}/seed_{0,1}.json` and
`x3_scale/source_val/{fold}/{model}/seed_{0,1}.json`.
