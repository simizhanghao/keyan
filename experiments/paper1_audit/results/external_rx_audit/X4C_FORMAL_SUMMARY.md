# X4-C Formal Shen-style Baseline Summary

Status: complete, development receivers only, 2026-08-24.

This is a six-fold receiver-held-out evaluation with two seeds and two
Shen-style PyTorch adaptation arms. No official blind receiver was opened.

| Held-out RX | Shen-CIS Acc | Shen-RA Acc | RA-CIS |
|---|---:|---:|---:|
| rtl_2 | 86.481% | 85.812% | -0.669 pp |
| rtl_5 | 89.169% | 88.650% | -0.519 pp |
| b200_1 | 92.681% | 92.269% | -0.413 pp |
| b200_mini_1 | 99.150% | 99.194% | +0.044 pp |
| b210_1 | 98.756% | 98.788% | +0.031 pp |
| pluto_1 | 98.369% | 98.369% | +0.000 pp |

Values are means over seeds 0 and 1, with receiver-fold as the top-level
experimental unit. Across all 12 fold-seed runs:

| Arm | Accuracy | Macro-F1 | Best epoch median |
|---|---:|---:|---:|
| Shen-CIS | 94.101% | 94.001% | 73.5 |
| Shen-RA | 93.847% | 93.744% | 71 |

The paired receiver mean difference is `RA-CIS = -0.254 pp`; the receiver-wise
differences are not directionally consistent (two positive, three negative,
one zero). The result therefore does not support a claim that the adversarial
receiver head improves this protocol. It is reported as a strong matched
baseline and a negative/neutral mitigation result, not as evidence that the
original Shen method is ineffective.

The two arms are a Shen-style adaptation, not an exact official TensorFlow
reproduction. CIS uses no receiver labels; RA uses source receiver labels and
a unit-weight gradient-reversal receiver objective.
