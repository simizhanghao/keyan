# Day4 RX-factor attribution

Frozen 1C C' checkpoints. No retraining. Day5 unused. R0/R6 not rerun.
Primary: window drop vs clean C'. Frozen D_full = 30.3±2.0 pp.
File-Acc is recorded and does not decide cases.
Independent RNG per arm: non-additivity ≠ interaction by itself.

## Window drop (pp)

| Arm | s0 | s1 | s2 | s3 | s4 | Mean±Std | vs full 30.3 |
| --- | --: | --: | --: | --: | --: | -------: | -----------: |
| tilt | 5.9 | 5.6 | 3.1 | 5.6 | 5.7 | 5.2±1.2 | -25.1 |
| oob_scale | 25.5 | 28.5 | 32.3 | 28.9 | 28.3 | 28.7±2.4 | -1.6 |
| gain | 7.4 | 0.4 | 8.9 | 4.5 | 8.8 | 6.0±3.6 | -24.3 |
| phase | 0.5 | 0.3 | -0.0 | 0.5 | 0.2 | 0.3±0.2 | -30.0 |
| noise | 0.1 | 0.3 | 0.0 | 0.1 | 0.3 | 0.2±0.1 | -30.1 |
| spec | 28.6 | 28.9 | 33.1 | 31.0 | 30.7 | 30.5±1.8 | +0.2 |
| nonspec | 0.4 | 1.2 | -0.2 | 0.1 | 0.4 | 0.4±0.5 | -29.9 |

## File drop (pp, not deciding)

| Arm | s0 | s1 | s2 | s3 | s4 | Mean±Std |
| --- | --: | --: | --: | --: | --: | -------: |
| tilt | -4.2 | 4.2 | 0.0 | 0.0 | 4.2 | 0.8±3.5 |
| oob_scale | 37.5 | 45.8 | 41.7 | 33.3 | 45.8 | 40.8±5.4 |
| gain | 8.3 | 8.3 | 0.0 | 0.0 | 4.2 | 4.2±4.2 |
| phase | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0±0.0 |
| noise | 0.0 | 4.2 | 0.0 | 0.0 | 0.0 | 0.8±1.9 |
| spec | 37.5 | 41.7 | 41.7 | 33.3 | 54.2 | 41.7±7.8 |
| nonspec | 0.0 | 8.3 | 0.0 | 0.0 | 0.0 | 1.7±3.7 |

Frozen case (computed, does not open training): **magnitude_candidate**
Canonicalizer GO: True
Need OOB-only tilt localization: False

Case 1 is magnitude-family candidate only. Canonicalizer GO requires D_oob_scale >= 15pp. R_spec alone does not authorize DCT. Non-additivity cannot by itself be read as factor interaction: arms use independent perturbation draws plus network nonlinearity.

PAPER1_AUDIT_REPORT / Day5 / 1D / 1E / RCOF / utility / DCT are not opened here.
