# X3-A initial scale audit

Frozen X2 checkpoints were evaluated without training or blind receiver access.
The first paired case is `rtl_2`, seed 0:

| Model | clean | scale 0.5 | scale 0.707 | scale 1.414 | scale 2.0 |
|---|---:|---:|---:|---:|---:|
| B1 | 66.13% | 10.00% | 9.99% | 12.73% | 10.03% |
| C' | 28.70% | 10.01% | 10.25% | 18.66% | 14.48% |

This is an initial mechanism check, not the X3 gate. Both OOB-aware backbones
are strongly scale-sensitive on this receiver, while C' and B1 differ in clean
performance and degradation profile. Full fold/seed curves, shuffle,
occlusion, left/right intervention, and probes remain to be run.
