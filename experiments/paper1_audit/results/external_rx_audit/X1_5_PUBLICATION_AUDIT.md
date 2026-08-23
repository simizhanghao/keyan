# X1.5 publication-level audit

**Status:** GO for X2 protocol freeze. Paper 2 method remains HOLD. No training; official six blind receivers unopened.

## Independent-unit handling

The 144,000 packets were not treated as 144,000 independent observations. The audit aggregates to 240 `receiver x device x file` cells and uses receiver/device/day cluster summaries. HDF5 has no capture/session IDs, so no stronger independence claim is made.

| Effect summary | Std. of cell/group means |
|---|---:|
| Device | 0.400 dB |
| Receiver | 0.083 dB |
| Day | 0.014 dB |

All 10 DUTs show nonzero receiver variation. The largest is DUT 33 (`0.552 dB`); the other nine range from `0.048` to `0.144 dB`. This is consistency evidence, not a claim of identical effects.

For the two receivers with drift days, within-receiver day variation is `0.007 dB` (N210-1) and `0.030 dB` (RTL-6). The latter is still below the device-level variation and is reported separately rather than pooled into a misleading global day number.

## SNR/CFO sensitivity

At cell level, a descriptive regression `ratio ~ receiver + device + day + SNR_z + CFO_z` gives SNR and CFO coefficients of `-0.163` and `+0.335 dB` per standard deviation. The receiver term remains nonzero after including both covariates (receiver component spread `0.301 dB`). This is a sensitivity analysis, not causal identification; it does not justify an authentication claim.

Machine-readable output: `x1_5_publication_audit.json`.

## Decision

X1 is **Signal-Level GO**: device variation > receiver variation > day drift. X1.5 closes the publication-level statistical gate. The external C' model shortcut and F0 remain untested/HOLD and are deferred to X3/X5. X2 may freeze protocols and baselines; the six official blind receivers remain sealed until X6.
