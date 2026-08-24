# X6 Official Blind Confirmation Report

Status: **COMPLETE / STRONG GO**

The frozen X6 queue completed all `630 / 630` evaluations across six official blind receivers, five seeds, five model families, and the pre-registered intervention conditions. Every run evaluated `2,000` packets. No training, checkpoint selection, or parameter change occurred after blind opening.

## Clean Accuracy (%; mean over five seeds)

| Receiver | Shen-CIS | Shen-RA | B1-OOB | C'-OOB | C'-TrueIB |
|---|---:|---:|---:|---:|---:|
| b200_2 | 99.48 | 99.48 | 56.17 | 72.15 | 58.42 |
| b200_mini_2 | 99.98 | 100.00 | 60.68 | 88.35 | 59.64 |
| b210_2 | 99.67 | 99.59 | 55.68 | 88.89 | 58.86 |
| n210_2 | 99.99 | 99.99 | 63.01 | 86.23 | 71.61 |
| n210_3 | 99.58 | 99.36 | 66.11 | 79.34 | 66.17 |
| pluto_2 | 96.76 | 96.41 | 67.95 | 78.93 | 66.24 |

Shen-CIS/RA remain strong cross-receiver baselines. The OOB-aware models are not uniformly the best clean classifiers, so the result is a mechanism study rather than a leaderboard claim.

## Receiver-Level Intervention Effects (percentage points)

`disruption_drop` is clean minus the mean of same-receiver OOB shuffle and development-derived neutral replacement. `worst_scale_drop` is clean minus the minimum of the four frozen OOB scale conditions. `left_minus_right_drop` is the left-OOB accuracy drop minus the right-OOB accuracy drop. All quantities were averaged over seeds within receiver before aggregation.

| Model | Receiver | Clean | Disruption drop | Worst scale drop | Left minus right |
|---|---|---:|---:|---:|---:|
| B1-OOB | b200_2 | 56.17 | 43.09 | 46.17 | 28.32 |
| B1-OOB | b200_mini_2 | 60.68 | 47.69 | 50.68 | 29.35 |
| B1-OOB | b210_2 | 55.68 | 42.81 | 45.68 | 26.03 |
| B1-OOB | n210_2 | 63.01 | 50.41 | 52.98 | 34.79 |
| B1-OOB | n210_3 | 66.11 | 52.63 | 56.10 | 34.45 |
| B1-OOB | pluto_2 | 67.95 | 54.74 | 57.95 | 30.42 |
| C'-OOB | b200_2 | 72.15 | 66.22 | 59.30 | 57.45 |
| C'-OOB | b200_mini_2 | 88.35 | 80.69 | 77.98 | 74.57 |
| C'-OOB | b210_2 | 88.89 | 71.30 | 71.68 | 74.44 |
| C'-OOB | n210_2 | 86.23 | 79.20 | 76.28 | 74.32 |
| C'-OOB | n210_3 | 79.34 | 71.19 | 69.64 | 66.74 |
| C'-OOB | pluto_2 | 78.93 | 72.25 | 68.99 | 66.79 |

## Frozen Decision

- B1-OOB: `GO=True`, core-positive receivers `6/6`, median disruption drop `49.05 pp`, median worst-scale drop `51.83 pp`, left-greater-right `6/6`.
- C'-OOB: `GO=True`, core-positive receivers `6/6`, median disruption drop `71.78 pp`, median worst-scale drop `70.66 pp`, left-greater-right `6/6`.
- Overall: **STRONG GO**.

The confirmatory claim is that OOB utility and receiver-sensitive OOB dependence replicate across architectures and all six official blind receivers, with the pre-specified left/right asymmetry also replicating. This does not establish that OOB is universally harmful, and it does not imply that controlled OOB intervention equals strict in-band removal; C'-TrueIB is the bandwidth-path control.
