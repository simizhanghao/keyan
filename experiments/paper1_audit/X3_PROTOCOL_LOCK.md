# X3 model-shortcut protocol lock

X2 has authorized X3 for both B1 and C'. No official blind receiver signal may
be opened. No F0/F0-CT, retuning, or new backbone is allowed.

For each frozen X2 checkpoint, evaluate the same source/held-out internal folds
under only these pre-registered interventions:

1. OOB-scale intervention: multiply OOB magnitude by fixed factors while
   preserving in-band IQ;
2. OOB shuffle: replace OOB with a same-file/different-device donor;
3. OOB occlusion: zero the OOB branch;
4. Left/right OOB intervention: scale left and right spectral OOB halves
   separately;
5. device probe and receiver probe on frozen embeddings.

Primary unit is receiver x seed; packet predictions are aggregated within each
unit. The goal is mechanism replication, not a new accuracy leaderboard.

## X3-A status (2026-08-23)

The frozen-checkpoint scale audit is complete for all 6 folds, 2 seeds, and both
backbones, on both held-out and source-validation splits. Fixed scales were
`0.5, 0.7071, 1, 1.4142, 2`; no training or blind receiver access was used.

Held-out minimum-scale degradation (mean over seeds, percentage points):

| fold | B1 | C' |
|---|---:|---:|
| rtl_2 | 46.79 | 16.54 |
| rtl_5 | 55.63 | 38.68 |
| b200_1 | 34.09 | 59.57 |
| b200_mini_1 | 47.78 | 64.87 |
| b210_1 | 49.38 | 53.62 |
| pluto_1 | 37.11 | 75.42 |
| **mean / median** | **45.13 / 47.28** | **51.45 / 56.60** |

Both B1 and C' meet Gate B on 6/6 folds (>=5 pp) and Gate C. This authorizes
the remaining pre-registered interventions for both backbones. F0/F0-CT remain
CLOSED until X3 is complete.

Raw curves are under
`results/external_rx_audit/x3_scale/{heldout,source_val}/`.

## X3-B status (2026-08-23)

Same-receiver cross-device OOB shuffle is complete for the same 24 checkpoint
evaluations (plus source-validation counterparts). Donor samples were drawn
from the same HDF5 receiver with a different device label; original IQ and
labels were retained. Held-out accuracy degradation (clean minus shuffled,
mean over seeds) was 45.95 pp for B1 and 55.71 pp for C', with 6/6 folds at
the 5 pp threshold for each. This independently confirms an OOB-dependent
shortcut under the locked intervention.

Raw results are under `results/external_rx_audit/x3_shuffle/`.
