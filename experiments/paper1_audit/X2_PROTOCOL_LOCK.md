# X2 protocol lock — external receiver development

**Status:** frozen before any accuracy is inspected. Applies only to the 14 source/train receivers. The six official test receivers remain sealed until X6.

## Internal pseudo-unseen folds

The following receiver IDs are fixed from the X0 archive inventory and cover same-type and cross-type shifts:

| Fold | Held-out receiver | Type |
|---|---|---|
| P1 | `rtl_2` | RTL, same-type holdout |
| P2 | `rtl_5` | RTL, same-type holdout |
| P3 | `b200_1` | B200 |
| P4 | `b200_mini_1` | B200-mini |
| P5 | `b210_1` | B210 |
| P6 | `pluto_1` | Pluto |

For each fold, all other source receivers are training candidates; the held-out receiver is validation/test for development only. Fold choice, normalization, LR/WD, checkpoints, and mechanism decisions must be made without inspecting the six official blind receivers.

## Model order

1. B0: reproduce the released author spectrogram-CNN loader and labels as a pipeline sanity check.
2. B1: input-and-capacity-matched multi-view CNN, without HSTU or cross-attention.
3. C': audited OOB hybrid, with chirp removed and attribution described as an HSTU-inspired lightweight sequence encoder. Main tokens query an OOB frequency memory; no physical local-token alignment claim.
4. F0 and F0-CT remain closed until X3 mechanism GO. X3 has now reached
   `MECHANISM GO`; the F0 control lock is defined in `F0_PROTOCOL_LOCK.md`.

All models use the same receiver folds, data budget, checkpoint rule, and reporting unit. No new OOB statistic search is permitted in X2.
