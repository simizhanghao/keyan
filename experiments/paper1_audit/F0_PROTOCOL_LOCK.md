# F0 / F0-CT protocol lock

Status: authorized after X3 `MECHANISM GO` (2026-08-23). Official six blind
receivers remain SEALED. This phase uses only the 14 source receivers and the
same six internal pseudo-unseen folds as X2.

Dataset role: 20-SDR is the primary dataset for this phase. OSU is a separate
secondary replication dataset and is not pooled into F0 gates or metrics.

## Purpose

F0 tests whether controlled OOB-scale augmentation improves robustness after
the X3 mechanism audit. It is not an architecture comparison and does not
reopen tuning.

## Arms

For each fold, backbone (`B1`, `C'`) and seed (`0`, `1`):

1. **Base**: the frozen X2 checkpoint, evaluated unchanged.
2. **CT**: continuation training from that checkpoint with the identical clean
   recipe and budget, to control for extra optimization alone.
3. **F0**: continuation training from the same checkpoint and budget, with
   paired OOB scale augmentation on the training source data; the original and
   scaled OOB views share the device label. No held-out receiver is used for
   training or checkpoint selection.

CT and F0 must use identical optimizer, learning rate, weight decay, epochs,
batch size, source validation checkpoint rule, and seed. The only intended
delta is the paired OOB-scale augmentation.

## Frozen recipe

`AdamW`, `lr=1e-3`, `weight_decay=5e-4`, `epochs=5`, `batch_size=64`; source
validation selects the checkpoint. Scale factors are sampled symmetrically
from the locked X3 range `[0.5, 2.0]` (log-uniform); in-band IQ is unchanged.

Primary report: held-out receiver Accuracy and Macro-F1, paired by fold and
seed. No packet-level pooled confidence interval. No blind receiver access.

F0 is successful only if it is compared against both Base and CT and does not
trade away clean source-validation performance. If F0 fails, do not retune in
this phase; retain X3 as the mechanism result and stop before X6.
