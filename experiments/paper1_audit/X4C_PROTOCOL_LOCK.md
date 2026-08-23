# X4-C Receiver-Agnostic Baseline

Status: FORMAL RECIPE LOCKED, 2026-08-23.

## Purpose

Provide one representative Shen/Zhang receiver-agnostic comparison on the
same 20-SDR task. This is a method-family baseline, not an architecture claim.

## Fair protocol

- Same six pseudo-unseen folds as X2/X4-B.
- Fourteen development receivers only for training and source validation.
- Six official receivers remain sealed.
- Ten DUT labels are the device target; source receiver ID is the adversarial
  domain label.
- Held-out receiver is evaluation only; no target data or checkpoint selection.
- Report Accuracy/Macro-F1 by receiver fold and seed, with receiver-level
  aggregation.

## Implementation rule

The released reference implementation uses TensorFlow/Keras. The formal runs
use a clearly labeled `Shen-style PyTorch adaptation`, not an exact
reproduction. Both arms use the native `52 x 126` channel-independent
spectrogram and the official-equivalent residual/pooling/flatten topology.

## Locked formal recipe

- Arms: `Shen-CIS` and `Shen-RA`.
- Six folds, seeds `0,1`: 24 runs total.
- Batch size 64; SGD, learning rate `1e-3`, momentum `0.9`.
- Maximum 500 epochs; early-stopping patience 20.
- `ReduceLROnPlateau`, factor `0.2`, patience 10.
- Source validation accuracy alone selects the checkpoint.
- Held-out receiver is evaluated once for Accuracy and Macro-F1.
- Shen-RA uses a gradient-reversal receiver head with unit loss weight;
  Shen-CIS does not consume receiver labels.
- TensorBoard records train/source-validation accuracy, source-validation
  Macro-F1, and learning rate.

## Comparison assumptions

The baseline is allowed to use source receiver labels, as required by its
adversarial objective. This assumption must be explicit beside B1, C', and
TrueIB, which do not use receiver labels.
