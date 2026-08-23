# Final blind protocol

Status: LOCKED BEFORE X6; official six receivers remain SEALED.

## Final training rule

No new receiver split is created before blind evaluation. For each headline
model, the fixed epoch budget is computed from development runs as the median
best epoch across P1-P6 and seeds 0/1 (recorded in
`FINAL_TRAINING_MANIFEST.md`). Each final seed 0-4 then trains on all 14
development receivers for exactly that fixed budget. There is no early stopping
or checkpoint selection in final training, and no blind-dependent choice.

## Frozen models

- B1-OOB
- C'-OOB
- C'-TrueIB/MainOnly
- Shen-CIS (Shen-style PyTorch adaptation)
- Shen-RA (Shen-style PyTorch adaptation)

These are the five headline models. `B1-TrueIB` is a supplementary blind
control, not a headline model. RSA/F0 is excluded from blind evaluation.

All preprocessing, receiver-held-out folds, checkpoint rules, metrics, and
receiver-level aggregation are frozen. No blind receiver may select a model or
hyperparameter.

## Confirmatory hypotheses

- H1: OOB shuffle and OOB occlusion each cause positive degradation on most
  blind receivers for OOB-aware models.
- H2: full OOB scale causes positive degradation on most blind receivers.
- H3: left-side degradation exceeds right-side degradation as a development
  hypothesis; failure downgrades it to a development-only observation.
- H4: TrueIB and OOB-aware models exhibit a measurable utility/sensitivity
  trade-off; no direction of accuracy superiority is prespecified.

## Analysis unit

Receiver is the top-level unit, with device/capture/session hierarchy below it.
Seeds are training-randomness replicates, not independent receivers. Report
per-receiver effects, mean, median, and direction counts; do not use pooled
packet-level confidence intervals as the primary test.
