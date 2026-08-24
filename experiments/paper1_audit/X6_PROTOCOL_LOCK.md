# X6 Confirmatory Protocol Lock

Status: **PRE-BLIND FROZEN**. The development-only dry run passed on 2026-08-24. Training and model development are closed.

## X6-A: clean zero-target evaluation

- Models: Shen-CIS, Shen-RA, B1-OOB, C'-OOB, and C'-TrueIB.
- Seeds: 0-4; official blind receivers: b200_2, b200_mini_2, b210_2, n210_2, n210_3, pluto_2.
- Every receiver packet is evaluated exactly once. No target labels enter training, selection, normalization, or adaptation.
- Report Accuracy and Macro-F1 per receiver and seed. Receiver is the top-level statistical unit; packet-pooled confidence intervals are prohibited.

## X6-B: frozen mechanism replication

- Models: B1-OOB and C'-OOB only; seeds 0-4; all six official blind receivers.
- Conditions: clean; OOB-branch scale 0.5, 0.70710678, 1.41421356, and 2.0; same-receiver cross-device OOB shuffle; development-derived neutral OOB replacement; left-OOB scale 0.5; right-OOB scale 0.5.
- The intervention changes the model's OOB branch while retaining clean main IQ. It must not be described as strict removal of all OOB information. C'-TrueIB is the strict bandwidth-path control.
- Shuffle donors are deterministic, come from the same receiver, and always have a different transmitter label.
- Neutral OOB is computed once from development receivers only, then hashed and frozen before blind access.

## Analysis lock

- Primary paired quantities are receiver-level changes from clean, first averaged across five seeds within receiver.
- Report all six receiver effects, their median, mean, sign count, and architecture contrast. Do not infer six independent studies from packet count.
- GO: OOB utility and receiver-sensitive degradation replicate on a majority of blind receivers without being driven by one receiver.
- STRONG GO: GO holds and the pre-observed left-greater-than-right sensitivity also replicates on a majority of blind receivers.
- DOWNGRADE: the core utility/sensitivity pattern does not replicate. No model, checkpoint, intervention, or threshold may be changed after opening blind data.

## Execution gate

The runner must first pass a development-only dry run. A final freeze commit must include this protocol, checkpoint/data/code hashes, the neutral artifact hash, and the tested runner. Only then may `blind-confirmatory` mode be invoked.
