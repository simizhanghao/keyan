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
- Define disruption drop as clean Accuracy minus the mean Accuracy of shuffle and neutral replacement. Define worst-scale drop as clean Accuracy minus the minimum Accuracy across the four non-unit scale conditions. Both are first averaged over five seeds within receiver.
- An architecture reaches GO when both drops are positive on at least 4/6 receivers and both receiver-median drops are at least 5 percentage points. Overall GO requires at least one architecture to reach GO.
- STRONG GO additionally requires left-minus-right drop to be positive on at least 4/6 receivers and have a positive receiver median for an architecture that reaches GO.
- DOWNGRADE: the core utility/sensitivity pattern does not replicate. No model, checkpoint, intervention, or threshold may be changed after opening blind data.

## Execution gate

The runner must first pass a development-only dry run. A final freeze commit must include this protocol, checkpoint/data/code hashes, the neutral artifact hash, and the tested runner. Only then may `blind-confirmatory` mode be invoked.
