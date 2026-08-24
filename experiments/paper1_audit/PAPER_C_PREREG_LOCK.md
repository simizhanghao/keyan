# Paper C Few-Shot Calibration Preregistration Lock

Status: PROTOCOL-ONLY FREEZE BEFORE X6; NO PAPER C TRAINING AUTHORIZED.

## Question

When source-only receiver generalization is insufficient, how much labeled
target-receiver enrollment is required for reliable closed-set LoRa transmitter
identification?

## Frozen protocol

- Target labels are permitted only in Paper C, never in Paper B.
- Enrollment budget per DUT: `K = {1, 3, 5, 10, 20}`.
- Enrollment capture must differ from query capture. If session identifiers are
  available, enrollment and query sessions must also differ.
- Baselines: frozen source model, Nearest Class Mean/prototype, linear probe,
  full or prespecified few-layer fine-tuning, and RCPA.
- Metrics: receiver-wise Accuracy and Macro-F1, calibration cost, seed
  variability, and receiver-level mean/median.
- Receiver is the top-level statistical unit; packets and seeds are not treated
  as independent target domains.

## Publication gate

RCPA becomes a methods-paper candidate only if its median gain over the best
simple calibration baseline is positive and it wins on at least 4 of 6 target
receivers across multiple K values. A mean gain near 3-5 pp would constitute
meaningful supporting evidence. If the gate fails, Paper C may become a
calibration study only when independent enrollment/query data exhibit stable
multi-receiver trends; otherwise Paper C stops.

Opening X6 for Paper B does not authorize Paper C tuning or training.
