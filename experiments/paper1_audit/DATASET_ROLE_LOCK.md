# Dataset role lock

The new 20-SDR external-receiver dataset is the primary dataset for the
revised paper. Its 14 source receivers support development and its six official
blind receivers remain sealed for the final X6 endpoint.

The OSU dataset is retained as a secondary dataset and mechanism-history
source. OSU results are reported separately and are not pooled with 20-SDR
receiver-level estimates, confidence intervals, or gate decisions.

## Separation rules

- Do not transfer checkpoints, normalization statistics, or held-out choices
  between 20-SDR and OSU.
- State the dataset, receiver hierarchy, and analysis unit for every result.
- Cross-dataset mechanism replication is qualitative/protocol-paired, not a
  pooled packet-level significance test.
- 20-SDR X2/X3/F0 gates are the primary decision path; OSU cannot reopen a
  failed 20-SDR gate.
- Official 20-SDR blind receivers remain SEALED until X6.
