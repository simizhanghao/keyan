# Revised Paper-B route lock

Date: 2026-08-23

## Frozen interpretation

The paper is a cross-architecture, multi-receiver OOB-shortcut mechanism
study with a negative RSA mitigation boundary. It is not currently an F0
method paper and makes no authentication claim.

## Execution order

```text
X0-X3                 complete
X4-A                  CT vs RSA stress boundary (inference only)      complete
X4-B                  True physical in-band control                  complete
X4-B2                 Matched-split fairness repair                   complete
X4-C                  Shen-style receiver-agnostic baseline           formal in progress
claim/protocol freeze                                            complete
development data checksum audit                                  complete
final training manifest                                          partial; waits for X4-C
X6                    official six blind receivers
integrated OSU + 20-SDR analysis
paper draft
```

No third-dataset search is required before X4-A. OSU remains a separate
secondary dataset and cannot reopen a failed 20-SDR gate.

## Current claim set

1. OOB contains substantial device-discriminative utility.
2. Learned OOB reliance is entangled with receiver-sensitive relative spectral
   scale.
3. Frequency asymmetry is a development-set finding that must replicate on
   blind receivers to remain a primary result.
4. The evaluated RSA recipe does not outperform matched CT on clean
   unseen-receiver utility; this is a boundary result, not a universal
   impossibility claim.
