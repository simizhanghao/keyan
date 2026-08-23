# Paper B claim lock

Status: development claims frozen before official blind evaluation, 2026-08-23.

## Claims allowed

1. OOB carries substantial device-discriminative utility under the audited
   multi-receiver setting.
2. OOB reliance is receiver-sensitive and frequency-structured, demonstrated
   across B1 and C' by controlled scale, shuffle, neutral replacement, and
   left/right interventions plus probes.
3. RSA reduces the tested broadband scale sensitivity but does not outperform
   matched CT on clean unseen-receiver utility and does not remove shuffle or
   left/right dependence.
4. TrueIB and matched-split controls show architecture-dependent dependence on
   OOB evidence; this is not a universal claim about all RFFI models.

## Claims forbidden

- OOB is pure receiver noise or pure device fingerprint.
- RSA is a receiver-robust solution.
- The result is authentication, spoofing detection, or open-set security.
- C' is universally superior to B1.
- Shen-RA is an exact official reproduction. It is a Shen-style PyTorch
  adaptation with protocol differences disclosed explicitly.

The X4-C five-epoch screening is a development check and is not reported. The
separately registered X4-C formal run is the reportable multi-seed baseline.
Under the formal receiver-held-out protocol, Shen-RA is not claimed to improve
over Shen-CIS: its paired mean difference is negative and small.
