# Audited C' smoke

**Verdict: PASS; runtime only, no accuracy claim.**

The audited C' uses the existing RF-HSTU implementation with chirp disabled,
`cnn_stem` front-end, `ratio` OOB normalization, and one-way OOB
cross-attention. Main RF tokens query an OOB spectral memory; no temporal/local
one-to-one alignment claim is used.

- output logits: `[2, 10]`
- embedding: `[2, 64]`
- parameters: `625,896`
- official blind receivers: unopened

Machine-readable result: `cprime_audited_smoke.json`.
