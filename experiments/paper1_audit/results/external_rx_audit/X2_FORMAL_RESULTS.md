# X2 Formal Pilot Results

**Runs:** 24/24 complete. Official blind receivers were not opened.

| Held-out receiver | B1 seed0 | B1 seed1 | C' seed0 | C' seed1 | C' - B1 mean |
|---|---:|---:|---:|---:|---:|
| `b200_1` | 32.40 | 54.64 | 75.21 | 64.21 | +26.19 pp |
| `b200_mini_1` | 57.26 | 58.19 | 79.36 | 70.43 | +17.17 pp |
| `b210_1` | 64.01 | 53.25 | 59.11 | 63.94 | +2.89 pp |
| `pluto_1` | 39.12 | 59.13 | 88.90 | 81.97 | +36.31 pp |
| `rtl_2` | 66.12 | 47.41 | 28.70 | 24.40 | -30.22 pp |
| `rtl_5` | 65.80 | 63.75 | 49.38 | 51.85 | -14.16 pp |

Packet-level run aggregation (reported descriptively; not an independent-packet
CI) gives B1 `55.09%` mean / `57.73%` median and C' `61.46%` mean / `64.07%`
median. C' has a `+6.36 pp` mean run gain, `+10.03 pp` median fold gain, and
wins 4/6 fold means. C' also has larger seed-run spread (`20.15 pp` vs
`10.68 pp`), driven by receiver-dependent behavior, especially the RTL folds.

## Decision

This meets the pre-registered continuation gate, but it is not a universal C'
win. X3 therefore audits **both B1 and C'** for OOB-scale shortcut behavior.
F0/F0-CT remain closed and the official six blind receivers remain sealed.

Machine-readable result: `x2_formal_summary.json`.
