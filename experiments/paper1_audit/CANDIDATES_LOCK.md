# 1B Decision Lock — two OOB preprocesses for 1C

**Locked:** 2026-08-18 after Day1–4 spectral audit  
**Day5:** not used  
**Training:** none  
**Frozen Paper 1 numbers:** unchanged

## Gate

| Item | Result |
|------|--------|
| files | 96 (Day1–4), Day5 absent |
| `rho_day < 1` | yes for 25/30 configs |
| leakage RED (Hann+guard wipes all device info) | **no** |
| rectangular + guard on zscore | probe 25.0% → **8.3%** (band-edge dependence) |
| decision | **GO_TWO_CANDIDATES** — enter 1C, do not open RCOF |

OOB is a **weak but real** same-RX fingerprint (`rho` 0.80–0.95, Day4 centroid 25–33% vs chance 4.2%). It is not a collapse, and it is not a strong standalone ID.

## Locked candidates (exactly two)

| Role | `oob_norm` | FFT window | guard | Day4 probe | rho |
|------|------------|------------|------:|-----------:|----:|
| A Paper 1 control | `zscore` (legacy) | rectangular | 0 | 25.0% | 0.946 |
| B corrected | `ratio` | rectangular | 0 | 33.3% | 0.804 |

Both already exist in `features.py`. 1C does **not** need new zscore algebra or Hann for the training grid.

## What 1B does *not* claim

- P0/P1/P2 look identical here because distances used **OOB bins only**. The neural OOB tensor still contains legacy in-band fill `−μ/σ`. 1C zscore is the true Paper 1 input, including that artifact.
- Rectangular zscore **needs the ±62.5–75 kHz skirt**. Hann keeps ~25–29% even with a 12.5 kHz guard, so this is **partial leakage dependence**, not a full RED.
- `log_ratio` is rejected (`rho≈1`, probe down to chance).
- Centroid probe ≠ RF-HSTU File-Acc. 1C can still go YELLOW if matched Main ≈ Full.

## 1C recipe when approved

Seed0 first, Day4 for checkpointing, Day5 sealed:

```text
A  CNN-IQ
B  Exact Main-only  (CNN-stem + RF-HSTU + chirp, --no-oob)
C  Full-OOB zscore  (Paper 1 control)
C' Full-OOB ratio   (1B winner)
```

Do not add Hann, guard-band, or corrected_zscore until seed0 of these four is interpreted.
