# Paper 2 Phase 2A-0 — Scale–shape decoupling (no training)

**Locked after Case 1A.** Do not pick C1 vs C2 on a real target receiver.

## Representations

```text
C0  oob_norm=ratio          Paper 1 C'  (frozen control)
C1  oob_norm=ratio_rms      R / RMS_OOB(R)
C2  oob_norm=ratio_logdc    log(R+ε) − mean_OOB(log(R+ε))
```

`ratio` algebra is unchanged. No DCT, MLP, utility, CORAL, adversarial, or prototype in this step.

Scale `s` is a diagnostic / later reliability feature. It must **not** enter the identity tokens in C1/C2.

## Allowed data

```text
manifest   data/paper/cross_day_day1to5_source_only.csv
days       train+val = Day1–4
Day5       unused
real RX2   unused
perturbation   existing OOB-scale only, lock_inband=1, ranges 0.5–2.0
```

## Tests

1. **Scale invariance (sanity, not a paper result).**  
   Primary metric is **relative L2** `||z-z_a||/||z||`. Cosine is scale-blind and will look “perfect” on C0 even when scale is the failure mode. C1/C2 should drop sharply vs C0.

2. **Device separability (the GO).**  
   File/day grouped means.  
   `ρ = d_same / d_diff` on Day1–4.  
   Want `ρ_can ≤ ρ_raw`, or at least no large rise.  
   Optional probe: Day4 file → nearest Day1–3 centroid. No random window split.

## Decision (source-only)

| Result | Next |
| --- | --- |
| invariance holds and ρ does not collapse | Human GO may open seed 0/1 matched train of **one** C (C1 or C2), vs frozen C' |
| invariance holds but ρ / probe collapses | **NO-GO** for “just delete scale”; keep s as a gated feature |
| invariance fails | implementation bug; do not train |

Do not open seed 0/1 training from this file automatically. Real RX1↔RX2 is later and frozen.

## 2A-0 result (source Day1–4, 96 files, 1536 windows)

| Rep | rel-L2 | rho | Day4→D123 |
| --- | -----: | --: | --------: |
| C0 ratio | 0.4191 | 0.783 | 29.2% |
| **C1 ratio_rms** | **0.0000** | **0.653** | **33.3%** |
| C2 ratio_logdc | 0.0002 | 0.960 | 29.2% |

Source-only decision: **C1 selected**. Invariance holds; ρ improved (0.783→0.653); probe did not collapse (29.2→33.3, chance 4.2%).  
**C2 rejected** (ρ rose to 0.960). Not a target-RX pick.

Human GO given 2026-08-18: train **C1 only**, seeds **0 and 1**, matched 1C recipe, `--oob-norm ratio_rms`.  
Clean Day4 gate: `Δ_window(C1−C') ≥ −2 pp`.  

## 2A-1 result (C1 seed 0/1, Day4, K=256, mean-logits)

| Seed | C' window | C1 window | Δ window | Gate ≥ −2 pp | C' file | C1 file |
| --- | --------: | --------: | -------: | --- | ------: | ------: |
| 0 | 43.8% | 43.4% | −0.3 | PASS | 70.8% | 75.0% |
| 1 | 44.3% | 24.2% | −20.0 | FAIL | 87.5% | 66.7% |

Two-seed clean gate: **FAIL (1/2)**. C1 is **not** a drop-in replacement for C'.  
Do **not** write “C1 already replaces C'”. Do not retune lr/epoch/K.  
OOB-scale / full-RX stress, 5-seed, Day5, RX2, C2, utility stay closed until a later Human GO.  
Source: `results/matched_seed0/c1_clean_vs_cprime.json`.

## Still forbidden

Day5, 1D, 1E, Hann/guard, utility, DCT K-sweep, target-RX model selection, 5-seed canonical training.
