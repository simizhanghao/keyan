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
Source: `results/matched_seed0/c1_clean_vs_cprime.json`.

## 2A-2 C1 seed 0 RX stress (Human GO 2026-08-19)

Eval only. No training. **Seed 0 only** (the clean-gate pass). Seed 1 unused. Day5 unused. Real RX2 unused. C' R0/R6 / 7-arm not rerun.

```text
ckpt     runs/C_full_ratio_rms/seed_0/best.pt
oob_norm ratio_rms
arms     oob_scale  |  full RX-style (tilt+oob_scale+gain+phase+noise)
lock_inband 1
K=256, mean-logits, Day4 val
```

Frozen comparators (not moved after seeing C1 numbers):

| Comparator | window drop pp |
| --- | ---: |
| C' mean oob_scale | 28.7 |
| C' mean full RX | 30.3 |
| C' seed 0 oob_scale | 25.5 |
| C' seed 0 full RX | 28.5 |
| C1 seed 0 clean | 43.4 |

Primary: `D = Acc(C1 clean) − Acc(C1 stressed)` on **window**. File-Acc recorded, not deciding.

Registered reading:

| C1 oob_scale drop | Reading |
| --- | --- |
| < 5 pp | scale shortcut **KILLED** on this trained seed |
| 5–15 pp | **PARTIAL** |
| ≥ 15 pp | **NOT TRANSFERRED** (tokens invariant, classifier still uses scale) |

Full-RX is diagnostic vs the oob_scale arm, not a second GO.  
This seed-0 table is **not** a 5-seed claim and does **not** authorize C1 as a C' replacement.

## 2A-2 result (C1 seed 0, Day4, K=256)

| Arm | C1 clean | C1 stressed | C1 drop | C' seed0 drop | C' mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| oob_scale | 43.4 | 20.2 | **23.2** | 25.5 | 28.7 |
| full_rx | 43.4 | 14.9 | 28.5 | 28.5 | 30.3 |

File-Acc recorded only: oob_scale drop 25.0 pp; full-RX drop 33.3 pp.

**Reading: NOT_TRANSFERRED.** 23.2 ≥ 15. C1 tokens were scale-invariant in 2A-0 (rel-L2 0), but this trained classifier still dies under the same `oob_scale` operator. C1 drop ≈ C' seed 0 (23.2 vs 25.5; full-RX both 28.5).  
Do **not** write “C1 killed the scale shortcut”. Do not treat C1 as an RX-robust train target. Source: `results/matched_seed0/c1_seed0_rx_stress.json`.

Likely remaining path (not a new gate): 2A-0 only checked OOB tokens. Main-path views can still move. That is 2A-3.

## 2A-3 scale path leak (Human GO 2026-08-19)

No training. Day4 val only. Same `oob_scale` + `lock_inband=1` as 2A-2.  
Eval path (do not change it): dataset `iq_rms` **once** → RX → model views. **No second RMS.**

| Path | What it is | Expect if hypothesis holds |
| --- | --- | --- |
| `oob_c1` | C1 OOB tokens | **STABLE** (else 2A-0 broken) |
| `fft_inband_linear` | in-band \|X\| | **STABLE** (else lock_inband broken) |
| `fft_log_zscore` | model FFT view (full-spectrum z-score) | may be **LIVE** |
| `fft_inband_of_log_zscore` | in-band bins of that view | may be **LIVE** (z-score shift) |
| `iq_time` | time-domain IQ into CNN | may be **LIVE** |
| `amp_phase` | envelope / phase | may be **LIVE** |
| `cnn_stem` | C1 seed 0 main tokens | may be **LIVE** |

Registered reading (do not move after seeing numbers):

| rel-L2 | Reading |
| --- | --- |
| < 0.01 | **STABLE** |
| 0.01–0.05 | **WEAK** |
| ≥ 0.05 | **LIVE** |

`LEAK_CONFIRMED` if `oob_c1` is STABLE **and** at least one of `{fft_log_zscore, fft_inband_of_log_zscore, iq_time, amp_phase, cnn_stem}` is LIVE.

This does **not** authorize changing `fft_norm` / `input_norm`, opening RX2, or starting a new train.

## 2A-3 result (Day4, 24 files, 384 windows)

Official table is the full Day4 run, not smoke.

| Path | rel-L2 | Reading |
| --- | -----: | --- |
| oob_c1 | 0.0000 | STABLE |
| fft_inband_linear | 0.0000 | STABLE |
| fft_log_zscore | 0.0488 | WEAK |
| fft_inband_of_log_zscore | 0.0213 | WEAK |
| iq_time | 0.0458 | WEAK |
| **amp_phase** | **0.1192** | **LIVE** |
| **cnn_stem** | **0.0591** | **LIVE** |

`rms(after RX)/rms(before) = 1.0046`.  
**Verdict: LEAK_CONFIRMED.** Live paths: `amp_phase`, `cnn_stem`.

Do **not** promote `fft_log_zscore` 0.0488 to LIVE. Smoke (4 windows) called it LIVE; Day4 does not.  
Rejected: second `iq_rms` / total-energy leak; broken `lock_inband`; C1 token non-invariance.  
Remaining path: OOB IFFT changes the time-domain envelope/phase; CNN stem (which concatenates IQ + FFT + amp/phase) still sees scale. Source: `results/scale_path_leak/scale_path_leak_day4.json`.

## 2A-4 in-band view ablation (Human GO 2026-08-19)

No training. Frozen C1 seed 0 `cnn_stem` only. Day4 val. Same `oob_scale` + `lock_inband=1`.  
Do **not** change `features.py` / `models.py` / `evaluate.py`.  
`E_all_inband` STABLE is a **control** (follows from lock_inband), not a paper finding. Do not write “we discovered in-band recon is invariant”.

| Arm | iq | fft | amp_phase | Role |
| --- | --- | --- | --- | --- |
| R0_full | full | full | full | must stay **LIVE** (2A-3 replicate) |
| A_amp | full | full | inband | smallest interesting kill |
| B_iq | inband | full | full | |
| C_fft | full | inband | full | |
| D_iq_amp | inband | full | inband | |
| E_all_inband | inband | inband | inband | control; must be **STABLE** |

Same thresholds as 2A-3. Decision: **smallest arm in order A→B→C→D→E** whose stem rel-L2 is STABLE. That arm is the candidate operator.  
`NO_KILL` (even E fails) = implementation bug; stop.  
This does **not** authorize training, RX2, or changing default norms.

## 2A-4 result (Day4, 24 files, 384 windows)

Official table is the full Day4 run. R0 matches 2A-3 `cnn_stem` 0.0591.

| Arm | stem rel-L2 | Reading |
| --- | -----: | --- |
| R0_full | 0.0591 | LIVE |
| A_amp | 0.0590 | LIVE |
| B_iq | 0.0590 | LIVE |
| **C_fft** | **0.0021** | **STABLE** |
| D_iq_amp | 0.0590 | LIVE |
| E_all_inband | 0.0000 | STABLE (control) |

**Verdict: SMALLEST_KILL = C_fft.**  
A_amp does **not** kill the stem. 2A-3 `amp_phase` LIVE is a **view** movement, not the stem-causal path. Do not keep writing “amp_phase → cnn_stem”.  
Do not promote 2A-3 `fft_log_zscore` 0.0488 to LIVE. The stem still uses that WEAK FFT shift.

Candidate operator (locked, **not** a train GO): compute the **FFT view from in-band-reconstructed IQ**; leave full-IQ `iq` / `amp_phase` / C1 OOB as they are.  
Do **not** swap this for “in-band-only z-score on the full spectrum” after seeing the table — that is a different operator.  
Source: `results/inband_view_ablation/inband_view_ablation_day4.json`.

## Still forbidden

Day5, 1D, 1E, Hann/guard, utility, DCT K-sweep, target-RX model selection, 5-seed canonical training, seed 1 stress, C2, retune lr/epoch/K, change `fft_norm` / `input_norm`, open RX2 from C1.
