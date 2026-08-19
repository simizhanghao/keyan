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

## 2A-5 Cross-band leakage intervention (Human GO 2026-08-19)

This is **not** a generic OOB canonicalizer, RCOF, or utility gate.  
Hypothesis (frozen before C train): full-spectrum `log_zscore` lets OOB scale rewrite in-band FFT values (cross-band normalization leakage). 2A-4 killed the **stem**; 2A-5 asks whether that is an **end-to-end classification shortcut**.

2×2 (complete; CLOSED after D clean FAIL):

| | OOB = C' `ratio` | OOB = C1 `ratio_rms` |
| --- | --- | --- |
| FFT = full | **A** frozen C' | **B** C1 (clean 1/2 FAIL; stress NOT_TRANSFERRED 23.2) |
| FFT = C_fft `inband` | **C** clean 2/2 PASS; stress **NOT_KILLED** 31.4 | **D** clean **FAIL** 1/2; stress closed |

Cell **C** unique change: `--fft-source inband --oob-norm ratio`.  
IQ / amp_phase from full IQ. OOB stays C' ratio. `fft_norm` stays `log_zscore`. Do **not** implement in-band-only z-score on the full spectrum.

```text
flag     --fft-source {full,inband}   default full (frozen C'/C1 unchanged)
C_fft    reconstruct_inband_iq → same log_zscore on that spectrum
```

### Beat 1 — operator smoke (this GO)

No accuracy. No training.

| Test | Pass |
| --- | --- |
| 1 deterministic | two C_fft FFT calls, rel-L2 < 1e-6 |
| 2 OOB-scale | C_fft FFT rel-L2 < 1e-3; full FFT rel-L2 ≥ 0.01 |
| 3 other paths | IQ / amp_phase / ratio-OOB match `fft_source=full` (rel-L2 < 1e-6) |

Fail any test = implementation bug; do not train.

### Beat 1 result — SMOKE_PASS

2 Day4 files, 4 windows. No accuracy.

| Test | rel-L2 | Gate | Reading |
| --- | ---: | --- | --- |
| 1 deterministic | 0 | < 1e-6 | PASS |
| 2 C_fft under oob_scale | 5.19e-6 | < 1e-3 | PASS |
| 2 full FFT under oob_scale | 0.0398 | ≥ 0.01 | PASS (perturbation applied) |
| 3 IQ / amp_phase / ratio-OOB | 0 / 0 / 0 | < 1e-6 | PASS |

Operator is wired. This is **not** an end-to-end kill and **not** a paper result. Source: `results/cfft_operator_smoke/cfft_operator_smoke.json`.

### Beat 2 — C seed 0/1 train (Human GO 2026-08-19)

Same 1C recipe. Day1–3 / Day4. Day5 unused. RX2 unused. No retune.

Clean: `Δ = Acc(C_fft) − Acc(C')` on window. C' seed 0 = 43.8, seed 1 = 44.3.  
**PASS** = both seeds `Δ ≥ −2 pp` and neither collapses. **FAIL** = any seed ~20s / chance; stop; no lr/epoch/seed rescue.

### Beat 2 result — CLEAN_PASS (2/2)

| Seed | C' window | C_fft window | Δ pp | Gate | C' file | C_fft file |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 43.8 | 46.2 | +2.4 | PASS | 70.8 | 70.8 |
| 1 | 44.3 | 46.1 | +1.8 | PASS | 87.5 | 75.0 |

Both seeds above C' on window; neither collapsed. File-Acc recorded only (seed 1 file −12.5 pp does not move the gate).  
Unlike C1 (1/2 FAIL), C_fft is a drop-in-capable **clean** train target for these two seeds. This does **not** mean the scale shortcut is killed. Source: `results/matched_seed0/cfft_clean_vs_cprime.json`.

### Beat 3 — C stress (Human GO 2026-08-19; eval only)

Eval-only `oob_scale` and full RX on **seeds 0 and 1**. No training. Day5 unused.  
`D = Acc_clean − Acc_stress` on window, vs each seed's own C_fft clean. Headline reading uses the **mean oob_scale drop**. Per-seed bins are recorded; if they disagree, headline is `DISAGREE_*` and D stays closed.

| D oob_scale | Reading |
| --- | --- |
| < 5 pp | Main-FFT shortcut **KILLED** (C1 may be unnecessary) |
| 5–15 pp | **PARTIAL** → may open D |
| ≥ 15 pp | **NOT_KILLED** → D more necessary |

### Beat 3 result — NOT_KILLED (2/2 agree)

| Seed | C_fft clean | oob_scale | Δ oob | full RX | Δ full | C' oob Δ | C' full Δ | Bin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 46.2 | 14.2 | 32.0 | 11.6 | 34.6 | 25.5 | 28.5 | NOT_KILLED |
| 1 | 46.1 | 15.4 | 30.7 | 12.2 | 33.8 | 28.5 | 29.3 | NOT_KILLED |

Mean oob_scale drop **31.4** (≥ 15). Mean full RX drop 34.2.  
Both seeds in the same bin. Headline **NOT_KILLED**.

Dissociation (do not collapse these two facts):
- 2A-4 / smoke: C_fft **view** is STABLE under oob_scale (rel-L2 5.19e-6).
- 2A-5 Beat 3: the **trained classifier** still collapses, and the drop is **larger** than C' (31.4 vs 28.7 / 25.5–28.5).

C_fft-only removed a view-level leak but did not kill the end-to-end scale shortcut. Remaining scale path is the OOB `ratio` branch (and/or IQ / amp_phase). Do **not** write “in-band FFT failed as an operator”. Do **not** write “C_fft is more RX-robust”.

D = C_fft + C1 is **more necessary** by the locked table. Source: `results/matched_seed0/cfft_rx_stress.json`.

### Beat 4 — D = C_fft + C1 train (Human GO 2026-08-19)

Unique flags: `--fft-source inband --oob-norm ratio_rms`. Name: `C_full_ratio_rms_inband`.  
Same 1C recipe. Seeds 0 and 1. Day1–3 / Day4. Day5 unused. No retune.  
Does not overwrite A/B/C artifacts.

Clean: `Δ = Acc(D) − Acc(C')` on window. C' seed 0 = 43.8, seed 1 = 44.3.  
**PASS** = both seeds `Δ ≥ −2 pp` and neither collapses (`window < 15%`).  
**FAIL** = any seed ~20s / chance; stop; no lr/epoch/seed rescue.  
C1 / C_fft windows are recorded for context only and do **not** move the gate.

C1 clean 1/2 FAIL is a known risk (seed 1 was 24.2). If D fails the same way, that is a result, not a license to retune.

### Beat 4 result — CLEAN_FAIL (1/2)

| Seed | C' win | D win | Δ vs C' | Gate | C1 win | C_fft win | D file |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 43.8 | 53.8 | +10.0 | PASS | 43.4 | 46.2 | 79.2 |
| 1 | 44.3 | 31.9 | −12.4 | FAIL | 24.2 | 46.1 | 66.7 |

Seed 1 did not collapse (`window` 31.9 ≥ 15) but missed `Δ ≥ −2`.  
C_fft seed 1 was 46.1 PASS; C1 seed 1 was 24.2 FAIL. D recovered some of C1's seed-1 loss (24.2→31.9) but not enough to replace C'.  
Seed 0 is the strongest clean window in the 2×2 (53.8). That does **not** move a 1/2 FAIL.

Do **not** retune lr/epoch/K. Do **not** open D stress, 5-seed, Day5, RX2, or utility.  
`ratio_rms` is not a drop-in OOB train target under the frozen recipe, with or without C_fft.  
Source: `results/matched_seed0/d_clean_vs_cprime.json`.

2×2 clean status (window %):

| | OOB `ratio` | OOB `ratio_rms` |
| --- | --- | --- |
| FFT full | A C' 43.8 / 44.3 | B C1 43.4 / **24.2 FAIL** |
| FFT inband | C 46.2 / 46.1 PASS; stress 31.4 **NOT_KILLED** | D 53.8 / **31.9 FAIL** |

No cell is both clean-stable (2/2) **and** scale-killed. C is the only 2/2 clean cell and it is not RX-robust.

## 2A-5 close-out — mechanism freeze (2026-08-19)

2A-5 is **CLOSED**. No further cell in this 2×2. No retune. No D stress.

Three layers must stay separate:

| Layer | What was tested | Result |
| --- | --- | --- |
| Token / view | C1 rel-L2; C_fft FFT under oob_scale | C1 invariant (0); C_fft STABLE (5.19e-6) |
| Clean classifier | 2-seed Day4 vs C', `Δ ≥ −2 pp` | C PASS; B FAIL; D FAIL |
| Stressed classifier | oob_scale drop vs own clean | C1 23.2 NOT_TRANSFERRED; C_fft 31.4 NOT_KILLED |

Allowed claims:
- Dominant synthetic nuisance is OOB **relative magnitude scale** (1C.mech-4 Case 1A).
- Full-spectrum `log_zscore` is a **cross-band normalization leak**: OOB scale rewrites in-band FFT values. Paper 1 `no OOB` = no explicit OOB branch, not OOB-free.
- Killing the leak at the **view** does not kill the **classifier** shortcut. C_fft can even increase the drop (31.4 vs C' 28.7) because the model can lean harder on remaining scale-sensitive paths (`ratio` OOB, IQ, amp_phase).
- `ratio_rms` is scale-invariant as a token and is **not** a drop-in train target under the frozen 1C recipe (B 1/2 FAIL; D 1/2 FAIL, same seed).
- No 2×2 cell is both 2/2-clean and scale-killed. There is **no frozen Paper 2 operator** from this grid.

Forbidden claims:
- “C1 replaces C'.”
- “C_fft is RX-robust / the operator failed.”
- “D is the Paper 2 method” (seed 0 +10 pp does not rescue seed 1 −12.4).
- “Scale shortcut is killed.”
- “amp_phase → cnn_stem” (2A-4: A_amp does not kill the stem).
- Opening 5-seed, RX2, Day5, Oracle, utility, or a title “Receiver-Canonicalized …” from this close-out.

Internal working title retired for Paper 2 method work. Use **Scale-Robust OOB Hybrid** (`PHASE2B0_SCALE_AUG_LOCK.md`).  
Paper 1 wording: current `no OOB` is **without explicit OOB branch**, not OOB-free, unless Main views are also in-band-only (revision reserve).

## Still forbidden

C1/C_fft/D v2, retune `ratio_rms` / z-score / lr / epoch / seed-rescue, Day5, Hann/guard, utility, DCT K-sweep, target-RX model selection, 5-seed S0/S1, C2, open RX2, D stress, `--augment-receiver-style` (wrong operator), true in-band Main baseline. 2B-0 is the only open train path.
