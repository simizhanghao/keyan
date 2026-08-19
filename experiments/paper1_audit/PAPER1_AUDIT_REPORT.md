# Paper 1 Audit Report

**Status: YELLOW — not PASS.**  
**Locked:** 2026-08-18  
**Repo:** `keyan` (`/data1/hcc/llm4RF/new_phase`)  
**Frozen Paper 1 numbers:** `/data1/hcc/llm4RF/outputs/paper_ready_v3/` (read-only)  
**This audit:** Day4 only. Day5 unused. No rewrite of the submitted IoTJ table.

This file is a **revision reserve** and a Paper 2 fact source. It does **not** authorize changing the already-submitted manuscript tonight.

---

## Overall

```text
Overall:                              YELLOW

1. Protocol integrity                 PASS
2. Matched 5-seed stability           YELLOW
   Main 3/5 collapse (training crutch)
   C' window 45.5±1.3  <  CNN 49.9±2.7
3. OOB identity claim                 SHRUNK / NOT SUPPORTED
   trained-subset shuffle mean −5.9 pp
4. Oracle complementarity             CONDITIONAL PASS
   stable-Main seeds 8.7 / 18.7 pp
5. Receiver/style sensitivity         STRONGLY SUPPORTED
   combined RX 30.3±2.0 pp, 5/5
6. Mechanism attribution              Case 1A
   Dominant nuisance = OOB magnitude scale
   D_oob_scale 28.7±2.4 ≈ D_spec 30.5±1.8 ≈ D_full 30.3
   phase / noise / nonspec ≈ 0

Architecture empirical finding:       SUPPORTED (OOB path is predictive)
Stable OOB device-identity story:     NOT SUPPORTED
Paper 2 motivation:                   GO
Paper 2 algorithm:                    S1 FROZEN; 5-seed CLEAN_FAIL (2/5, mean −2.72); S0 SCALE_TAX / Case D; RX2 closed
Utility gate:                         POSTPONED (recompute after a scale-robust method, not after C1)
1D vote (revision reserve):           H4_PASS; 1C K=256 table unchanged
1D per-device:                        MIXED (file 10/3/11; window 9/15)
1E / Day5:                            closed until a later Human GO
```

Safe one-sentence claim:

> Paper 1’s OOB branch is predictive on same-RX Day4, but it is not a stable transmitter identity; it is highly sensitive to OOB relative magnitude scale.

Unsafe claims this audit forbids:

> OOB ≈ stable device fingerprint.  
> Full > CNN is a unique OOB effect.  
> A utility gate can already fall back to Main.  
> Receiver style is a vague entanglement that needs generic DA.

---

## 1. Protocol integrity — PASS

Development protocol (`PROTOCOL_LOCK.md`):

```text
Train Day1–3 / val Day4 / test Day5 sealed
24 classes, raw Device9 excluded
window 8192, K=256, mean-logits
manifest data/paper/cross_day_day1to5_source_only.csv
```

1A machine audit passed. Frozen `paper_ready_v3` was not overwritten. Oracle-target-val manifests were not used for training. Day5 was not used for design, checkpointing, or any 1C.mech table.

1B (no training) selected two OOB norms on Day1–4 only: legacy `zscore` and corrected `ratio`. Hann/guard were not opened.

---

## 2. Matched 5-seed — YELLOW

Recipe freeze: 80 ep, lr 3e-3, bs 128, dim 64, Day4 val-acc checkpoint. C (`zscore`) and C' (`ratio`) reported separately.

**Window-Acc % (primary for mechanism)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean±Std |
| --- | --: | --: | --: | --: | --: | --: |
| B Main | 22.2 | 49.2 | 5.7 | 4.6 | 4.3 | 17.2±19.5 |
| C zscore | 27.9 | 31.8 | 31.2 | 28.0 | 36.8 | 31.1±3.5 |
| C' ratio | 43.8 | 44.3 | 46.1 | 46.5 | 46.7 | **45.5±1.3** |
| A CNN | 52.2 | 52.8 | 46.6 | 50.1 | 47.9 | **49.9±2.7** |

**File-Acc % (K=256; recorded, not the identity claim)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean±Std |
| --- | --: | --: | --: | --: | --: | --: |
| B Main | 41.7 | 62.5 | 4.2 | 4.2 | 4.2 | 23.4±27.2 |
| C zscore | 66.7 | 70.8 | 58.3 | 66.7 | 66.7 | 65.8±4.6 |
| C' ratio | 70.8 | 87.5 | 70.8 | 79.2 | 87.5 | 79.2±8.4 |
| A CNN | 75.0 | 75.0 | 58.3 | 62.5 | 58.3 | 65.8±8.6 |

Hits:

- Main collapsed to chance on seeds 2–4 while Full stayed up → **training crutch**.
- H3 “Full > CNN” fails on **window** for C and C' (0/5). C' File>CNN on 4/5 is still not a mechanism win.
- H4 file-vote (revision reserve): **H4_PASS**. C' mean_logits 65.8/66.7/66.7/73.3/75.8/79.2 at K=8…256. Already > CNN at K=8 (56.7) and K=64 (65.8). CNN saturates at K=64; C' keeps climbing. Not a K=256 spike. Source: `results/matched_seed0/file_vote_k.md`.
- GREEN (Full>Main on ≥4/5 **and** a clean identity story) does **not** fire.

Source: `results/matched_seed0/audit_5seed.md`.

---

## 3. OOB identity — SHRUNK

Shuffle trains a new C' with same-day different-device OOB donors. Window drop = C' − shuffle.

| seed | drop pp | note |
| ---: | ---: | --- |
| 0 | −9.4 | trained; shuffle better |
| 1 | 4.4 | trained |
| 2 | 41.5 | shuffle ≈ chance |
| 3 | 41.5 | shuffle ≈ chance |
| 4 | −12.7 | trained; shuffle better |

all-5 mean 13.1±26.7 is **collapse-dominated**. Trained subset {0,1,4} mean **−5.9 pp**.

Do not write “OOB identity confirmed.” The pairing of OOB to the labeled device is not a stable identity map.

---

## 4. Oracle — CONDITIONAL PASS

`Δ = Acc(Main ∨ C') − max(Main, C')`. Frozen DROP if window Δ < 5 pp. Collapsed Mains are diagnostic, not a moved threshold.

Stable-Main seeds 0/1: **8.7 / 18.7 pp**. Utility gate is **not DROP**.  
`Δ ≥ 5` only means headroom exists. It does **not** prove a learned gate will recover it. Main’s 3/5 collapse still makes `r→0` fallback unsafe.

---

## 5–6. RX-style and factor attribution — STRONG / Case 1A

Combined RX-style (tilt / OOB-scale / gain / phase / noise, in-band scale locked) on frozen C':

```text
window drop 30.3±2.0 pp    5/5    RX-entangled = True
```

R0/R6 were not rerun. 7-arm leave-one-in + families:

| Arm | window drop | vs 30.3 |
| --- | --: | --: |
| tilt | 5.2±1.2 | −25.1 |
| **oob_scale** | **28.7±2.4** | **−1.6** |
| gain | 6.0±3.6 | −24.3 |
| phase | 0.3±0.2 | −30.0 |
| noise | 0.2±0.1 | −30.1 |
| **spec** | **30.5±1.8** | **+0.2** |
| nonspec | 0.4±0.5 | −29.9 |

Frozen rules (15 / 18.2 / 12.1) were set before this table:

- `D_spec ≥ 18.2` and `D_nonspec < 18.2` and phase/noise < 15 → magnitude-family **candidate**
- `D_oob_scale ≥ 15` → **Case 1A, Canonicalizer GO**, OOB-only tilt localization **not required**

Interpretation:

```text
ratio-OOB  R(f) = |X_OOB(f)| / RMS_inband
R'(f) = a R(f)   already drops ~29 pp
```

The network uses **OOB relative scale** as a shortcut. Spectral slope (tilt) and non-amplitude (phase/noise) do not explain the 30.3 pp collapse.

Paper 2 first module is therefore **scale–shape decoupling**, not tilt DCT, not phase robustness, not generic DA.

Source: `results/matched_seed0/rx_style_eval.md`, `rx_factor_attribution.md`.

---

## What this audit does *not* close

| Item | Why closed now |
| --- | --- |
| Day5 sealed test | method not frozen for a one-shot test |
| 1D K-sweep | done H4_PASS; does not choose Paper 2 |
| 1E LODO | revision reserve; 3 seeds × 5 folds later |
| Hann/guard retrain | drop is OOB scale, not band-edge |
| Utility / RCOF | wait for scale-canonical Oracle |
| Submitted IoTJ rewrite | reserve, not tonight |

---

## Paper 2 working title (internal)

```text
Scale-Robust OOB Hybrid
```

Not a manuscript title. The frozen synthetic method is **S1**: C' + `--paired-view oob_scale`. Hard canonicalization is closed.  
Utility stays closed until a later Oracle on this frozen method.  
Paper 1 `no OOB` means **without explicit OOB branch**. A true in-band Main baseline is revision reserve.

Real RX1↔RX2 stays closed until a later GO (optional 5-seed S1 first). See `PHASE2B0_SCALE_AUG_LOCK.md`.

---

## 7. Phase 2A after this audit (source-only)

**2A-0 probe (no training).** Day1–4, 96 files, 1536 windows. Day5 unused. Real RX2 unused.

| Rep | rel-L2 | ρ | Day4→D123 |
| --- | -----: | --: | --------: |
| C0 ratio | 0.4191 | 0.783 | 29.2% |
| C1 ratio_rms | 0.0000 | 0.653 | 33.3% |
| C2 ratio_logdc | 0.0002 | 0.960 | 29.2% |

C1 selected on source-only invariance + ρ. C2 rejected (ρ rose). Not a target-RX pick.

**2A-1 C1 train (seed 0/1 only).** Same 1C recipe; unique change `--oob-norm ratio_rms`. Gate: `Δ_window(C1−C') ≥ −2 pp`.

| Seed | C' window | C1 window | Δ | Gate |
| --- | --------: | --------: | --: | --- |
| 0 | 43.8% | 43.4% | −0.3 | PASS |
| 1 | 44.3% | 24.2% | −20.0 | FAIL |

Two-seed clean: **FAIL (1/2)**. C1 is not authorized as a C' replacement.

**2A-2 C1 seed 0 RX stress (eval only).** Seed 1 unused. Day5 unused.

| Arm | C1 clean | C1 stressed | C1 drop | C' seed0 | C' mean | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| oob_scale | 43.4 | 20.2 | 23.2 | 25.5 | 28.7 | **NOT_TRANSFERRED** (≥15) |
| full_rx | 43.4 | 14.9 | 28.5 | 28.5 | 30.3 | diagnostic; matches C' seed 0 |

C1 did not kill the scale shortcut. Token invariance did not transfer to the trained classifier.

**2A-3 path leak (Day4, 24 files, 384 windows, no training).** Smoke is not the decision table.

| Path | rel-L2 | Reading |
| --- | -----: | --- |
| oob_c1 | 0.0000 | STABLE |
| fft_inband_linear | 0.0000 | STABLE |
| fft_log_zscore | 0.0488 | WEAK |
| iq_time | 0.0458 | WEAK |
| amp_phase | 0.1192 | **LIVE** |
| cnn_stem | 0.0591 | **LIVE** |

**LEAK_CONFIRMED.** Views that move: `amp_phase` LIVE, `cnn_stem` LIVE. Not a second-RMS leak (`rms` ratio 1.0046). Not a broken lock_inband. Do not retune `fft_norm` because 0.0488 is WEAK. Stem-causal path is 2A-4, not this view table.

**2A-4 view ablation (Day4, 24 files, 384 windows, frozen C1 stem).** Smoke is not the decision table.

| Arm | iq | fft | amp_phase | stem rel-L2 | Reading |
| --- | --- | --- | --- | -----: | --- |
| R0_full | full | full | full | 0.0591 | LIVE |
| A_amp | full | full | inband | 0.0590 | LIVE |
| B_iq | inband | full | full | 0.0590 | LIVE |
| **C_fft** | full | **inband** | full | **0.0021** | **STABLE** |
| D_iq_amp | inband | full | inband | 0.0590 | LIVE |
| E_all_inband | inband | inband | inband | 0.0000 | STABLE (control) |

**SMALLEST_KILL = C_fft.** Replacing only amp/phase does not kill the stem. Candidate operator: FFT view from in-band-reconstructed IQ.

**2A-5 Cross-band leakage.** Operator smoke **PASS**. C_fft-only clean **PASS** (2/2). Stress **NOT_KILLED**.

| Seed | C_fft clean | oob_scale | Δ oob | full RX | Δ full | Bin |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 46.2% | 14.2% | 32.0 | 11.6% | 34.6 | NOT_KILLED |
| 1 | 46.1% | 15.4% | 30.7 | 12.2% | 33.8 | NOT_KILLED |

Mean oob_scale drop **31.4** (C' mean 28.7). View-level C_fft is STABLE; the trained classifier is not.

**Cell D** (`inband` + `ratio_rms`) clean **FAIL** (1/2). Stress closed.

| Seed | C' | D | Δ | Gate | C1 | C_fft |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 43.8% | 53.8% | +10.0 | PASS | 43.4% | 46.2% |
| 1 | 44.3% | 31.9% | −12.4 | FAIL | 24.2% | 46.1% |

No 2×2 cell is both clean-stable and scale-killed. Do not retune.

**2A-5 CLOSED.** View leak can be killed; the trained shortcut cannot, not with C_fft, not with C1, not with both under the frozen recipe. Hard scale removal is not a drop-in. Do not open 5-seed / RX2 / Day5 / utility from this grid.

### Paper 1 claim freeze (revision reserve)

**Supported.** Frozen-protocol cross-day file-level gain exists. The OOB branch is predictive. The model is highly sensitive to OOB relative magnitude scale (28.7±2.4 ≈ full RX 30.3±2.0). Full-spectrum `log_zscore` leaks that scale into Main FFT.

**Shrunk.** OOB is not a proven stable transmitter fingerprint. `no OOB` means **without explicit OOB branch**. View isolation does not give classifier robustness.

**Failed interventions (negative evidence, keep).** C1, C_fft-only, and D. Do not retune them.

1D vote closed **H4_PASS**. Per-device closed **MIXED** (file win 10 / lose 3 / tie 11; window 9/15). Source: `results/matched_seed0/per_device_day4.md`.  
Queued later: True In-Band Main; LODO.

**2B-0 clean.** S0 Gate 0 PASS; S1 Gate 1 PASS (2/2). Not a robustness claim.

| Seed | C' | S0 | Δ S0 | S1 | Δ S1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 43.8% | 46.4% | +2.7 | 43.4% | −0.4 |
| 1 | 44.3% | 43.1% | −1.2 | 43.4% | −0.9 |

**2B-0 Gate 2 STRONG / Case A.** S1 mean oob_scale drop **1.3**. S0 mean **29.6**.

**2B-0 Gate 3 TRACKS_SCALE.** S1 mean full RX drop **10.3** (9.2 / 11.5) vs C' 30.3. S0 mean **30.9**. Seed 0/1 synthetic chain done.

**2B-0 5-seed CLEAN_FAIL.** pass 2/5, mean Δ −2.72, collapse 0. Stress not read. Source: `results/matched_seed0/s1_5seed_stability.md`.

**2B-0 S0 seeds 2/3/4 SCALE_TAX.** Focus S0 2/3 PASS, mean Δ −0.93; S1 mean Δ −4.10. Case D on stronger C' seeds. RX2 / retune still closed. Source: `results/matched_seed0/s0_seeds234_diag.md`.

---

## Pointers

| Artifact | Path |
| --- | --- |
| Protocol | `PROTOCOL_LOCK.md` |
| Gates | `EXPECTED_FAILURES.md` |
| 1B | `CANDIDATES_LOCK.md` |
| 5-seed | `results/matched_seed0/audit_5seed.md` |
| Oracle | `results/matched_seed0/oracle_headroom.md` |
| Shuffle | `results/matched_seed0/oob_identity_shuffle.md` |
| RX combined | `results/matched_seed0/rx_style_eval.md` |
| RX factors | `results/matched_seed0/rx_factor_attribution.md` |
| 2A-0 probe | `results/scale_canonical_probe/scale_canonical_probe_source_day1to4.md` |
| 2A-1 C1 vs C' | `results/matched_seed0/c1_clean_vs_cprime.json` |
| 2A-2 C1 seed 0 RX | `results/matched_seed0/c1_seed0_rx_stress.json` |
| 2A-3 path leak | `results/scale_path_leak/scale_path_leak_day4.json` |
| 2A-4 view ablation | `results/inband_view_ablation/inband_view_ablation_day4.json` |
| 2A-5 C_fft smoke | `results/cfft_operator_smoke/cfft_operator_smoke.json` |
| 2A-5 C_fft clean | `results/matched_seed0/cfft_clean_vs_cprime.json` |
| 2A-5 C_fft RX | `results/matched_seed0/cfft_rx_stress.json` |
| 2A-5 D clean | `results/matched_seed0/d_clean_vs_cprime.json` |
| 2B-0 pre-reg | `PHASE2B0_SCALE_AUG_LOCK.md` |
| 2B-0 clean | `results/matched_seed0/s0_s1_clean_vs_cprime.json` |
| 2B-0 oob_scale | `results/matched_seed0/s0_s1_rx_oob_scale.json` |
| 2B-0 full RX | `results/matched_seed0/s0_s1_rx_full.json` |
| 2B-0 5-seed | `results/matched_seed0/s1_5seed_stability.md` |
| 2B-0 S0 2/3/4 | `results/matched_seed0/s0_seeds234_diag.md` |
