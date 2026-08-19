# Phase 2B-0 — Counterfactual OOB-scale training (pre-registered)

**Status:** LOCKED. Seed 0/1 synthetic chain **DONE**. Gate 2 STRONG (1.3). Gate 3 **TRACKS_SCALE** (10.3). **5-seed S1 CLEAN_FAIL.** S0 seeds 2/3/4 diagnostic **SCALE_TAX** (Case D on stronger C' seeds). RX2 closed.  
**Hard canonicalization (C1 / C_fft / D and any v2) is CLOSED.** Do not return to representation surgery.

Question (the only question this beat answers):

> Can the frozen Paper 1 C' architecture drop the confirmed OOB-scale shortcut if training shows the same identity under a different OOB scale — without deleting scale from the features?

Working title (internal, not a manuscript title): **Scale-Robust OOB Hybrid**.  
Not “Receiver Canonicalization”. Not a utility gate.

---

## Why this beat, and not another normalizer

2A-5 already answered the hard-deletion question:

| Fact | Number |
| --- | --- |
| OOB-scale ≈ full RX-style | 28.7±2.4 vs 30.3±2.0 |
| C_fft view STABLE, classifier worse | stress drop 31.4 |
| C1 / D clean 1/2 FAIL | seed 1 = 24.2 / 31.9; C_fft seed 1 = 46.1 |

View-level invariant ≠ classifier-level robust.  
Hard-removing scale breaks source-domain structure that C' already uses.  
The next question is **training-time counterfactual invariance**, not a stronger z-score.

A reviewer will ask: “why not just randomize OOB scale in training?”  
2B-0 is that baseline. Later modules are allowed only after this number exists.

---

## Architecture (frozen C')

```text
model        C'  (oob_norm=ratio, fft_source=full, fft_norm=log_zscore)
recipe       1C: 80 ep, bs 128, lr 3e-3, dim 64, depth 2, K=256, mean-logits
manifest     data/paper/cross_day_day1to5_source_only.csv
splits       Day1–3 train / Day4 val
Day5         CLOSED
RX2          CLOSED
```

Do **not** change features, C_fft, `ratio_rms`, network, loss extras, lr, epoch, K.

**Do not turn on `--augment-receiver-style`.**  
That path uses `lock_inband=False` and the full five-atom bundle (tilt / oob_scale / gain / phase / noise plus in-band scale). It is a different experiment.

2B-0 must call the **eval** operator: `apply_receiver_style(..., lock_inband=True)` with `rx_factor=oob_scale` only. Range stays **0.5–2.0**.

---

## Two new train arms only

C' is a frozen reference. Do not retrain it.

| Arm | Name | Pair in one step | Unique change |
| --- | --- | --- | --- |
| S0 | `C_full_ratio_paired_clean` | `(x, x)` | second forward is the **same clean** window |
| S1 | `C_full_ratio_paired_scale` | `(x, T_a(x))` | second forward is OOB-scale only |

Same unique window `x`, same label `y`.  
`a` is drawn **per window** from the frozen range, independent across the batch, same as Day4 `oob_scale` eval.

Loss (both arms):

```text
L = 0.5 * CE(f(x), y) + 0.5 * CE(f(x'), y)
```

S0: `x' = x` (two forwards; if the net is deterministic this is CE(x), but optimizer / batch-norm / compute match S1).  
S1: `x' = T_a(x)`.

Forbidden in this beat: KL, teacher, contrastive, prototype, gate, adversarial, CORAL, consistency beyond the two CEs, extra modules.

Val and checkpoint stay **clean Day4**. Do not augment val. Do not select the ckpt on stressed acc.

Epoch = one pass over the same unique Day1–3 windows as C'. Pairing does **not** add unique samples. It only doubles the forward.

---

## Gates (frozen before any S0/S1 number)

Primary metric: **window** Acc. File-Acc recorded only.

**Gate 0 — pairing sanity (S0).**  
`Δ_clean(S0 − C') ≥ −2 pp` on **both** seeds.  
FAIL ⇒ pairing / two-forward path is broken. Do **not** interpret S1. Do not retune.

**Gate 1 — clean (S1, and S0 already).**  
`Δ_clean(arm − matching-seed C') ≥ −2 pp`, 2/2, no collapse (`window < 15%`).  
C' seed 0 = 43.8, seed 1 = 44.3.

### Clean result — CLEAN_PASS (2026-08-19)

| Seed | C' | S0 | Δ S0 | Gate 0 | S1 | Δ S1 | Gate 1 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 43.8 | 46.4 | +2.7 | PASS | 43.4 | −0.4 | PASS |
| 1 | 44.3 | 43.1 | −1.2 | PASS | 43.4 | −0.9 | PASS |

Pairing is not broken (Gate 0). S1 keeps clean Day4; seed 1 does **not** collapse (unlike C1 24.2 / D 31.9).  
This is **not** scale robustness. File-Acc recorded only (S0 83.3 / 83.3; S1 75.0 / 75.0).  
Source: `results/matched_seed0/s0_s1_clean_vs_cprime.json`.

**Gate 2 — oob_scale stress (only if Gate 0 and Gate 1 both PASS).**  
Eval-only, frozen ckpt, same Day4 operator as 1C.mech-4.  
`D = Acc_clean − Acc_oob_scale` vs **that arm’s own clean**.  
Stress **both** S0 and S1. Headline is S1 mean drop; S0 is the compute-matched control.

| S1 mean `D_scale` | Reading |
| --- | --- |
| < 8 pp | **STRONG** |
| 8–15 pp | **PASS** |
| ≥ 15 pp | **FAIL** |

**2/2 seeds must be PASS or better.** One seed ≥ 15 ⇒ overall **FAIL** (no `DISAGREE` rescue).

S0 `D_scale` is expected to stay near C' (28.7 / seed-paired 25.5 and 28.5).  
If S0 also falls below 15, do not claim “scale randomization” until that is explained; the pairing itself may have regularized the shortcut.

### Gate 2 result — STRONG / Case A (2026-08-19)

| Arm | Seed | Clean | oob_scale | Δ | C' Δ | Bin |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 0 | 46.4 | 15.0 | 31.4 | 25.5 | FAIL |
| S0 | 1 | 43.1 | 15.2 | 27.9 | 28.5 | FAIL |
| S1 | 0 | 43.4 | 42.3 | **1.1** | 25.5 | STRONG |
| S1 | 1 | 43.4 | 41.8 | **1.5** | 28.5 | STRONG |

Mean S0 drop **29.6** (≈ C' 28.7). Mean S1 drop **1.3** (< 8). Both S1 seeds STRONG.

Dissociation: two-forward pairing alone does **not** kill the shortcut (S0 still collapses). Targeted OOB-scale pairing does.  
Do **not** write “any extra compute regularizes RX”. Do **not** write “real RX2 is solved”. Do **not** open factorization / a new module from Case A.

Source: `results/matched_seed0/s0_s1_rx_oob_scale.json`.  
**Gate 3 — full RX-style (Human GO 2026-08-19; eval only).**  
Same five-atom eval as C'. Recorded. Headline is S1 mean `D_full` vs own clean. S0 is the compute control.

| S1 mean `D_full` | Reading |
| --- | --- |
| < 15 pp | **TRACKS_SCALE** (full RX mostly followed the killed scale shortcut) |
| 15–20 pp | **WITHIN_IDEAL** (lock band) |
| ≥ 20 pp | **RESIDUAL** (other atoms remain; still no new module this beat) |

Not a license to retune. RX2 / 5-seed stay closed.

### Gate 3 result — TRACKS_SCALE (2026-08-19)

| Arm | Seed | Clean | full RX | Δ | C' Δ | Bin |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 0 | 46.4 | 14.7 | 31.7 | 28.5 | RESIDUAL |
| S0 | 1 | 43.1 | 12.9 | 30.1 | 29.3 | RESIDUAL |
| S1 | 0 | 43.4 | 34.2 | **9.2** | 28.5 | TRACKS_SCALE |
| S1 | 1 | 43.4 | 31.9 | **11.5** | 29.3 | TRACKS_SCALE |

Mean S0 full drop **30.9** (≈ C' 30.3). Mean S1 full drop **10.3** (< 15). Both S1 seeds TRACKS_SCALE.

Full RX mostly followed the killed scale shortcut (30.3 → 10.3). The leftover ~9 pp is other atoms, not a license to add a module. S0 still collapses, so the gain is from scale pairing, not extra forwards.

Source: `results/matched_seed0/s0_s1_rx_full.json`.

### Seed 0/1 synthetic close-out

2B-0 on Day1–4 synthetic nuisance is complete for these two seeds:

| | Clean vs C' | oob_scale drop | full RX drop |
| --- | --- | ---: | ---: |
| C' | — | 28.7 | 30.3 |
| S0 | PASS | 29.6 | 30.9 |
| S1 | PASS | **1.3** | **10.3** |

Do **not** return to C1 / C_fft / D. Do **not** open RX2 or 5-seed from this file. Factorization stays closed (Case A: keep the method small).

## Frozen method — S1 (2026-08-19)

Case A fired. The Paper 2 **source-only synthetic** method is now S1, not a canonicalizer.

```text
name       C_full_ratio_paired_scale
arch       frozen C'  (oob_norm=ratio, fft_source=full, fft_norm=log_zscore)
flag       --paired-view oob_scale
T_a        apply_receiver_style(lock_inband=True, rx_factor=oob_scale)
range      a ~ U[0.5, 2.0] per window  (same as 1C.mech-4)
loss       0.5 CE(f(x),y) + 0.5 CE(f(T_a(x)),y)
val/ckpt   clean Day4 window acc; no stressed selection
control    S0 = --paired-view clean   (same two-forward compute)
recipe     1C: 80 ep, bs 128, lr 3e-3, dim 64, K=256, mean-logits
```

Forbidden to add without a new Human GO: KL, teacher, contrastive, prototype, gate, adversarial, CORAL, `--augment-receiver-style`, C_fft, `ratio_rms`, new z-score, DCT, retune lr/epoch/range.

**Allowed (seed 0/1, Day4, synthetic only).**  
Hard scale removal is not a drop-in. View invariant ≠ classifier robust. Targeted OOB-scale pairing keeps clean Day4 and cuts the confirmed shortcut (28.7 → 1.3) and most of full RX (30.3 → 10.3). S0 proves the gain is not extra forwards.

**Forbidden.** Real RX2 is solved. 5-seed is done. Residual 10.3 pp means we must add factorization. C1/C_fft/D is the method. `no OOB` is OOB-free.

**Still needs a later Human GO (5-seed did not unlock these):** RX1↔RX2; Day5; 5-seed S1 stress reading; Oracle recompute; utility; True In-Band Main / LODO.

---

## 5-seed S1 stability (pre-registered before seeds 2/3/4)

Human GO 2026-08-19. This is **stability**, not method search.

```text
train        S1 only, seeds 2/3/4  (0/1 frozen; do not retrain)
S0           NOT retrained (compute control already shown on 0/1)
recipe       frozen S1: 80 ep, bs 128, lr 3e-3, a~U[0.5,2.0], clean Day4 ckpt
eval         Day4 clean + oob_scale + full RX  (same operators)
forbidden    RX2, Day5, retune, C_fft, utility, factorization, new loss
```

Primary metric remains **window** Acc.

**Collapse** (locked here; the old `window < 15%` check would miss C1 seed 1 at 24.2%):

```text
collapse  :=  Δ_clean(S1 − same-seed C') ≤ −15 pp
```

**Clean (hard gate, before any stress reading):**

- per-seed PASS: `Δ ≥ −2 pp` and not collapse
- 5-seed CLEAN_PASS: ≥4/5 PASS, **0/5 collapse**, mean Δ ≥ −2 pp
- CLEAN_FAIL ⇒ do not interpret scale / full RX; do not open RX2; do not retune

**oob_scale (hard gate, only if CLEAN_PASS):**

- `D = Acc_clean(S1) − Acc_oob_scale(S1)` vs own clean
- SCALE_STRONG: mean D < 8 and 5/5 D < 15
- SCALE_PASS: mean D < 15 and 5/5 D < 15
- SCALE_FAIL: any seed D ≥ 15 or mean D ≥ 15

**full RX (recorded, not a hard gate):**

- report mean±std
- TRACKS_SCALE if mean D_full < 15; else WITHIN_IDEAL / RESIDUAL
- do not retune from leftover pp

**S1_5SEED_GO** (only this unlocks a later RX2 GO): CLEAN_PASS and not SCALE_FAIL.  
A later Human GO is still required before any real RX1↔RX2. Target data must not choose the method.

Runner: `experiments/paper1_audit/scripts/run_s1_5seed.sh`  
Output: `results/matched_seed0/s1_5seed_stability.md`

### 5-seed result — CLEAN_FAIL (2026-08-19)

| seed | C' | S1 | Δ | gate |
| ---: | ---: | ---: | ---: | --- |
| 0 | 43.8 | 43.4 | −0.4 | PASS |
| 1 | 44.3 | 43.4 | −0.9 | PASS |
| 2 | 46.1 | 41.7 | −4.4 | FAIL |
| 3 | 46.5 | 42.3 | −4.2 | FAIL |
| 4 | 46.7 | 43.0 | −3.7 | FAIL |

mean Δ **−2.72** (need ≥−2). pass **2/5** (need ≥4/5). collapse **0**.  
oob_scale / full RX **not read**. RX2 stays closed. Gates not moved.

S1 itself is stable (41.7–43.4). The misses are vs stronger C' seeds 2–4 (46.1–46.7), not a C1-style −20 pp crash.  
S0 seeds 2/3/4 now exist: reading **SCALE_TAX** (below). Case D is the 5-seed clean branch.

Do not retune range/lr/epoch. Do not open RX2. Do not return to C1/C_fft/D.

### S0 seeds 2/3/4 diagnostic (pre-registered)

Human GO after CLEAN_FAIL. **Diagnosis only.** Does not reopen S1, move −2 pp, or open RX2.

```text
train        S0 only, seeds 2/3/4   (--paired-view clean)
S1           frozen; do not retrain
eval         clean Day4 only
forbidden    stress, RX2, Day5, retune, overwrite s0_s1_clean_vs_cprime.json
```

Same Δ ≥ −2 / collapse ≤ −15 vs same-seed C'. Reading uses **seeds 2/3/4 only**:

| Reading | If |
| --- | --- |
| PAIRING_TAX | S0 FAIL on ≥2/3 of {2,3,4} |
| SCALE_TAX | S0 PASS on ≥2/3 of {2,3,4} (S1 already FAIL there) |
| MIXED | 1/3 |

PAIRING_TAX ⇒ the 5-seed clean miss is two-forward / pairing, not uniquely scale invariance.  
SCALE_TAX ⇒ Case D on the stronger C' seeds: forcing scale invariance costs ~4 pp identity.  
Neither reading opens RX2 or a retune.

Runner: `experiments/paper1_audit/scripts/run_s0_seeds234.sh`  
Output: `results/matched_seed0/s0_seeds234_diag.md`

### S0 seeds 2/3/4 result — SCALE_TAX (2026-08-19)

| seed | C' | S0 | S1 | Δ S0 | Δ S1 | S0 gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 43.8 | 46.4 | 43.4 | +2.6 | −0.4 | PASS |
| 1 | 44.3 | 43.1 | 43.4 | −1.2 | −0.9 | PASS |
| 2 | 46.1 | 46.1 | 41.7 | +0.0 | −4.4 | PASS |
| 3 | 46.5 | 47.2 | 42.3 | +0.7 | −4.2 | PASS |
| 4 | 46.7 | 43.2 | 43.0 | −3.5 | −3.7 | FAIL |

Focus 2/3/4: S0 mean Δ **−0.93** (**2/3 PASS**); S1 mean Δ **−4.10**.  
Pre-registered reading **SCALE_TAX**. S1 5-seed **CLEAN_FAIL** is not moved.

Case D: on the stronger C' seeds, forcing OOB-scale invariance costs ~4 pp identity. The miss is not two-forward / pairing (S0 holds 2/3). Seed 4 S0 also misses (−3.5); that does not flip the ≥2/3 rule.

Neither this table nor SCALE_TAX opens RX2, a retune, S1 retrain, or 5-seed stress reading.

---

## Decision tree (one way; no return to 2A)

```text
Gate 0 FAIL                 → implementation bug; fix pairing only
S1 clean FAIL, S0 clean PASS → Case D: forcing scale invariance hurts identity
S1 clean FAIL, S0 clean FAIL → pairing broken or 2×-forward broken
S1 D_scale ≥ 15, clean PASS → Case C: aug-only not enough; later module GO
S1 D_scale 8–15             → Case B: aug weakens shortcut; later factorization GO
S1 D_scale < 8              → Case A: aug-only works; keep method small
```

Seed 0/1 synthetic gates remain Case A (D_scale 1.3 / D_full 10.3).  
5-seed clean is Case D (SCALE_TAX). `S1_5SEED_GO` did not fire. Real RX1↔RX2 stays closed.

Closed until a later Human GO: Day5, RX2, 5-seed S1 stress reading, retune, utility, RCOF, C2, DCT K, Hann/guard, True In-Band Main, LODO.  
**Next beat:** Phase 2B-2 real RX manifest audit (`PHASE2B2_REAL_RX_LOCK.md`). F0 5-seed is **F0_5SEED_GO**. Do not reopen S1. Do not open F1.

---

## Paper 1 in this phase

Mechanism audit is closed. Paper 1 work is **revision reserve**, no-train first:

1. File-vote K ∈ {8,16,32,64,128,256} × {mean_logits, mean_prob, majority} — done; H4_PASS.
2. Per-device Day4 table (CNN / Main / C') — done; MIXED (10/3/11).
3. True In-Band Main — queued, needs train, not this beat.
4. LODO — after 2B-0, not now.

---

## Implementation (smoke PASS; S0/S1 clean train opened)

- `--paired-view {off,clean,oob_scale}` in `add_common_args`. Default `off` keeps C' identical.
- `paired_second_view()` uses the eval operator: `lock_inband=True`, `rx_factor=oob_scale` on the second view only, then restores `rx_factor`.
- `run_epoch` trains with `0.5 CE(x)+0.5 CE(x')`. Val (`train=False`) stays a single clean forward.
- Combined with `--augment-receiver-style` / extra losses is rejected.
- Smoke: `experiments/paper1_audit/scripts/smoke_paired_view.py` → **SMOKE_PASS**.
  S0 rel-L2 0; in-band 1.02e-7; OOB 0.356; IQ 0.036. Source: `results/paired_view_smoke/paired_view_smoke.json`.
- Clean train runner: `experiments/paper1_audit/scripts/run_s0_s1_paired.sh` (S0 first; Gate 0 FAIL skips S1).

Gate 2/3 are done. Do not re-run them unless code changes.
