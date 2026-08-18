# Experiment 1 — Expected Failures and GO / NO-GO

Registered before training. A “good-looking” Day5 number does **not** override these gates.

## 1A Protocol lock (this round)

Fail if any of:

- train/val/test path overlap on the primary manifest
- Device9 present
- labels not covering `{0..23}`
- Day5 used for checkpoint selection in the **development** protocol
- oracle-target-val manifest used as a training recipe
- frozen `outputs/paper_ready_v3/` overwritten

## 1B Spectral audit (not this round)

| Result | Decision |
|--------|----------|
| legacy OOB has device structure, Hann+guard wipe it | **RED**: Paper 1 likely used leakage. Stop RCOF. |
| all P0–P4 keep device structure and `ρ_day < 1` | continue; pick 2 norms on **Day4 only** |
| corrected OOB has no device structure even without Hann | **RED** |

Day5 accuracy is forbidden as a selection metric here.

## 1C Matched retraining

Seed0 is **candidate evidence only**. Do not rewrite the story, move this table, or open Day5 / 1D / 1E / RCOF from one seed.

Seeds 1–4 must reuse the seed0 recipe with **zero** knob changes: same split, preprocessing, architecture, 80 ep, lr 3e-3, bs 128, Day4 checkpoint on val acc, K=256, mean-logits. Do not add Hann/guard, do not retune K/lr/epoch.

Do not collapse C (zscore) and C' (ratio) into one “Full”. Report both. Do not pick the better Full after seeing the seeds.

Required 5-seed artifacts (file **and** window):

| Quantity | Why |
|----------|-----|
| per-seed File-Acc and Window-Acc for A/B/C/C' | aggregation can inflate File-Acc |
| `Δ_file(C−B)` and `Δ_file(C'−B)` per seed | registered H3 is paired, not mean-vs-mean |
| `Δ_window(C−B)` and `Δ_window(C'−B)` per seed | seed0 zscore was +5.6pp window vs +25pp file |
| `Δ(C−A)` and `Δ(C'−A)` per seed | H3 Full>CNN; seed0 CNN vs C' is 1 file, not a finding |
| count `C>B` and `C'>B` over 5 seeds | GREEN uses this count, not a post-hoc p-value |

| Result | Decision |
|--------|----------|
| C > B on ≥4/5 seeds, mean Δ_file > 0, and window Δ is not opposite in sign | **GREEN** for Paper 1 zscore OOB on this backbone |
| C' > B on ≥4/5 seeds, same window-sign check | **GREEN** for the 1B ratio candidate |
| C>B (or C'>B) on only 2/5 or 3/5, or mean near 0 with large std | **YELLOW**: seed-dependent; do not sell OOB as a stable main effect |
| Hann/guard Full collapses vs legacy 75% | **RED**: stop Paper 2 (not in the current 1C grid) |
| matched Main stays collapsed (~8.3%) while Full is high | OOB may be a training crutch; report honestly |
| Full>Main GREEN, but Full < CNN on ≥4/5 seeds | OOB helps this backbone; **not** “OOB is the uniquely dominant fingerprint”; do not open RCOF as if H3 Full>CNN passed |

Count threshold is frozen at **4/5**. Do not move it after seeing seeds 1–4.

These rows classify the **Day4 matched 5-seed table only**. They do not schedule Day5, 1D, 1E, or RCOF. Experiment 2 stays closed until a later human GO.

## 1C.mech Day4 mechanism audit

Opened after 1C triggered the **crutch clause**. Still Day4 only. Do not rescue Main. Do not change K=256.

Order (one step at a time):

```text
1. Label-oracle headroom   ← done; not DROP
2. OOB identity shuffle    ← done; C' only; identity claim SHRUNK
3. RX-style corruption     ← done; mean window drop 30.3±2.0pp; RX-entangled
4. RX-style factor attribution ← done; Case 1A; see RX_FACTOR_ATTRIBUTION.md
```

### Oracle (label oracle)

Not the forbidden `*_oracle_target_val.csv` protocol. This uses frozen 1C predictions only.

Primary pair: `B_exact_main_no_oob` vs `C_full_ratio`. Secondary: B vs `C_full_zscore`.

Per window (and separately per file): `oracle_correct = Main correct OR Full correct`.

```text
Δ = Acc_oracle − max(Acc_Main, Acc_Full)
```

Frozen stop rule, **not moved**: if window `Δ < 5pp` on the primary pair, subsequent **utility gate is DROPPED**. This is an upper bound on any learned gate. `Δ ≥ 5pp` only means headroom exists, not that a gate will recover it.

**Independence note (do not turn into a new threshold):** Main collapsed to chance on seeds 2–4, so `max(Main, Full) ≈ Full` and `Δ` is almost automatically small. The 5pp number stays 5pp. The report must show **all 5 seeds** and the **Main-trained subset {0,1}** as diagnostics. Do not DROP solely because collapsed seeds have tiny Δ. Do not ignore the subset if it is also <5pp.

Mechanism judgement uses **window** Δ. File Δ is recorded because K=256 can inflate File-Acc.

### Shuffle (done)

Shuffle trains a new Full **C'** (`C_full_ratio_oob_shuffle`) with the frozen 1C recipe. Main IQ, label, architecture, OOB marginals, and `torch_rf_views` stay the same. Only the OOB↔device pairing is broken:

- donor = **same day, different device**
- train: donor is redrawn every epoch (no stationary OOB identity)
- eval Day4: donor is frozen per window (reproducible)
- not a fixed 24-class derangement (that would still be a 1–1 identity map)

Primary comparison: window `drop = Acc(C') − Acc(C'_shuffle)`. File-Acc is recorded, not the gate. All 5 C' seeds are used (C' did not collapse). C zscore shuffle is **not** this step.

Frozen rule, **not moved**:

| Result | Decision |
|--------|----------|
| mean window drop < 5pp | identity claim shrinks; OOB was not a stable device identity for this Full |
| mean window drop ≥ 5pp | OOB carried predictive identity under this protocol; still may be RX-entangled |

`drop ≥ 5pp` does **not** open RCOF, Day5, 1D, or a utility gate.

**Independence note (do not turn into a new threshold):** Shuffled C' collapsed to chance on seeds 2–3, so those window drops (~41.5pp) are almost `C' − chance`. The 5pp number stays 5pp. The report must show **all 5 seeds** and the **trained subset {0,1,4}**. Do not claim “OOB identity confirmed” solely because collapsed seeds have huge drop. Do not ignore the subset if its mean drop is also <5pp.

RX-style is the next diagnostic and is not skipped. Small or negative shuffle drop still leaves crutch vs shared-RX-style open.

### RX-style (this step)

Eval-only on **frozen 1C C'** checkpoints. No retraining. Day4 only.

Reuse existing operators (`apply_receiver_style`): spectral tilt, OOB scale, gain, phase, noise. **In-band scale locked at 1** so Main-band content is kept as far as those operators allow. Hann/guard stay closed.

Primary: window `drop = Acc(C') − Acc(C'_rx)`. File-Acc is recorded, not the gate. All 5 C' seeds (C' did not collapse).

Frozen rule, **not moved**:

| Result | Decision |
|--------|----------|
| mean window drop < 5pp | not strongly RX-entangled at inference |
| mean window drop ≥ 5pp | predictive but receiver-entangled OOB |

Neither row opens RCOF, Day5, 1D, a utility gate, or Hann/guard. Hann/guard open only if a later human reading says the drop is band-edge.

### Stage reading (human, after steps 1–3; not a moved 1C count)

Registered before 1C.mech-4. Do not rewrite after seeing factor drops.

| Read | Status | Why |
|------|--------|-----|
| Paper1 “stable OOB device identity” | **SHRUNK** | trained-subset shuffle drops −9.4 / +4.4 / −12.7 pp (mean −5.9) |
| Matched 5-seed H3 | **YELLOW** | Main 3/5 collapse (crutch); C' window 45.5±1.3 < CNN 49.9±2.7 |
| Oracle utility headroom | **CONDITIONAL PASS** | stable-Main seeds 8.7 / 18.7 pp; not a gate implementation |
| Combined RX-style | **STRONG** | 30.3±2.0 pp window, 5/5 |
| Paper2 motivation | **GO** | receiver/style-robust OOB modeling |
| Paper2 algorithm | **HOLD** | magnitude canonicalizer vs noise/phase vs augmentation is 1C.mech-4 |
| Utility gate | **POSTPONED** | Main endpoint itself collapses on 3/5 seeds |
| 1D / 1E / full RCOF | **CLOSED** | they do not choose the Paper2 nuisance |

Do not title Paper 2 “Receiver-Canonicalized …” until Case 1 **and** the OOB-path localization gate in `RX_FACTOR_ATTRIBUTION.md` §5.1 both fire, then a Human GO opens Phase 2A. `R_spec ≥ 18.2` alone is only a magnitude-family **candidate**.

### RX-style factor attribution (1C.mech-4, done)

Window drop vs clean C', frozen `D_full=30.3`. Complete 7×5. Day5 unused.

| Arm | mean window drop |
|-----|-----------------:|
| tilt | 5.2±1.2 |
| oob_scale | **28.7±2.4** |
| gain | 6.0±3.6 |
| phase | 0.3±0.2 |
| noise | 0.2±0.1 |
| spec | **30.5±1.8** |
| nonspec | 0.4±0.5 |

Frozen reading (thresholds not moved): **Case 1A magnitude-family + OOB-path**.  
`D_spec=30.5≥18.2`, `D_nonspec=0.4<18.2`, `max(phase,noise)<15`, `D_oob_scale=28.7≥15` → **Canonicalizer GO**. OOB-only tilt localization **not required**.

This does **not** start DCT training. Next writing: `PAPER1_AUDIT_REPORT.md`. Next experiment after a Human GO: non-learned **OOB-scale** invariance (not tilt-first DCT), no classifier training.

## 1D File voting

Fail to claim “authentication-style robustness” if Full only wins at K=256 and loses at K≤64.

## 1E LODO

Run only after 1B–1D freeze. 3 seeds per fold, 5 seeds on the original Day5 protocol.

## Experiment 2

**Not opened in this round.** Full RCOF and utility gate stay closed.

1C.mech-4 Case 1A fired. A later Human GO may open **Phase 2A Canonicalizer-only**, starting with a **non-learned OOB-scale / DC residual** invariance test (tilt-only DCT is the wrong first module: `D_tilt=5.2<15`). No utility gate. Do not rescue the frozen 1C Main table.

`PAPER1_AUDIT_REPORT.md` is written (YELLOW). Phase 2A-0 selected C1 (ρ 0.65); C2 rejected. Phase 2A-1 two-seed clean **FAIL (1/2)**: seed 0 Δ −0.3 pp PASS, seed 1 Δ −20.0 pp FAIL. C1 is not a C' replacement. File-vote / per-device / LODO remain revision reserve.
