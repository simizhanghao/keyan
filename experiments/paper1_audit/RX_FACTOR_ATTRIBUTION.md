# 1C.mech-4 — RX-style factor attribution

**Locked before any factor eval.** Do not move these rules after seeing drops.  
**Human GO:** implement `--rx-factor` + seed0 tilt smoke only. 7×5 is a later GO.  
**Question this step may answer:**

```text
Which nuisance family drives the frozen combined RX-style
window collapse of 30.3±2.0 pp on C'?
```

It may **not** train a canonicalizer, open RCOF, open Day5 / 1D / 1E / Hann/guard, rescue Main, change K, or pick a Paper 2 title.

---

## 1. Freeze (copy 1C / combined RX)

```text
checkpoints   experiments/paper1_audit/results/matched_seed0/runs/C_full_ratio/seed_{0..4}/best.pt
recipe        Day1–3 train / Day4 val / Day5 unused
model         C' Full ratio, OOB cross-attn, dim 64
eval          K=256, mean-logits, --eval-split val, --rx-style-eval, lock_inband=1
operators     apply_receiver_style  (existing ranges; do not retune)
in-band scale locked at 1
```

Reuse, do **not** re-run:

| Arm | Source |
|-----|--------|
| R0 clean C' | `eval_val/C_full_ratio/seed_*` |
| R6 full RX | `eval_val/C_full_ratio_rx_style/seed_*` → window drop **30.3±2.0 pp** |

Re-running R6 with a new RNG would create a second “30.3” and invite post-hoc replacement. The frozen combined number stays 30.3±2.0.

---

## 2. What the current operator actually is

`apply_receiver_style` is **not** five independent physical knobs. Read this before attributing.

| Name in 30.3 writeup | Code | Identity (off) | On range (frozen) |
|----------------------|------|----------------|-------------------|
| tilt | `10**((tilt_db * f/fmax)/20)` on **all** bins | `tilt_db=0` | ±3 dB |
| OOB scale | multiply OOB bins | `oob_scale=1` | 0.5–2.0 |
| in-band scale | multiply in-band bins | already 1 | **locked**; not a factor |
| gain | **global** `10**(gain_db/20)` on the whole spectrum | `gain_db=0` | ±6 dB |
| phase | global `e^{jφ}` after IFFT | `φ=0` | currently **hardcoded** `U(-π,π)` |
| noise | additive IQ noise | `noise_std=0` | 0–0.01 |

Implementation may add `--rx-factor` (or equivalent) that turns unused operators to identity. **On-ranges must stay exactly the table above.** Adding a phase-off hook is required; today phase cannot be disabled by argparse.

### Confounds (registered; do not “discover” them after the table)

1. **OOB is magnitude-only.** `torch_rf_views` / `oob_norm=ratio` uses `|X(f)| / inband_RMS`. A global phase rotation is invisible to the OOB branch. A large phase-only drop is **Main IQ-path** sensitivity, not “OOB phase fingerprint”.
2. **Gain is likely a near-null.** `input_norm=iq_rms` plus ratio-OOB both cancel a global scale. `D_gain ≈ 0` does **not** mean “receiver gain is not a real-world nuisance”; it means this frontend already removed it. Do not use a null gain arm to argue Case 4.
3. **Tilt still tilts in-band.** `lock_inband` only freezes the flat in-band scale. At 1 MHz fs, ±3 dB tilt to Nyquist is ~0.4 dB at the LoRa band edge and the full 3 dB in far OOB. Tilt is OOB-weighted, not OOB-exclusive.
4. **OOB-scale is the cleanest OOB-only arm.** It changes `|OOB|` and therefore the ratio feature by construction (0.5×–2×).
5. **D_j are independently sampled.** This step does **not** reuse the per-window random draws from R6. Therefore `D_full ≠ Σ D_j` for two reasons: nonlinearity **and** different RNG. Do not treat non-additivity alone as Case 4.

---

## 3. Arms to run (eval-only)

Leave-one-in, plus two superclasses the Case-1 rule needs. Five singles alone **cannot** test “combined magnitude style reproduces most of the 30.3 pp collapse”.

| ID | Enable | Disable others to identity |
|----|--------|----------------------------|
| R1 | tilt | scale=1, gain=0, phase=0, noise=0 |
| R2 | OOB scale | tilt=0, gain=0, phase=0, noise=0 |
| R3 | gain | tilt=0, scale=1, phase=0, noise=0 |
| R4 | phase | tilt=0, scale=1, gain=0, noise=0 |
| R5 | noise | tilt=0, scale=1, gain=0, phase=0 |
| R_spec | tilt + OOB scale + gain | phase=0, noise=0 |
| R_nonspec | phase + noise | tilt=0, scale=1, gain=0 |

5 seeds × 7 new evals. No training. R0/R6 already exist.

Output dirs (English, no `phase*` experiment tree):

```text
eval_val/C_full_ratio_rx_tilt/
eval_val/C_full_ratio_rx_oob_scale/
eval_val/C_full_ratio_rx_gain/
eval_val/C_full_ratio_rx_phase/
eval_val/C_full_ratio_rx_noise/
eval_val/C_full_ratio_rx_spec/
eval_val/C_full_ratio_rx_nonspec/
rx_factor_attribution.json
rx_factor_attribution.md
```

---

## 4. Metrics

Primary (decides cases):

```text
D_j = Acc_window(R0) − Acc_window(R_j)     # percentage points
```

Report per seed and 5-seed mean±std. Compare to frozen `D_full = 30.3±2.0`.

Secondary (recorded, not case-moving):

| Metric | How |
|--------|-----|
| File-Acc drop | same formula, K=256 |
| Prediction flip rate | fraction of windows whose `pred` differs from R0, paired on `(file_path, window_index)` |
| Confidence change | mean `confidence` (already in `predictions.csv`) R_j − R0 |

Required-if-cheap (two extra scalars per window, not a full logit dump):

| Metric | How |
|--------|-----|
| True-class logit drop | mean `logit[y]` R0 − R_j |
| Entropy change | mean `H(softmax)` R_j − R0 |

If adding those two columns is not ready in the same implementation step, ship the table with window/file/flip/confidence and leave logit/entropy as an explicit hole. **Do not delay the 7-arm window table for prettier internals.** Do not decide cases from File-Acc.

---

## 5. Frozen case rules (not moved after seeing data)

Let `D_full = 30.3`.  
A factor is **individually large** if mean window `D_j ≥ 15 pp`.  
A superclass is **majority** if its combined-arm drop `≥ 0.60 × D_full` (18.2 pp).

Read **R_spec / R_nonspec**, not a sum of singles, when the rule says “combined magnitude style”.

| Case | If | Then (algorithm HOLD until Human GO) |
|------|----|--------------------------------------|
| **1 Magnitude candidate** | `D_spec ≥ 18.2` **and** `D_nonspec < 18.2` **and** `max(D_phase, D_noise) < 15` | **Not Canonicalizer GO.** See §5.1 path-localization |
| **2 Noise** | `D_nonspec ≥ 18.2` **and** `D_spec < 12.1` **and** `D_noise ≥ 15` | **Do not** start a tilt canonicalizer. First module = SNR / uncertainty-aware OOB reliability |
| **3 Phase** | `D_nonspec ≥ 18.2` **and** `D_spec < 12.1` **and** `D_phase ≥ 15` **and** `D_phase` is the unique max among singles | Main/hybrid path is phase-unstable. **Not** an OOB-phase paper. Optional later Main-only control on seeds 0/1 |
| **4 Compound** | `max(D_1…D_5) < 15` **and** neither family ≥ 18.2 | Prefer receiver-style augmentation; no single-factor canonicalizer |
| **Mixed** | `D_spec ≥ 18.2` **and** `D_nonspec ≥ 18.2` | **Algorithm HOLD.** Do not pick spectral because it looks nicer |
| **Gain-null** | `D_gain` near 0 | Expected. Ignore for Case 4. Do not call gain “not a receiver effect” |

`D_j ≥ 15` does **not** open Phase 2A. The 7 arms classify the nuisance family only.

### 5.1 Publication gate: OOB-path localization (after Case 1, not this smoke)

`R_spec` large is **magnitude-family candidate**, not proof that OOB is the injured path. Tilt still tilts in-band FFT/AP/CNN-stem.

| After Case 1 | If | Then |
|--------------|----|------|
| A | `D_oob_scale ≥ 15` | **Canonicalizer GO** (OOB-only magnitude already breaks C') |
| B | `D_spec ≥ 18.2` and `D_oob_scale < 15` | Run **OOB-only tilt** frozen inference: Main IQ / AP / in-band FFT unchanged; only the spectrum fed to the OOB branch is tilted. Drop large → OOB spectral canonicalizer GO. Drop small → R_spec was whole-spectrum / Main-path; **do not** spend weeks on an OOB DCT |

OOB-only tilt is **not implemented in this step**. It opens only after the 7×5 table and a later Human GO.

Collapsed-Main independence note does **not** apply here: all five C' seeds trained. Show all 5.

---

## 6. Still forbidden

- Day5 eval, 1D K-sweep, 1E LODO
- Hann / guard
- Retraining, new checkpoints, changing 1C numbers
- Utility gate, feature gate, RCOF
- Saving Main, changing lr / epoch / K
- Sweeping RX ranges or inventing new operators
- Writing `PAPER1_AUDIT_REPORT.md` before `rx_factor_attribution.md` exists
- Starting DCT/spline code in the same step as the eval

---

## 7. After the table exists

Stop. Report the 7-arm window drops against 30.3±2.0 and name the frozen case.  
Then, and only then, a human may ask for `PAPER1_AUDIT_REPORT.md`.

Suggested report skeleton (fill attribution after the table; do not hard-PASS Paper 1):

```text
1. Protocol integrity              PASS
2. Matched 5-seed stability        YELLOW
3. OOB identity claim              SHRUNK
4. Oracle complementarity          CONDITIONAL PASS
5. Receiver/style sensitivity      STRONG  30.3±2.0 pp
6. Mechanism attribution           [Case 1/2/3/4/Mixed]
Overall: Paper1 remains an empirical architecture result;
         “stable OOB device fingerprint” is unsupported.
Paper2: motivation GO; algorithm follows Case *.
```

---

## 8. Not this step (later, if Case 1 + Human GO)

Phase 2A would be a **non-learned** log-mag residual `L(f)−b(f)` with small K ∈ {4,8}, invariance vs device-separability on Day4 pairs `(x, x')`, **then** seed 0/1 `C'` vs `C_can` only. Utility gate stays behind a later Stable-Main endpoint. Do not implement any of that here.
