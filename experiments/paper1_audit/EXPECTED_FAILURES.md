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

| Result | Decision |
|--------|----------|
| Full > Main on ≥4/5 seeds; mean gain stable and positive | **GREEN** for OOB mechanism |
| Full ≈ Main | **YELLOW**: Paper 1 gain is RF-HSTU, not OOB. Do not sell RCOF as “OOB fingerprint” |
| Hann/guard Full collapses vs legacy 75% | **RED**: stop Paper 2 |
| matched Main stays collapsed (~C’s 8.3%) while Full is high | OOB may be a training crutch; report honestly; RCOF still possible but story changes |

## 1D File voting

Fail to claim “authentication-style robustness” if Full only wins at K=256 and loses at K≤64.

## 1E LODO

Run only after 1B–1D freeze. 3 seeds per fold, 5 seeds on the original Day5 protocol.

## Experiment 2

**Not opened in this round.** Open only after a human GO on `PAPER1_AUDIT_REPORT.md`.
