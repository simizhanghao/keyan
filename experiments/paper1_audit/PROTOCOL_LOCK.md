# PROTOCOL_LOCK — Paper 1 Audit

**Locked:** 2026-08-18  
**Repo:** `keyan` (`/data1/hcc/llm4RF/new_phase`, origin `simizhanghao/keyan`)  
**Frozen results (read-only):** `/data1/hcc/llm4RF/outputs/paper_ready_v3/`  
**This file is the Experiment 1 protocol.** Older `docs/experiment_protocol.md` is historical and is **not** the development protocol.

Do not train, do not edit `features.py`, and do not open Paper 2 until 1A is accepted and 1B–1C complete.

---

## 1. Primary development protocol (only protocol allowed until freeze)

```text
Train:        Day1 + Day2 + Day3
Validation:   Day4          ← method / preprocess / early-stopping only
Final test:   Day5          ← sealed until 1C/1D freeze, then opened once
Classes:      24
Excluded:     raw Device9
Window:       8192
Eval K:       256 windows / file (1D later sweeps K)
File vote:    mean logits
Manifest:     data/paper/cross_day_day1to5_source_only.csv
```

Generator already matches this lock (`generate_paper_manifests.py`: `day_manifest([1,2,3],[4],[5], ...)`), even though a comment in that file still says “Day1-4 -> Day5”.

### CSV static counts (verified)

| Split | Days | Files | Devices | Device9 |
|-------|------|------:|--------:|---------|
| train | 1,2,3 | 72 | 24 | absent |
| val | 4 | 24 | 24 | absent |
| test | 5 | 24 | 24 | absent |

Remap: raw Device1–8 → experiment device 1–8 / label 0–7; raw Device10–25 → device 9–24 / label 8–23.

One capture per device per day: `IQ_1.dat` only.

---

## 2. Hard Day5 rule

Until the method is frozen after 1B + seed0 1C:

**Day5 must not be used for**

- model design
- OOB preprocessing choice
- Hann / guard-band choice
- loss weights
- early-stopping rule changes
- gate / fusion design (Paper 2 is closed anyway)

Day5 appears in the primary manifest as `split=test` only so that the **final** eval job can read it later. Development jobs must pass `--eval-split val` (Day4) or omit test metrics from selection.

LODO (`data/paper/lodo_source_only/`) is **sealed until 1E**. Several LODO folds use Day5 as **val** (`val_day = max(remaining)`). That is incompatible with this lock if run during development.

---

## 3. Forbidden manifests for Experiment 1 development

| File | Why forbidden now |
|------|-------------------|
| `data/paper/cross_day_day1to5_oracle_target_val.csv` | val = Day5 = test; target-label leakage |
| `data/paper/rx*_oracle_target_val.csv` | Paper 2 / diagnosis only |
| `shiyaner/` RAOF | not in frozen baseline |

---

## 4. Document conflicts (do not “fix” by mixing protocols)

| Source | What it says | Status |
|--------|----------------|--------|
| IoTJ §Experiments + `PAPER_RESULTS_SUMMARY.md` | Day1–3 / Day4 / Day5 | **canonical** |
| `data/paper/cross_day_day1to5_source_only.csv` | same | **canonical** |
| `docs/experiment_protocol.md` §2 | Day1–4 train → Day5 test | **legacy, superseded** |
| `generate_paper_manifests.py` comment | “Cross-day: Day1-4 -> Day5” | stale comment; code is Day1–3/4/5 |
| LODO generator | remaining days; val = max remaining (often Day5) | allowed only in 1E after freeze |

---

## 5. Frozen Paper 1 training recipe (legacy, for matched retrain)

From `scripts/paper/lib/v3_job_defs.py` and `outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md`.

| Item | Value |
|------|-------|
| Manifest | `data/paper/cross_day_day1to5_source_only.csv` |
| Epochs / lr / batch | 80 / 3e-3 / 128 |
| dim / depth | 64 / 2 |
| input_norm / fft_norm | `iq_rms` / `log_zscore` |
| OOB norm (legacy Full) | **`zscore`** |
| loss / LS / wd | CE / 0 / 5e-4 |
| checkpoint | source val **acc** |
| vote | `mean_logits` |
| Main model | `F_cross_attn_chirp_plain` |
| CNN | `A_cnn_iq` |

### Matched Main-only (1C) — lock this definition now

Paper Table I “RF-HSTU no OOB” is **`B_linear_no_oob`**: linear patch embed, 3 seeds, **not** CNN-stem. That is **not** the 1C Main expert.

1C Main-only must match Full except OOB:

```text
A. CNN-IQ              A_cnn_iq
B. Exact Main-only     CNN-stem + RF-HSTU + chirp, --no-oob
C. Full-OOB            F_cross_attn_chirp_plain  (OOB cross-attn on)
```

Identical: stem, HSTU, dim, depth, epoch, optimizer, classifier, manifest, seeds, checkpoint rule.

Known risk: frozen `C_cnn_stem_chirp_no_oob` seed0 File-Acc **8.3%** (collapsed). 1C must re-run this matched Main; do not treat 66.7% as the OOB-off baseline.

---

## 6. Legacy OOB definition (do not change until 1B)

`src/rfhstu/features.py` `oob_norm="zscore"`:

```text
oob_view = (mag * oob_mask - fft_mean) / fft_std
```

In-band bins become `-μ/σ`, not 0. FFT has no Hann in this function. Guard band = 0 (OOB starts at ±62.5 kHz).

1B will compare P0–P4, Hann vs rectangular, guard ∈ {0, 12.5, 25} kHz **without training**, select **two** candidates on Day4, then 1C trains.

---

## 7. Frozen result pointers (read-only)

| Artifact | Path |
|----------|------|
| Table I | `/data1/hcc/llm4RF/outputs/paper_ready_v3/final_tables/table1_cross_day_main.csv` |
| Step1 report | `.../step1_phase7_clean/STEP1_REPORT_FOR_GPT.md` |
| Job defs | `scripts/paper/lib/v3_job_defs.py` |

Do not rewrite these numbers. Audit writes only under `experiments/paper1_audit/`.

---

## 8. Experiment 1 order (do not skip)

```text
1A  this file + machine audit          ← done
1B  spectral audit, no training        ← done (GO_TWO_CANDIDATES)
1C  CNN / Main / Full, 5 seeds Day4    ← done; YELLOW / crutch; not GREEN
1C.mech  Day4 mechanism audit (no Day5)
    oracle headroom                    ← done; utility gate not DROP
    OOB identity shuffle               ← done; identity claim SHRUNK
    RX-style corruption                ← done; RX-entangled 30.3±2.0pp
    RX-style factor attribution        ← done; Case 1A; D_oob_scale 28.7; D_spec 30.5
1D  file-vote K sweep                  ← protocol K stays 256; revision reserve opened below
1E  LODO                               ← closed this audit
    → PAPER1_AUDIT_REPORT.md           ← written; YELLOW
    → Phase 2A-0 scale probe           ← done; C1 GO (ρ 0.65); C2 no
    → Phase 2A-1 C1 seed 0/1 train     ← done; two-seed clean FAIL (1/2)
    → Phase 2A-2 C1 seed 0 RX stress   ← done; NOT_TRANSFERRED (23.2 pp)
    → Phase 2A-3 scale path leak       ← done; views: amp_phase + cnn_stem LIVE
    → Phase 2A-4 in-band view ablation ← done; SMALLEST_KILL=C_fft
    → Phase 2A-5 C_fft intervention    ← CLOSED; hard canonicalization CLOSED
    → Phase 2B-0 paired OOB-scale train ← S1 FROZEN; 5-seed CLEAN_FAIL; S0 SCALE_TAX; RX2 closed
    → Phase 2B-1 identity-anchored      ← F0_5SEED_GO; F1 closed
    → Phase 2B-2 real RX source-only    ← RX_FAIL; OSU 2-RX method STOPPED
    → Phase 2C X0 external audit       ← OOB_OK; official 14/6 matched; blind six sealed
    → Phase X1 source-only signal audit ← done; receiver > day drift, device > receiver; no classifier training
    → Phase X1.5 publication audit    ← done; clustered cells + DUT/SNR/CFO/day checks
    → Phase X2 protocol design         ← locked; B0/B1 PASS, C' smoke PASS, equal-budget pilot next; blind six remain sealed
    → Experiment 2 / RCOF / utility gate closed
    1D file-vote K sensitivity         ← done; H4_PASS (not a K=256 spike)
    1D per-device Day4                 ← done; MIXED (10 win / 3 lose / 11 tie)
```

Human stage reading after 1C + oracle + shuffle + combined RX (not a moved numeric gate):

```text
Paper1 identity claim     SHRUNK
Paper1 matched 5-seed     YELLOW (Main crutch; C' window < CNN)
Oracle complementarity    CONDITIONAL PASS (stable-Main seeds)
RX-style sensitivity      STRONG (30.3±2.0pp)
Paper2 motivation         GO (OOB-scale shortcut is real)
Paper2 algorithm          F0 synthetic GO; real OSU 2-RX RX_FAIL; pivot 2C external audit
Working title             Identity-Anchored Counterfactual OOB (internal; not a paper title)
```

Do not rescue Main, retune lr/epoch, reopen C1/C_fft/D, open Day5, Hann/guard, RCOF, or a utility gate. C1 is not a C' replacement. Current `no OOB` means no explicit OOB branch. S1 is frozen (`CLEAN_FAIL` / `SCALE_TAX`). F0 5-seed is **F0_5SEED_GO** (clean +3.14, D_scale 1.5, D_full 6.9). F1 closed. Phase 2C X0 is **OOB_OK** with the official 14-source/6-blind mapping verified from archive metadata. X1 signal-level GO and X1.5 publication audit are complete; the device > receiver > day ordering survives cell aggregation, per-DUT checks, and SNR/CFO sensitivity. Paper 2 method remains HOLD. X2 is protocol-locked in `X2_PROTOCOL_LOCK.md`; B0/B1 implementation and nominal pilot pass, audited C' runtime smoke passes, and the equal-budget B1/C' pilot is next; F0/F0-CT stay closed. The six official blind HDF5 signals stay sealed until X6. Do not train F0, retune F0, or open F1. Paper 1 True In-Band Main / LODO stay queued.

---

## 9. Machine audit

Run `experiments/paper1_audit/scripts/audit_protocol.py` (验收命令 in the chat). It writes:

- `results/protocol_audit.json`
- `results/manifest_hashes.json`
- `results/protocol_audit.md`

IQ `.dat` files are **not** inside `keyan`. Check against `--data-root` (default `/data1/hcc/llm4RF`).
