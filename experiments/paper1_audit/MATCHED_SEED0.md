# 1C seed0 — matched retraining (Day4 only)

Do not evaluate Day5 until this seed0 table is interpreted.

| ID | Model | Unique change |
|----|-------|----------------|
| A | CNN-IQ | baseline |
| B | Exact Main-only | CNN-stem + RF-HSTU + chirp, `--no-oob` |
| C | Full zscore | Paper 1 OOB control |
| C' | Full ratio | 1B spectral winner |

Shared: Day1–3 train, Day4 val checkpoint, 80 ep, **bs 128 / lr 3e-3** (Paper 1 matched), dim 64, seed 0, mean-logits, K=256.

Speed: 2-GPU waves (`GPUS=4,5`) + `--num-workers 8`. Do **not** inflate batch to fill 80GB — the hybrid is ~1.16M params; bs=128 already uses ~2GB, and a 4k+ batch would change optimizer dynamics and break the audit.

Frozen 75% / 66.7% / 8.3% numbers stay in `outputs/paper_ready_v3/`. This run writes only under `results/matched_seed0/`.

## Day4 val numbers (seed 0, n=24 files, Day5 unused)

Source: `results/matched_seed0/summary_val.json`. File-Acc step size is 1/24 ≈ 4.2 pp.

| ID | file_acc | window_acc | file_macro_f1 | best epoch |
|----|---------:|-----------:|--------------:|-----------:|
| A CNN-IQ | 75.0% (18/24) | 52.2% | 67.4% | 57 |
| B Main `--no-oob` | 41.7% (10/24) | 22.2% | 31.2% | 78 |
| C Full zscore | 66.7% (16/24) | 27.9% | 57.6% | 74 |
| C' Full ratio | 70.8% (17/24) | 43.8% | 66.0% | 77 |

`eval_val/*/run_config.json` dumps `evaluate.py` argparse defaults (`epochs=10`, `lr=0.001`). Those are **not** the training recipe. Training was 80 ep / 3e-3 / bs 128, confirmed by `summary_val.json` and checkpoint epochs above.

## Seed0 gate (not the 5-seed 1C decision)

Registered in `EXPECTED_FAILURES.md`: GREEN needs Full > Main on ≥4/5 seeds. This file is seed0 only.

| Registered test | Seed0 observation | Fires? |
|-----------------|-------------------|--------|
| GREEN: Full > Main on ≥4/5 seeds | Full > Main on **1/1** seed | **no** |
| YELLOW: Full ≈ Main | C 66.7 / C' 70.8 vs B 41.7 | **no** |
| RED: Hann/guard Full collapse | Hann/guard not in this grid | **no** |
| Main collapsed ~8.3% | B = 41.7%, not collapsed | **no** |
| H3 Full > CNN | CNN 75.0 > C' 70.8 > C 66.7 | **fails on seed0** |

**Status: HOLD — continue 5 seeds on Day4. Not GREEN. Not YELLOW. Not RED. Experiment 2 closed.**

Do not treat these Day4 seed0 numbers as Paper 1 Day5 5-seed means. Do not open Day5, 1D, 1E, or RCOF from this table.

## Locked next step — Day4 seeds 1–4 (no new experiment)

Question this step is allowed to answer:

```text
Is seed0 Full > Main a stable paired effect, or one-seed noise?
```

Recipe freeze (copy seed0; change only `--seed`):

```text
manifest     data/paper/cross_day_day1to5_source_only.csv
split        Day1–3 train / Day4 val / Day5 unused
models       A_cnn_iq, B_exact_main_no_oob, C_full_zscore, C_full_ratio
epochs / lr / bs / dim   80 / 3e-3 / 128 / 64
checkpoint   Day4 val acc
vote / K     mean_logits / 256
oob_norm     none | zscore | ratio   (per model, unchanged)
```

Forbidden while this step runs: Day5 eval, 1D, 1E, RCOF, Hann/guard, new K, new lr/epoch, picking a “winning” Full after peeking.

After seeds 1–4 exist, run `scripts/audit_matched_5seed.py`, then **stop**. Do not treat a 4/5 count as permission to open Day5 or RCOF. Seed0 CNN 75.0 vs C' 70.8 is one file and does not decide H3.
