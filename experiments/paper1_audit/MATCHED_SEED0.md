# 1C seed0 — matched retraining (Day4 only)

Do not evaluate Day5 until this seed0 table is interpreted.

| ID | Model | Unique change |
|----|-------|----------------|
| A | CNN-IQ | baseline |
| B | Exact Main-only | CNN-stem + RF-HSTU + chirp, `--no-oob` |
| C | Full zscore | Paper 1 OOB control |
| C' | Full ratio | 1B spectral winner |

Shared: Day1–3 train, Day4 val checkpoint, 80 ep, bs 128, lr 3e-3, dim 64, seed 0, mean-logits, K=256.

Frozen 75% / 66.7% / 8.3% numbers stay in `outputs/paper_ready_v3/`. This run writes only under `results/matched_seed0/`.
