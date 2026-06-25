# Step1.5 Statistical Report

## F vs A (cross-day Day5 test, 5 seeds)

- F File-Acc: **75.0 ± 5.3%** (n=5 seeds)
- A File-Acc: **54.2 ± 14.2%**
- Mean paired gain (F−A): **+20.8 pp**
- Seed wins: F better **4**, tie **1**, A better **0**

### Paired bootstrap CI (File-Acc difference F−A)

- seed 0: mean diff +20.8 pp, 95% CI [-4.2, +45.8] pp (n=24 files)
- seed 1: mean diff +29.2 pp, 95% CI [+4.2, +54.2] pp (n=24 files)
- seed 2: mean diff +8.3 pp, 95% CI [-16.7, +33.3] pp (n=24 files)
- seed 3: mean diff +45.8 pp, 95% CI [+20.8, +66.7] pp (n=24 files)
- seed 4: mean diff +0.0 pp, 95% CI [-20.8, +25.0] pp (n=24 files)
- pooled (descriptive): mean diff +20.8 pp, 95% CI [+9.2, +32.5] pp (n=120 file-seed pairs)

### McNemar (per seed, appendix)

- seed 0: F-only=8, A-only=3, p=0.228
- seed 1: F-only=9, A-only=2, p=0.070
- seed 2: F-only=6, A-only=4, p=0.752
- seed 3: F-only=12, A-only=1, p=0.006
- seed 4: F-only=4, A-only=4, p=0.724

## F best.pt vs last.pt (eval-only robustness check)

- seed 0: best=83.3%, last=62.5%, Δ=-20.8 pp
- seed 1: best=70.8%, last=75.0%, Δ=+4.2 pp
- seed 2: best=70.8%, last=75.0%, Δ=+4.2 pp
- seed 3: best=79.2%, last=79.2%, Δ=+0.0 pp
- seed 4: best=70.8%, last=70.8%, Δ=+0.0 pp
- mean Δ(last−best): **-2.5 pp** (std 9.4 pp)

## D/H collapse diagnostic

### D_concat_oob_plain
- seed 0: unique_preds=13, mode_frac=0.12, top=[(0, 3), (16, 3), (17, 3)]
- seed 1: unique_preds=1, mode_frac=1.00, top=[(13, 24)]
- seed 2: unique_preds=2, mode_frac=0.96, top=[(0, 23), (21, 1)]
- seed 3: unique_preds=13, mode_frac=0.21, top=[(12, 5), (21, 4), (3, 2)]
- seed 4: unique_preds=1, mode_frac=1.00, top=[(21, 24)]
### H_gated_chirp_plain
- seed 0: unique_preds=1, mode_frac=1.00, top=[(22, 24)]
- seed 1: unique_preds=6, mode_frac=0.71, top=[(11, 17), (0, 2), (1, 2)]
- seed 2: unique_preds=1, mode_frac=1.00, top=[(14, 24)]
- seed 3: unique_preds=4, mode_frac=0.58, top=[(10, 14), (12, 7), (5, 2)]
- seed 4: unique_preds=19, mode_frac=0.12, top=[(12, 3), (1, 2), (4, 2)]
