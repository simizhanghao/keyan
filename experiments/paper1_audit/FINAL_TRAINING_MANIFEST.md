# Final training manifest

Status: development epoch budgets frozen before X6; final training pending.

- Final seeds: `0,1,2,3,4` for every headline model.
- Training data: all 14 development receivers.
- Blind six receivers: never read during training or model selection.
- Fixed epoch budget: per-model median best epoch over completed development
  receiver-held-out runs, recorded here before X6.
- Final run: no validation split, no early stopping, no checkpoint selection;
  the final state after the fixed epoch budget is evaluated once.
- Optimizer/input recipe: copied from the corresponding locked development
  protocol; no post-screening tuning.

The manifest must record commit SHA, environment versions, config hash, and
TensorBoard path for every final run.

Final Shen uses the locked development batch size `64`. CIS is precomputed
once into an exact float32 cache before training; this changes only the data
loading implementation, not the representation or optimizer trajectory.

## Fixed epoch budgets

The aggregation rule is the numerical median of the 12 development
best-checkpoint epochs (P1-P6, seeds 0/1). No blind result enters this table.

| Model | Development source | Best epochs (sorted) | Final epoch |
|---|---|---|---:|
| B1-OOB | `x4b2_oob` | 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 | **5** |
| C'-OOB | `x4b2_oob` | 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5 | **4** |
| C'-TrueIB | `x4b_trueib` | 2, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5 | **5** |
| Shen-CIS | `x4c_formal` | 37, 49, 51, 63, 71, 72, 72, 75, 75, 82, 89, 90 | **74** |
| Shen-RA | `x4c_formal` | 42, 47, 54, 61, 62, 70, 72, 76, 78, 79, 89, 91 | **71** |
| B1-TrueIB (supplement) | `x4b_trueib` | 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, 5, 5 | **2** |

The five-epoch development cap censors several OOB/TrueIB best epochs at the
upper boundary. This limitation is recorded rather than repaired after model
comparison; the preregistered median rule is applied unchanged.

For the integer final epoch budget, the numerical median is rounded to the
nearest integer (`73.5 -> 74`); this conversion is fixed before any blind
receiver is opened.
