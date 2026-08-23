# B1/C' equal-budget pilot

**Verdict: training pipeline PASS; no paper metric.**

Both controls used fixed P1 (`rtl_2`), seed 0, the same 52 source packets and 4
held-out packets, one epoch, batch size 8, `AdamW(lr=1e-3, wd=5e-4)`, and
source-only checkpoint logic. Official blind receivers were unopened.

| Model | Params | Loss first -> last | Source | Held-out |
|---|---:|---:|---:|---:|
| B1 | 539,178 | 2.3082 -> 1.1991 | 1.0 | 1.0 |
| C' | 625,896 | 2.9536 -> 0.1010 | 1.0 | 1.0 |

The held-out values use only four packets and are explicitly a pipeline check,
not evidence of generalization or a model ranking.

Machine-readable result: `equal_budget_pilot.json`.
