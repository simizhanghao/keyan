# F0 formal summary (20-SDR primary dataset)

All six internal held-out receiver folds, two seeds, and both backbones were
completed. CT is clean continuation training; F0 adds only paired log-uniform
OOB scale augmentation. Held-out values use receiver x seed units.

| model | Base mean | CT mean | F0 mean | F0 - Base | F0 - CT |
|---|---:|---:|---:|---:|---:|
| B1 | 55.09% | 64.85% | 61.69% | +6.60 pp | -3.16 pp |
| C' | 61.46% | 63.66% | 60.02% | -1.44 pp | -3.65 pp |

CT improves through extra optimization alone. F0 does not beat the matched CT
control for either backbone, so this phase is **F0 HOLD/NO-GO**. The result
does not weaken the X3 mechanism finding; this mitigation recipe is not
sufficient. Do not retune or open official blind receivers.
