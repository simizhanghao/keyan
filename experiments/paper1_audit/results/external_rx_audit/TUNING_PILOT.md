# B1/C' equal-budget tuning pilot

Six identical source-only configurations were run for both B1 and C' on P1,
seed 0, one epoch, using `lr={3e-4,1e-3,3e-3}` and
`wd={1e-4,5e-4}`. Official blind receivers were unopened.

This pilot used only 52 source packets. The `lr=3e-3` rows nearly memorized
the tiny source subset, so the grid is a budget/procedure check, not a recipe
selection result. Held-out accuracy was not used for any decision. The
follow-up formal pilot keeps the conservative nominal recipe `lr=1e-3,
wd=5e-4` until larger source folds are used.

Machine-readable rows are under `tuning_pilot/*.json`.
