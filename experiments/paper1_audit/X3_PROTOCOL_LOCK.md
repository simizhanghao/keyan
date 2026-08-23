# X3 model-shortcut protocol lock

X2 has authorized X3 for both B1 and C'. No official blind receiver signal may
be opened. No F0/F0-CT, retuning, or new backbone is allowed.

For each frozen X2 checkpoint, evaluate the same source/held-out internal folds
under only these pre-registered interventions:

1. OOB-scale intervention: multiply OOB magnitude by fixed factors while
   preserving in-band IQ;
2. OOB shuffle: replace OOB with a same-file/different-device donor;
3. OOB occlusion: zero the OOB branch;
4. Left/right OOB intervention: scale left and right spectral OOB halves
   separately;
5. device probe and receiver probe on frozen embeddings.

Primary unit is receiver x seed; packet predictions are aggregated within each
unit. The goal is mechanism replication, not a new accuracy leaderboard.
