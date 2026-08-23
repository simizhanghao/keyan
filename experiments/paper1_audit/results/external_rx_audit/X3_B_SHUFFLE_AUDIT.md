# X3-B same-receiver cross-device OOB shuffle

Frozen X2 checkpoints were evaluated after replacing only the OOB spectrum
with a donor from a different device in the same receiver file. The in-band IQ,
label, and checkpoint were unchanged. Values are held-out receiver means over
two seeds.

| held-out receiver | B1 drop (pp) | C' drop (pp) |
|---|---:|---:|
| rtl_2 | 45.89 | 17.59 |
| rtl_5 | 57.21 | 41.74 |
| b200_1 | 33.38 | 64.99 |
| b200_mini_1 | 50.04 | 71.23 |
| b210_1 | 48.89 | 56.81 |
| pluto_1 | 40.28 | 81.91 |
| **mean / median** | **45.95 / 47.39** | **55.71 / 60.90** |

Both backbones pass the locked 5 pp / 4-of-6 criterion at 6/6 folds. This is
mechanism evidence, not an authentication claim. Blind receivers remain sealed.
