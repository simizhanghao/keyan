# X3-C source-derived neutral OOB replacement

The OOB bins were replaced by a source-derived neutral magnitude vector (256
packets per source receiver, held-out receiver excluded), with zero phase. IQ,
labels, and frozen checkpoints were unchanged.

| held-out receiver | B1 drop (pp) | C' drop (pp) |
|---|---:|---:|
| rtl_2 | 43.88 | 16.52 |
| rtl_5 | 54.28 | 40.44 |
| b200_1 | 29.96 | 60.91 |
| b200_mini_1 | 44.79 | 64.74 |
| b210_1 | 44.01 | 52.62 |
| pluto_1 | 38.64 | 75.76 |
| **mean / median** | **42.59 / 43.94** | **51.83 / 56.77** |

Both backbones pass Gate B at 6/6 folds. Official blind receivers remain sealed.
