# Phase 2A-0 scale–shape probe

files=8  windows=16  days=[1, 2, 3, 4]  smoke=True
Day5 unused. Real RX2 unused. No training.

## Scale invariance (relative L2; cosine is scale-blind, recorded only)

| Rep | rel-L2 | cosine |
| --- | -----: | -----: |
| C0 | 0.3308 | 0.0000 |
| C1 | 0.0000 | -0.0000 |
| C2 | 0.0002 | 0.0000 |

## Device separability (file/day means)

| Rep | d_same | d_diff | rho | Day4→D123 acc |
| --- | -----: | -----: | --: | ------------: |
| C0 | 0.1113 | 0.1178 | 0.944385 | 50.0% |
| C1 | 0.1108 | 0.1177 | 0.941767 | 50.0% |
| C2 | 0.1554 | 0.1542 | 1.007255 | 50.0% |

C1/C2 are not chosen from a real target receiver.
This file does not start seed 0/1 training.
