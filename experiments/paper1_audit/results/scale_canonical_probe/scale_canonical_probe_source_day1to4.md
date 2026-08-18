# Phase 2A-0 scale–shape probe

files=96  windows=1536  days=[1, 2, 3, 4]  smoke=False
Day5 unused. Real RX2 unused. No training.

## Scale invariance (relative L2; cosine is scale-blind, recorded only)

| Rep | rel-L2 | cosine |
| --- | -----: | -----: |
| C0 | 0.4191 | -0.0000 |
| C1 | 0.0000 | -0.0000 |
| C2 | 0.0002 | 0.0000 |

## Device separability (file/day means)

| Rep | d_same | d_diff | rho | Day4→D123 acc |
| --- | -----: | -----: | --: | ------------: |
| C0 | 0.0237 | 0.0303 | 0.782682 | 29.2% |
| C1 | 0.0140 | 0.0215 | 0.652519 | 33.3% |
| C2 | 0.0189 | 0.0197 | 0.959518 | 29.2% |

C1/C2 are not chosen from a real target receiver.
This file does not start seed 0/1 training.
