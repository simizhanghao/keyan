# Day4 RX-style eval

Frozen 1C C' checkpoints. No retraining. Day5 unused.
Operators: tilt / OOB scale / gain / phase / noise. In-band scale locked at 1.

Frozen rule: mean window drop < 5pp → not strongly RX-entangled at inference.

| seed | C' win | RX win | drop pp | C' file | RX file | drop pp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 43.8 | 15.3 | 28.5 | 70.8 | 29.2 | 41.7 |
| 1 | 44.3 | 14.9 | 29.3 | 87.5 | 41.7 | 45.8 |
| 2 | 46.1 | 12.6 | 33.5 | 70.8 | 29.2 | 41.7 |
| 3 | 46.5 | 16.9 | 29.6 | 79.2 | 45.8 | 33.3 |
| 4 | 46.7 | 15.8 | 30.8 | 87.5 | 37.5 | 50.0 |

window drop all-5: [28.5, 29.3, 33.5, 29.6, 30.8]  mean 30.3±2.0
RX-entangled (mean drop ≥ 5pp): True

RCOF / Day5 / 1D / Hann/guard are not opened here.
