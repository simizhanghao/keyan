# Day4 label-oracle headroom

No training. Day5 unused. Primary pair B vs C'. Secondary B vs C.
Oracle = Main correct OR Full correct. Δ = oracle − max(Main, Full).

## Primary window: B Main vs C' Full ratio

| seed | Main win | Full win | best | oracle | Δ pp | Main-only | Full-only | collapsed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 22.2 | 43.8 | 43.8 | 52.5 | 8.7 | 8.7 | 30.3 | False |
| 1 | 49.2 | 44.3 | 49.2 | 67.9 | 18.7 | 23.6 | 18.7 | False |
| 2 | 5.7 | 46.1 | 46.1 | 49.2 | 3.1 | 3.1 | 43.5 | True |
| 3 | 4.6 | 46.5 | 46.5 | 48.2 | 1.7 | 1.7 | 43.6 | True |
| 4 | 4.3 | 46.7 | 46.7 | 48.2 | 1.6 | 1.6 | 43.9 | True |

File-Acc (same oracle, K=256; not the mechanism gate):

| seed | Main file | Full file | best | oracle | Δ pp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41.7 | 70.8 | 70.8 | 75.0 | 4.2 |
| 1 | 62.5 | 87.5 | 87.5 | 91.7 | 4.2 |
| 2 | 4.2 | 70.8 | 70.8 | 70.8 | 0.0 |
| 3 | 4.2 | 79.2 | 79.2 | 79.2 | 0.0 |
| 4 | 4.2 | 87.5 | 87.5 | 87.5 | 0.0 |

window Δ all-5: [8.7, 18.7, 3.1, 1.7, 1.6]  mean 6.8±7.3
window Δ Main-trained {0,1}: [8.7, 18.7]  mean 13.7±7.1
frozen DROP if window Δ < 5pp; collapsed seeds are diagnostic, not a moved threshold.

## Secondary window: B Main vs C Full zscore

| seed | Main win | Full win | best | oracle | Δ pp | Main-only | Full-only | collapsed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 22.2 | 27.9 | 27.9 | 40.9 | 13.0 | 13.0 | 18.7 | False |
| 1 | 49.2 | 31.8 | 49.2 | 62.4 | 13.2 | 30.5 | 13.2 | False |
| 2 | 5.7 | 31.2 | 31.2 | 35.3 | 4.2 | 4.2 | 29.6 | True |
| 3 | 4.6 | 28.0 | 28.0 | 30.9 | 2.9 | 2.9 | 26.3 | True |
| 4 | 4.3 | 36.8 | 36.8 | 39.3 | 2.5 | 2.5 | 35.0 | True |

File-Acc (same oracle, K=256; not the mechanism gate):

| seed | Main file | Full file | best | oracle | Δ pp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41.7 | 66.7 | 66.7 | 70.8 | 4.2 |
| 1 | 62.5 | 70.8 | 70.8 | 87.5 | 16.7 |
| 2 | 4.2 | 58.3 | 58.3 | 62.5 | 4.2 |
| 3 | 4.2 | 66.7 | 66.7 | 66.7 | 0.0 |
| 4 | 4.2 | 66.7 | 66.7 | 70.8 | 4.2 |

window Δ all-5: [13.0, 13.2, 4.2, 2.9, 2.5]  mean 7.2±5.5
window Δ Main-trained {0,1}: [13.0, 13.2]  mean 13.1±0.1
frozen DROP if window Δ < 5pp; collapsed seeds are diagnostic, not a moved threshold.

Utility gate is not opened here. Shuffle and RX-style are not started.
