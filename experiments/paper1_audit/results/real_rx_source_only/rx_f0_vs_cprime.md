# Real RX source-only: F0 vs C' (window Acc primary)

verdict=RX_FAIL  pooled_Δ_window=-0.52 pp
rx1_to_rx2 mean Δ=-0.95  rx2_to_rx1 mean Δ=-0.10
day4_ckpt=false  oracle=false  f1=false  cnn=false

| direction | seed | C' win | F0 win | Δ win | C' file | F0 file | Δ file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rx1_to_rx2 | 0 | 15.5 | 13.9 | -1.6 | 16.7 | 20.8 | +4.1 |
| rx1_to_rx2 | 1 | 16.9 | 16.6 | -0.3 | 16.7 | 16.7 | +0.0 |
| rx2_to_rx1 | 0 | 15.1 | 14.3 | -0.8 | 12.5 | 16.7 | +4.2 |
| rx2_to_rx1 | 1 | 15.4 | 16.0 | +0.6 | 12.5 | 8.3 | -4.2 |

STRONG_GO: both dirs F0>C', pooled ≥8, at least one dir ≥10.
WEAK_GO: both dirs >0 and 4≤pooled<8. FAIL: pooled<4 or any dir ≤−2.
Do not retune from target Acc. Do not open F1/CNN from FAIL.
