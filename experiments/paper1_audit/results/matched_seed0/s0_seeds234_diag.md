# S0 seeds 2/3/4 diagnostic (after S1 CLEAN_FAIL)

reading=SCALE_TAX  s1_5seed=CLEAN_FAIL_unchanged  day5=unused  rx2=unused

| seed | C' win | S0 win | S1 win | Δ S0 | Δ S1 | S0 gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 43.8 | 46.4 | 43.4 | +2.6 | -0.4 | PASS |
| 1 | 44.3 | 43.1 | 43.4 | -1.2 | -0.9 | PASS |
| 2 | 46.1 | 46.1 | 41.7 | +0.0 | -4.4 | PASS |
| 3 | 46.5 | 47.2 | 42.3 | +0.7 | -4.2 | PASS |
| 4 | 46.7 | 43.2 | 43.0 | -3.5 | -3.7 | FAIL |

Focus 2/3/4: S0 mean Δ=-0.93 (2/3 PASS); S1 mean Δ=-4.10

PAIRING_TAX = S0 fails ≥2/3 of seeds 2–4 (two-forward / pairing).
SCALE_TAX = S0 holds ≥2/3 while S1 already failed those seeds.
S1 CLEAN_FAIL is not moved. RX2 closed. Do not retune.
