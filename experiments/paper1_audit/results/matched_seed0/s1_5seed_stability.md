# S1 5-seed stability (Day4, frozen recipe)

verdict=CLEAN_FAIL  scale=skipped  full=skipped
s0_retrained=false  day5=unused  rx2=unused

## Clean vs C'

| seed | C' win | S1 win | Δ | S1 file | gate | collapse |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 43.8 | 43.4 | −0.4 | 75.0 | PASS | False |
| 1 | 44.3 | 43.4 | −0.9 | 75.0 | PASS | False |
| 2 | 46.1 | 41.7 | −4.4 | 66.7 | FAIL | False |
| 3 | 46.5 | 42.3 | −4.2 | 70.8 | FAIL | False |
| 4 | 46.7 | 43.0 | −3.7 | 75.0 | FAIL | False |

mean Δ=−2.72  pass=2/5  collapse=0

Pre-registered CLEAN_PASS needed ≥4/5, 0 collapse, mean Δ≥−2. Not moved after seeing data.

S1 windows cluster at 41.7–43.4 (not C1/D collapse). C' seeds 2–4 are the stronger C' draws (46.1–46.7). Stress skipped. RX2 closed. Do not retune.

Source: `s1_5seed_stability.json`.
