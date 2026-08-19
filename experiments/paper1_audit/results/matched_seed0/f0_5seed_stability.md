# F0 5-seed stability (Day4, identity-first)

verdict=F0_5SEED_GO  clean=True  scale=SCALE_STRONG  full=TRACKS_SCALE
retrained_234=false  f1=false  day5=unused  rx2=unused

## Clean vs matching C'

| seed | C' win | F0 win | Δ | gate | collapse |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 43.8 | 47.7 | +3.9 | PASS | False |
| 1 | 44.3 | 45.5 | +1.2 | PASS | False |
| 2 | 46.1 | 52.9 | +6.8 | PASS | False |
| 3 | 46.5 | 48.5 | +2.0 | PASS | False |
| 4 | 46.7 | 48.5 | +1.8 | PASS | False |

mean Δ=+3.14  5/5  collapse=0

## oob_scale drop vs own clean

| seed | clean | oob_scale | D | bin |
| ---: | ---: | ---: | ---: | --- |
| 0 | 47.7 | 45.1 | 2.6 | STRONG |
| 1 | 45.5 | 43.9 | 1.6 | STRONG |
| 2 | 52.9 | 50.9 | 2.0 | STRONG |
| 3 | 48.5 | 47.6 | 0.9 | STRONG |
| 4 | 48.5 | 48.0 | 0.5 | STRONG |

mean D_scale=1.5±0.8  SCALE_STRONG

## full RX drop vs own clean (recorded, not a retune knob)

| seed | clean | full RX | D | bin |
| ---: | ---: | ---: | ---: | --- |
| 0 | 47.7 | 40.0 | 7.7 | TRACKS_SCALE |
| 1 | 45.5 | 36.5 | 9.0 | TRACKS_SCALE |
| 2 | 52.9 | 48.5 | 4.4 | TRACKS_SCALE |
| 3 | 48.5 | 40.2 | 8.3 | TRACKS_SCALE |
| 4 | 48.5 | 43.4 | 5.1 | TRACKS_SCALE |

mean D_full=6.9±2.0  TRACKS_SCALE

F0_5SEED_GO → later Human GO for real RX1↔RX2. HOLD → do not open RX2 / F1 / retune.
