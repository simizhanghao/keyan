# F0 identity-first (seeds 2/3/4, Day4)

verdict=F0_GO  gate_a=True  gate_b=GATE_B_STRONG  gate_c=TRACKS_SCALE
pretrained=false  f1=false  day5=unused  rx2=unused

## Gate A — clean vs matching C'

| seed | C' win | F0 win | Δ | gate | collapse |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 46.1 | 52.9 | +6.8 | PASS | False |
| 3 | 46.5 | 48.5 | +2.0 | PASS | False |
| 4 | 46.7 | 48.5 | +1.8 | PASS | False |

mean Δ=+3.53  3/3  collapse=0

## Gate B — oob_scale drop vs own clean

| seed | clean | oob_scale | D | bin |
| ---: | ---: | ---: | ---: | --- |
| 2 | 52.9 | 50.9 | 2.0 | STRONG |
| 3 | 48.5 | 47.6 | 0.9 | STRONG |
| 4 | 48.5 | 48.0 | 0.5 | STRONG |

mean D_scale=1.1±0.8  GATE_B_STRONG

## Gate C — full RX drop vs own clean (recorded, not a retune knob)

| seed | clean | full RX | D | bin |
| ---: | ---: | ---: | ---: | --- |
| 2 | 52.9 | 48.5 | 4.4 | TRACKS_SCALE |
| 3 | 48.5 | 40.2 | 8.3 | TRACKS_SCALE |
| 4 | 48.5 | 43.4 | 5.1 | TRACKS_SCALE |

mean D_full=5.9±2.1  TRACKS_SCALE

F0_GO → later Human GO to expand seeds 0/1. Fail → F1, not S2/retune/RX2.
