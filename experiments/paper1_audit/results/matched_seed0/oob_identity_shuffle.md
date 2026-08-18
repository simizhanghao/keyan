# Day4 OOB identity shuffle

Primary: C' Full ratio vs shuffled C'. Main IQ/label kept. OOB from a same-day different device.
Train donors reshuffle each epoch. Eval donors are frozen per window. Day5 unused.

Frozen rule: mean window drop < 5pp → identity claim shrinks. Gate is not moved.

| seed | C' win | shuffle win | drop pp | C' file | shuffle file | drop pp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 43.8 | 53.2 | -9.4 | 70.8 | 75.0 | -4.2 |
| 1 | 44.3 | 39.8 | 4.4 | 87.5 | 66.7 | 20.8 |
| 2 | 46.1 | 4.6 | 41.5 | 70.8 | 4.2 | 66.7 |
| 3 | 46.5 | 5.0 | 41.5 | 79.2 | 4.2 | 75.0 |
| 4 | 46.7 | 59.4 | -12.7 | 87.5 | 70.8 | 16.7 |

window drop all-5: [-9.4, 4.4, 41.5, 41.5, -12.7]  mean 13.1±26.7
identity claim shrinks (mean drop < 5pp): False

Utility gate / RCOF / RX-style are not opened here.
