# X3-D left/right OOB scale audit

Only one OOB half was multiplied by `0.5`; in-band IQ and the other OOB half
were unchanged. Values are clean-minus-perturbed accuracy drops in percentage
points, averaged over two seeds per held-out receiver.

| model | left mean / median | right mean / median |
|---|---:|---:|
| B1 | **44.41 / 47.17** | 17.04 / 18.13 |
| C' | **51.12 / 55.72** | 1.12 / 0.83 |

Left-side drops pass the 5 pp criterion on all 6 folds for both models. The
right-side response is weak for C' and materially smaller for B1, indicating
frequency-side asymmetry rather than a uniform broadband effect. Official blind
receivers remain sealed.
