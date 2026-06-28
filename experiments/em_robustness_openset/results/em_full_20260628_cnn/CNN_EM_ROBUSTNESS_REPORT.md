# CNN-IQ EM Robustness Report

**CNN dir:** `experiments/em_robustness_openset/results/em_full_20260628_cnn`
**Ours dir:** `experiments/em_robustness_openset/results/em_full_20260628`

## Key comparison (file-acc %)

| Condition | CNN-IQ | Ours | Δ (Ours-CNN) |
|-----------|--------|------|--------------|
| AWGN 30 dB | 62.5 | 70.83 | +8.3 |
| CFO 0.003 | 4.17 | 4.17 | +0.0 |
| NBI 10 dB | 29.17 | 87.5 | +58.3 |
| Clean | 62.5 | 83.33 | +20.8 |

## CNN ranking

- Mixed stress: drop 79.16 pp
- AWGN: drop 58.33 pp
- CFO: drop 58.33 pp
- Phase noise: drop 58.33 pp
- IQ imbalance (amp): drop 54.17 pp
- Filter drift: drop 41.67 pp
- Narrowband: drop -4.17 pp