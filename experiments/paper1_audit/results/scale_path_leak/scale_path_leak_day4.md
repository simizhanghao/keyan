# 2A-3 scale path leak (Day4, no training)

files=24  windows=384  smoke=False
Day5 unused. Real RX2 unused. C1 seed 0 embedder only.

| Path | rel-L2 | Reading |
| --- | -----: | --- |
| oob_c1 | 0.0000 | STABLE |
| fft_inband_linear | 0.0000 | STABLE |
| fft_log_zscore | 0.0488 | WEAK |
| fft_inband_of_log_zscore | 0.0213 | WEAK |
| iq_time | 0.0458 | WEAK |
| amp_phase | 0.1192 | LIVE |
| cnn_stem | 0.0591 | LIVE |

rms(after RX) / rms(before) = 1.0046
verdict = LEAK_CONFIRMED
live paths: amp_phase, cnn_stem

STABLE < 0.01; LIVE ≥ 0.05. Do not retune norms from this file.
