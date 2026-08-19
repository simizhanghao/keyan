# 2A-3 scale path leak (Day4, no training)

files=2  windows=4  smoke=True
Day5 unused. Real RX2 unused. C1 seed 0 embedder only.

| Path | rel-L2 | Reading |
| --- | -----: | --- |
| oob_c1 | 0.0000 | STABLE |
| fft_inband_linear | 0.0000 | STABLE |
| fft_log_zscore | 0.0534 | LIVE |
| fft_inband_of_log_zscore | 0.0226 | WEAK |
| iq_time | 0.0438 | WEAK |
| amp_phase | 0.1153 | LIVE |
| cnn_stem | 0.0653 | LIVE |

rms(after RX) / rms(before) = 1.0036
verdict = LEAK_CONFIRMED
live paths: fft_log_zscore, amp_phase, cnn_stem

STABLE < 0.01; LIVE ≥ 0.05. Do not retune norms from this file.
