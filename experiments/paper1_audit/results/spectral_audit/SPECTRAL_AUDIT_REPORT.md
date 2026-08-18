# Paper 1 Audit 1B — OOB spectral audit

No training. Day5 was not loaded. Frozen paper numbers were not rewritten.

- files (Day1-4): 96
- windows/file: 8
- configs: 30
- decision: **GO_TWO_CANDIDATES**
- leakage_flag: False

## Candidates for 1C (Day4 selection only)

### Legacy control

- `{'norm': 'legacy_zscore', 'fft_window': 'rectangular', 'guard_hz': 0.0, 'rho_day': 0.9456311588340254, 'probe_acc': 0.25}`

### Corrected candidate

- `{'norm': 'ratio', 'fft_window': 'rectangular', 'guard_hz': 0.0, 'rho_day': 0.8035753243638373, 'probe_acc': 0.3333333333333333}`

## Full table

| norm | window | guard_kHz | rho_day | d_same | d_diff | Day4 probe |
|---|---|---:|---:|---:|---:|---:|
| legacy_zscore | rectangular | 0.0 | 0.946 | 0.012 | 0.012 | 25.0% |
| legacy_zscore | rectangular | 12.5 | 0.969 | 0.013 | 0.014 | 8.3% |
| legacy_zscore | rectangular | 25.0 | 0.985 | 0.012 | 0.012 | 8.3% |
| legacy_zscore | hann | 0.0 | 0.931 | 0.009 | 0.010 | 29.2% |
| legacy_zscore | hann | 12.5 | 0.957 | 0.010 | 0.010 | 29.2% |
| legacy_zscore | hann | 25.0 | 0.981 | 0.009 | 0.009 | 25.0% |
| corrected_zscore | rectangular | 0.0 | 0.946 | 0.012 | 0.012 | 25.0% |
| corrected_zscore | rectangular | 12.5 | 0.969 | 0.013 | 0.014 | 8.3% |
| corrected_zscore | rectangular | 25.0 | 0.985 | 0.012 | 0.012 | 8.3% |
| corrected_zscore | hann | 0.0 | 0.931 | 0.009 | 0.010 | 29.2% |
| corrected_zscore | hann | 12.5 | 0.957 | 0.010 | 0.010 | 29.2% |
| corrected_zscore | hann | 25.0 | 0.981 | 0.009 | 0.009 | 25.0% |
| oob_only_zscore | rectangular | 0.0 | 0.946 | 0.021 | 0.022 | 25.0% |
| oob_only_zscore | rectangular | 12.5 | 0.967 | 0.044 | 0.045 | 8.3% |
| oob_only_zscore | rectangular | 25.0 | 0.983 | 0.069 | 0.070 | 8.3% |
| oob_only_zscore | hann | 0.0 | 0.930 | 0.016 | 0.018 | 29.2% |
| oob_only_zscore | hann | 12.5 | 0.951 | 0.037 | 0.039 | 29.2% |
| oob_only_zscore | hann | 25.0 | 0.970 | 0.064 | 0.066 | 25.0% |
| ratio | rectangular | 0.0 | 0.804 | 0.032 | 0.039 | 33.3% |
| ratio | rectangular | 12.5 | 0.896 | 0.035 | 0.039 | 20.8% |
| ratio | rectangular | 25.0 | 0.968 | 0.032 | 0.033 | 12.5% |
| ratio | hann | 0.0 | 0.814 | 0.029 | 0.036 | 16.7% |
| ratio | hann | 12.5 | 0.901 | 0.040 | 0.045 | 16.7% |
| ratio | hann | 25.0 | 1.005 | 0.048 | 0.048 | 20.8% |
| log_ratio | rectangular | 0.0 | 0.976 | 0.001 | 0.001 | 12.5% |
| log_ratio | rectangular | 12.5 | 0.987 | 0.001 | 0.001 | 8.3% |
| log_ratio | rectangular | 25.0 | 0.994 | 0.001 | 0.001 | 4.2% |
| log_ratio | hann | 0.0 | 1.039 | 0.002 | 0.001 | 20.8% |
| log_ratio | hann | 12.5 | 1.043 | 0.001 | 0.001 | 20.8% |
| log_ratio | hann | 25.0 | 1.047 | 0.001 | 0.001 | 12.5% |

## How to read this

- `rho_day < 1`: same device across days is closer than different devices on the same day.
- Day4 probe is nearest-centroid using Day1-3 file means. Chance is 4.17%.
- If Hann+guard wipes device probe while legacy rectangular 0 kHz stays high: leakage RED.
- 1C must still retrain matched Main vs Full; this step only picks two OOB preprocesses.

