# EM Robustness Full Report

**Output:** `/data1/hcc/llm4RF/experiments/em_robustness_openset/results/em_full_20260628`
**Checkpoint:** `outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt`

## Clean baseline
- File-level accuracy (clean-equivalent AWGN≥100 dB): **83.33%**

## Perturbation ranking (by max accuracy drop)

| Family | Clean (%) | Avg robust (%) | Min (%) | Drop (pp) |
|--------|-----------|----------------|---------|-----------|
| AWGN | 83.33 | 31.25 | 4.17 | 79.17 |
| CFO | 83.33 | 6.25 | 4.17 | 79.17 |
| Mixed stress | 83.33 | 6.94 | 4.17 | 79.17 |
| Phase noise | 83.33 | 40.62 | 16.67 | 66.67 |
| Filter drift | 83.33 | 63.89 | 50.0 | 33.33 |
| IQ imbalance (amp) | 83.33 | 65.28 | 50.0 | 33.33 |
| Narrowband | 83.33 | 82.29 | 75.0 | 8.33 |

**Most destructive:** AWGN (drop 79.17 pp).

## AWGN
- Clean 83.3%; 30 dB → ~70.8%; steep cliff below 25 dB.

## CFO
- norm≥0.003 collapses to ~4.2%; norm=0.001 still ~20.8%.

## Narrowband
- Relatively mild: SIR 30–10 dB stays ~83–87.5%; 0 dB → 75%.

## Phase noise / IQ / Filter
- Phase σ≥0.05 and IQ amp≥3 dB cause strong degradation.
- Filter tilt moderate (0.1–0.2) → 75–67%.

## Recommended perturbation ranges for EM-CR

Use **moderate** ranges only for initial EM-CR training:

- AWGN SNR (dB): **30–15**
- CFO norm: **0.001–0.01**
- Narrowband SIR (dB): **30–10**
- Phase noise σ: **0.01–0.05**
- IQ amp (dB): **1–3**
- IQ phase (deg): **2–5**
- Filter tilt norm: **0.1–0.2**

**Forbidden for initial EM-CR training:**

- CFO norm 0.03 / 0.05 / 0.10
- AWGN 5 / 0 dB
- Extreme mixed stress presets (test-only)
