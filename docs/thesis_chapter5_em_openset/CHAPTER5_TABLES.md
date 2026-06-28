# Chapter 5 Tables

## Table 5-1 — Closed-set EM robustness (file-level accuracy, %)

**Model:** RF-HSTU seed0，256 windows/file，Day5 test。

| Perturbation | Clean | Moderate | Severe |
|--------------|-------|----------|--------|
| AWGN SNR (dB) | 83.3 (≥40) | 70.8 (30) / 37.5 (25) | 4.2 (≤10) |
| CFO norm | 83.3 (0) | 20.8 (0.001) | 4.2 (≥0.003) |
| Narrowband SIR (dB) | 83.3 (30) | 83.3–87.5 (10–20) | 75.0 (0) |
| Phase noise σ | 83.3 (0) | 79.2 (0.01) | 16.7 (≥0.05) |
| IQ amp (dB) | 83.3 (0) | 83.3 (1) | 50.0 (5) |
| Filter tilt norm | 83.3 (0) | 75.0 (0.1) | 50.0 (0.4) |

Source: `experiments/em_robustness_openset/results/em_full_20260628/em_robustness_summary.csv`

## Table 5-2 — Robustness summary by perturbation family

| Family | Clean (%) | Avg robust (%) | Min (%) | Drop (pp) |
|--------|-----------|----------------|---------|-----------|
| AWGN | 83.33 | 31.25 | 4.17 | 79.17 |
| CFO | 83.33 | 6.25 | 4.17 | 79.17 |
| Phase noise | 83.33 | 40.62 | 16.67 | 66.67 |
| Filter drift | 83.33 | 63.89 | 50.00 | 33.33 |
| IQ imbalance | 83.33 | 65.28 | 50.00 | 33.33 |
| Narrowband | 83.33 | 82.29 | 75.00 | 8.33 |

Source: `em_robustness_by_perturbation.csv`

## Table 5-3 — Open-set authentication (clean, 3 seeds)

| Scorer | AUROC | EER | FAR | FRR | Known acc |
|--------|-------|-----|-----|-----|-----------|
| Prototype distance | 0.917±0.059 | 0.0 | 0.17 | 0.13 | 81.7% |
| Mahalanobis | 0.913±0.062 | 0.0 | 0.25 | 0.13 | 81.7% |
| Energy | 0.575±0.118 | 0.017 | 0.58 | 0.33 | 81.7% |
| MSP | 0.425±0.221 | 0.033 | 0.58 | 0.52 | 81.7% |

Source: `openset_clean_summary.csv`

## Table 5-4 — EM-CR recommended training ranges

| Perturbation | Moderate range | Notes |
|--------------|------------------|-------|
| AWGN | 30–15 dB | No 5/0 dB in training |
| CFO norm | 0.001–0.01 | No 0.03+ in initial training |
| Narrowband SIR | 30–10 dB | |
| Phase noise σ | 0.01–0.05 | |
| IQ amp / phase | 1–3 dB / 2–5° | |
| Filter tilt | 0.1–0.2 | |

## Table 5-5 — CNN-IQ baseline status

CNN-IQ EM baseline **pending** dedicated full eval. Candidate checkpoint:  
`outputs/paper_ready_v3/step1_phase7_clean/runs/A_cnn_iq/seed_0/best.pt`

Current full curves focus on RF-HSTU (Chapters 3–4 backbone).
