# Open-Set Authentication Report (clean, full)

**Output:** `/data1/hcc/llm4RF/experiments/em_robustness_openset/results/openset_full_20260628_1123`

## Aggregate (3 split seeds)

| Scorer | AUROC (mean±std) | EER | FAR | FRR | Known acc (%) |
|--------|------------------|-----|-----|-----|---------------|
| energy | 0.575±0.118 | 0.017 | 0.583 | 0.333 | 81.7 |
| mahalanobis | 0.912±0.062 | 0.000 | 0.250 | 0.133 | 81.7 |
| msp | 0.425±0.221 | 0.033 | 0.583 | 0.517 | 81.7 |
| proto_dist | 0.917±0.059 | 0.000 | 0.167 | 0.133 | 81.7 |

## Per-seed Proto / Mahalanobis AUROC

- seed 0: msp=0.175, proto_dist=1.000, mahalanobis=1.000
- seed 1: msp=0.388, proto_dist=0.875, mahalanobis=0.863
- seed 2: msp=0.713, proto_dist=0.875, mahalanobis=0.875

## Interpretation
- Prototype distance and Mahalanobis outperform MSP/Energy on average.
- Seed 0 Proto/Maha AUROC=1.0 is likely small-sample; seeds 1–2 show 0.86–0.88.
- Threshold selected on validation only (see eval_openset_auth.py).

## CNN-IQ baseline
CNN-IQ EM baseline pending dedicated checkpoint in EM full script; current full run uses RF-HSTU (Chapters 3–4 backbone). Candidate: `outputs/paper_ready_v3/step1_phase7_clean/runs/A_cnn_iq/seed_0/best.pt`.