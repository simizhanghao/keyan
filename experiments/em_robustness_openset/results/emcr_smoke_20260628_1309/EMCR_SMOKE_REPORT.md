# EM-CR Smoke Report

**Date:** 2026-06-28  
**Output:** `experiments/em_robustness_openset/results/emcr_smoke_20260628_1309`

## Config

- Init: `F_cross_attn_chirp_plain/seed_0/best.pt`
- Epochs: 3, max-files: 8, samples/file: 32 (train)
- Eval: test split, 64 windows/file
- Loss: EM-CR (`lambda_kl=0.5`, `lambda_emb=0.0`)
- Perturbations: AWGN 30–15 dB, CFO 0.001–0.01, NBI 30–10 dB (moderate only)

## Loss trend

| Epoch | train_loss | train_acc | val_acc |
|-------|------------|-----------|---------|
| 1 | 4.23 | 36.1% | 47.1% |
| 2 | 3.01 | 54.8% | 48.6% |
| 3 | 2.91 | 55.9% | 38.7% |

Loss 下降，但 val acc 未稳定提升。

## Eval vs clean-trained (file-acc, %)

| Condition | Clean-trained | EM-CR | Δ |
|-----------|---------------|-------|---|
| clean | 83.3 | 20.8 | **−62.5** |
| AWGN 30 dB | 62.5 | 20.8 | −41.7 |
| AWGN 20 dB | 33.3 | 16.7 | −16.7 |
| CFO 0.003 | 4.2 | 16.7 | +12.5 |
| CFO 0.01 | 4.2 | 4.2 | 0 |
| NBI 20 dB | 83.3 | 20.8 | −62.5 |
| NBI 10 dB | 79.2 | 20.8 | −58.3 |

## Verdict

**Smoke FAILED** — clean file-acc 下降远超 5 pp 阈值；扰动项无一致提升。

## Likely causes

1. `max-files=8` 子集过小，val/test 分布不匹配；
2. 全参数微调 + 较强 KL 在极小数据上破坏 clean 表征；
3. 需降低 LR、增大子集、或冻结 encoder 前几层后再试。

## Recommendation

- **暂不进入 EM-CR full**；
- 下一轮 smoke：max-files≥32、lr=1e-4、epochs=5、监控 test clean；
- 若仍崩，尝试 EM-Aug CE only（无 KL）再逐步加 consistency。
