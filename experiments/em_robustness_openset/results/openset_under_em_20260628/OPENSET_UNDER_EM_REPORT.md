# Open-Set Under EM Report

**Model:** clean-trained RF-HSTU `F_cross_attn_chirp_plain/seed_0`  
**Open-set:** 20 known + 4 unknown, split seeds 0/1/2  
**Scorers (main):** Prototype distance, Mahalanobis distance

## Prototype distance (mean ± std, 3 seeds)

| Condition | AUROC | EER | Known acc |
|-----------|-------|-----|-----------|
| clean | **0.917 ± 0.059** | 0.0 | 81.7% |
| AWGN 30 dB | 0.896 ± 0.106 | 0.017 | 70.0% |
| AWGN 20 dB | 0.646 ± 0.116 | 0.050 | 28.3% |
| CFO 0.001 | 0.713 ± 0.191 | 0.033 | 16.7% |
| CFO 0.003 | 0.492 ± 0.126 | 0.017 | 3.3% |
| NBI 10 dB | 0.908 ± 0.068 | 0.0 | 85.0% |
| Phase σ=0.03 | 0.575 ± 0.067 | 0.017 | 48.3% |
| IQ 3 dB / 5° | 0.700 ± 0.250 | 0.017 | 68.3% |
| Filter 0.2 | 0.858 ± 0.112 | 0.0 | 66.7% |
| Mixed AWGN+CFO | 0.429 ± 0.133 | 0.017 | 8.3% |

## Mahalanobis distance (mean ± std)

| Condition | AUROC | EER | Known acc |
|-----------|-------|-----|-----------|
| clean | 0.912 ± 0.062 | 0.0 | 81.7% |
| AWGN 30 dB | 0.771 ± 0.190 | 0.017 | 70.0% |
| CFO 0.003 | 0.608 ± 0.131 | 0.033 | 3.3% |
| Filter 0.2 | **0.946 ± 0.033** | 0.050 | 66.7% |

## Key findings

1. **EM stress 下 AUROC 普遍下降**（相对 clean 0.917）：AWGN 20 dB、CFO 0.003、mixed stress 最明显。
2. **CFO 同时破坏 known classification 与 unknown detection**：CFO 0.003 时 known acc ≈ 3.3%，Proto AUROC ≈ 0.49。
3. **Prototype vs Mahalanobis：** clean / AWGN 30 dB 下 Prototype 更稳；部分条件（filter 0.2、cfo 0.003）Mahalanobis AUROC 略高，但方差更大。
4. **Known acc 降但 AUROC 仍中等**：如 phase 0.03（known 48.3%，AUROC 0.58）——几何距离仍部分区分 unknown，但已知类识别已受损。
5. **复杂电磁下 open-set 更难**：mixed AWGN+CFO AUROC 0.43 vs clean 0.92。
6. **窄带干扰相对温和**：NBI 10 dB 下 AUROC 仍 ≈ 0.91，与 closed-set 结论一致。

## Main scorer recommendation

**Prototype distance** 作为主方法（与 clean open-set 一致、AWGN 30 dB 下更稳）；Mahalanobis 可作为附表对比。

Figures: `fig_auroc_under_em.pdf`, `fig_eer_under_em.pdf`, `fig_known_acc_under_em.pdf`
