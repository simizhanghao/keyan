# EM-CR Debug Suite Report

**Date:** 2026-06-28  
**Init checkpoint:** `F_cross_attn_chirp_plain/seed_0/best.pt`  
**Settings:** 3 epochs, max-files=16, 32 win/file train, freeze-head-only, lr=1e-5, grad-clip=1.0  
**Eval:** 64 win/file, Day5 test (same as `eval_em_consistency.py`)

## Summary (file-acc %)

| Experiment | Clean | AWGN 30 dB | AWGN 20 dB | CFO 0.003 | NBI 10 dB |
|------------|-------|------------|------------|-----------|-----------|
| Init (full eval 256 win) | **83.3** | 70.8 | 29.2 | 4.2 | 87.5 |
| A clean-only FT | 79.2 | 62.5 | 33.3 | 4.2 | 79.2 |
| B EM-Aug CE | 79.2 | 62.5 | 33.3 | 4.2 | 83.3 |
| C weak CFO aug | 79.2 | **70.8** | 29.2 | 4.2 | 79.2 |
| D stopgrad KL | 79.2 | 62.5 | 33.3 | 4.2 | 79.2 |

Source: `debug_suite_summary.csv`

## Answers

1. **Clean-only fine-tune 是否保持 clean acc？**  
   在 debug 协议（64 win/file、16-file 子集、仅 classifier head）下，clean 从 full 基准 83.3% 降至 **79.2%**（约 −4 pp），**未出现 smoke 的 20.8% 级崩溃**。说明此前灾难性遗忘主要来自 **全主干更新 + 过长 epoch + 强 CFO 训练增强**，而非 eval 管线 bug。

2. **EM-Aug CE 是否比原 EM-CR 稳？**  
   比原 smoke（clean 20.8%）**明显更稳**；与 A clean-only 几乎同水平，NBI 10 dB 略好（83.3% vs 79.2%），**无鲁棒性实质增益**。

3. **CFO 是不是崩溃主因？**  
   原 smoke 含强 CFO 一致性 + 非冻结主干是主因。本 suite 中 **C weak CFO** 在 AWGN 30 dB 与 init 持平（70.8%），但 CFO 测试点仍 4.2%，**弱 CFO 训练未修复 CFO 敏感性**。

4. **KL consistency 是否有帮助？**  
   D stopgrad KL 与 A/B **无显著差异**；KL loss 下降但 **未带来 AWGN/CFO 提升**。

5. **是否值得 EM-CR full？**  
   **否。** 保守 3-epoch debug 未显示可重复的鲁棒增益，且 head-only 微调仍有小幅 clean 下降。建议将 EM-CR 作为 **negative / preliminary**，不进入 full training。

6. **论文定位**  
   第三创新点主线保持：**EM benchmark + embedding-based open-set**；EM-CR 写为 *preliminary consistency fine-tuning shows limited benefit and risk of forgetting under small-sample perturbation training; robust training left for future work*。

## Bug fixes applied this run

- `train_em_consistency.py`: CLI `epochs` 等训练超参不再被 checkpoint `args` 覆盖（此前误跑 20+ epoch）。
- `eval_openset_under_em.py`: 移除顶层 matplotlib 导入；补 `seed` 参数。
