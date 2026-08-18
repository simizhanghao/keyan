# Chapter 5 Tables

所有数值来源见 `experiments/em_robustness_openset/results/` 对应 CSV。

---

## Table 5-1 — EM perturbation benchmark

| 扰动类型 | 物理含义 | 测试强度（示例） | 主要影响 |
|----------|----------|------------------|----------|
| AWGN | 信道/接收机噪声 | SNR 40→0 dB | 25 dB 以下急剧崩溃 |
| CFO | 载波频偏 | norm 0→0.1 | **最强破坏源之一** |
| Narrowband (NBI) | 带内窄带干扰 | SIR 30→0 dB | 相对温和 |
| Phase noise | 相位抖动 | σ 0→0.1 | 中等，σ≥0.05 显著下降 |
| IQ imbalance | 幅度/相位失衡 | amp 0→5 dB | 中等退化 |
| Filter drift | 前端频响倾斜 | tilt 0→0.4 | 中等退化 |
| Mixed stress | AWGN+CFO 等组合 | preset | 接近最差单扰动 |

---

## Table 5-2 — Closed-set robustness summary (Ours, seed0, %)

| Perturbation | Clean Acc | Moderate Acc | Severe Acc | Drop (pp) |
|--------------|-----------|--------------|------------|-----------|
| AWGN | 83.3 | 70.8 (30 dB) / 37.5 (25 dB) | 4.2 (≤10 dB) | 79.2 |
| CFO | 83.3 | 20.8 (0.001) | 4.2 (≥0.003) | 79.2 |
| Narrowband | 83.3 | 83.3–87.5 (SIR 10–30) | 75.0 (0 dB) | 8.3 |
| Phase noise | 83.3 | 79.2 (σ=0.01) | 16.7 (σ≥0.05) | 66.7 |
| IQ imbalance | 83.3 | 83.3 (1 dB) | 50.0 (5 dB) | 33.3 |
| Filter drift | 83.3 | 75.0 (0.1) | 50.0 (0.4) | 33.3 |

**Family avg robust / min：** AWGN 31.3% / 4.2%；CFO 6.3% / 4.2%；Narrowband 82.3% / 75.0%。

Source: `em_full_20260628/em_robustness_summary.csv`, `em_robustness_by_perturbation.csv`

---

## Table 5-3 — CNN-IQ vs Ours under EM (file-acc %)

| Condition | CNN-IQ | Ours | Gain |
|-----------|--------|------|------|
| Clean | 62.5 | 83.3 | +20.8 |
| AWGN 30 dB | 62.5 | 70.8 | +8.3 |
| CFO 0.003 | 4.2 | 4.2 | 0.0 |
| NBI 10 dB | 29.2 | 87.5 | +58.3 |
| Filter drift 0.2 | 50.0 | 66.7 | +16.7 |

Checkpoint: `A_cnn_iq/seed_0` vs `F_cross_attn_chirp_plain/seed_0`  
Source: `em_full_20260628_cnn/`, `em_full_20260628/`

---

## Table 5-4 — Open-set clean authentication (3 seeds)

| Scorer | AUROC | EER | FAR | FRR | Known Acc |
|--------|-------|-----|-----|-----|-----------|
| **Prototype** | **0.917±0.059** | 0.0 | 0.17 | 0.13 | 81.7% |
| **Mahalanobis** | **0.913±0.062** | 0.0 | 0.25 | 0.13 | 81.7% |
| Energy | 0.575±0.118 | 0.017 | 0.58 | 0.33 | 81.7% |
| MSP | 0.425±0.221 | 0.033 | 0.58 | 0.52 | 81.7% |

Source: `openset_full_20260628_1123/`, `openset_clean_summary.csv`

---

## Table 5-5 — Open-set under EM (3 seeds, mean)

| Condition | Proto AUROC | Maha AUROC | Known Acc | Main finding |
|-----------|-------------|------------|-----------|--------------|
| clean | 0.917 | 0.912 | 81.7% | 基线 open-set 可靠 |
| AWGN 30 dB | 0.896 | 0.771 | 70.0% | AWGN 对拒识较温和 |
| AWGN 20 dB | 0.646 | 0.650 | 28.3% | 强 AWGN 双任务受损 |
| **CFO 0.003** | **0.492** | 0.608 | **3.3%** | **known + unknown 双崩** |
| NBI 10 dB | 0.908 | 0.900 | 85.0% | 与闭集一致，相对温和 |
| Filter 0.2 | 0.858 | 0.946 | 66.7% | Maha 个别条件略优 |
| Mixed AWGN+CFO | 0.429 | 0.579 | 8.3% | 复合应力最难 |

Source: `openset_under_em_20260628/openset_under_em_summary.csv`

---

## Table 5-6 — EM-CR debug suite (64 win/file eval)

| Experiment | Clean Acc | AWGN 30 dB | Conclusion |
|------------|-----------|------------|------------|
| A Clean-only FT | 79.2 | 62.5 | 不崩，管线无 bug |
| B EM-Aug CE | 79.2 | 62.5 | 无鲁棒增益 |
| C Weak CFO | 79.2 | **70.8** | AWGN 持平 init，CFO 仍失效 |
| D Stopgrad KL | 79.2 | 62.5 | KL 无显著帮助 |

**Decision：** 不进入 EM-CR full training；论文写 preliminary / future work。

Source: `emcr_debug_20260628/debug_suite_summary.csv`, `EMCR_DEBUG_REPORT.md`
