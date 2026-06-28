# Chapter 5 Results

## Phase A.5 Smoke Audit (`smoke_audit_20260628_1118`)

**Verdict:** Audit passed **YES** — clean-equivalent spread 0.0 pp.

**根因澄清：** 此前将 AWGN **30 dB**（真实扰动）与 CFO **norm=0**（clean）对比。所有 clean-equivalent 条件均为 **83.3%**（32 windows/file）。

---

## Phase B — Closed-set EM robustness full (`em_full_20260628`)

**Model:** RF-HSTU `F_cross_attn_chirp_plain/seed_0`  
**Protocol:** Day5 test, 256 windows/file, mean-logits voting  
**Python/CUDA:** `/new_nfs/haiyu/anaconda3/bin/python` (cu128)

### Clean baseline

- File-level accuracy (AWGN ≥100 dB / clean-equivalent): **83.3%**

### AWGN

- 40 dB: 83.3%；30 dB: **70.8%**；25 dB: 37.5%；20 dB: 29.2%
- 15 dB 及以下急剧崩溃至 ~4.2%
- **结论：** 25 dB 以下为主要退化区；30 dB 仍有明显但可解释的下降

### CFO

- norm=0: 83.3%；0.001: 20.8%；≥0.003: **~4.2%**
- **结论：** norm≥0.01 仍严重退化；温和网格 0.001–0.01 适合主曲线与 EM-CR moderate 训练

### Narrowband

- SIR 30→10 dB: **83.3–87.5%**；0 dB: 75%
- **结论：** 相对温和，是最不破坏的单一扰动族

### Phase noise / IQ / Filter

- Phase σ≥0.05：16.7%；IQ amp 3 dB：62.5%；Filter tilt 0.2：66.7%
- Filter 0.1–0.2 为中等退化

### Mixed severe stress

- awgn_cfo / awgn_nbi: ~8.3%；cfo_iq: 4.2%

### Ranking（按 max drop）

1. AWGN / CFO（~79 pp drop）
2. Phase noise（~67 pp）
3. IQ / Filter（~33 pp）
4. Narrowband（~8 pp）

**Average robust accuracy（非 clean 点均值）：** AWGN 31.3%；CFO 6.3%；Narrowband 82.3%

---

## Phase D — Open-set full (`openset_full_20260628_1123`)

20 known + 4 unknown，3 split seeds，clean。

| Scorer | AUROC (mean±std) | EER | Known acc |
|--------|------------------|-----|-----------|
| **Prototype** | **0.917±0.059** | 0.0 | 81.7% |
| **Mahalanobis** | **0.913±0.062** | 0.0 | 81.7% |
| Energy | 0.575±0.118 | 0.017 | 81.7% |
| MSP | 0.425±0.221 | 0.033 | 81.7% |

- **最佳 scorer：** Prototype distance（略优于 Mahalanobis）
- MSP/Energy 明显弱于 embedding-based 方法
- seed0 Proto AUROC=1.0 为小样本现象；seeds 1–2 约 0.86–0.88，结论更稳

---

## 第三创新点主结果状态

**主线（已定稿雏形）：**
- EM perturbation benchmark（Ours full + CNN-IQ baseline）
- Open-set authentication（clean + under EM，3 seeds）

**辅助 / negative：**
- EM-CR debug suite：保守 head-only 3-epoch **未通过 full 门槛**；原 smoke 灾难性遗忘来自非冻结主干 + 强 CFO + 过长训练。见 `emcr_debug_20260628/EMCR_DEBUG_REPORT.md`

---

## CNN-IQ EM baseline (`em_full_20260628_cnn`)

| Condition | CNN-IQ | Ours |
|-----------|--------|------|
| Clean | 62.5% | 83.3% |
| AWGN 30 dB | 62.5% | 70.8% |
| CFO 0.003 | 4.2% | 4.2% |
| NBI 10 dB | 29.2% | 87.5% |

Ours 在 clean 与多数 EM 应力下优于 CNN-IQ；窄带干扰优势最大（+58.3 pp at SIR 10 dB）。

---

## EM-CR debug suite (`emcr_debug_20260628`)

| Experiment | Clean (64 win) | AWGN 30 dB |
|------------|----------------|------------|
| A clean-only FT | 79.2% | 62.5% |
| B EM-Aug CE | 79.2% | 62.5% |
| C weak CFO | 79.2% | 70.8% |
| D stopgrad KL | 79.2% | 62.5% |

**决策：** 不进入 EM-CR full；论文中作为 preliminary negative result。

---

## 待完成

- ~~Open-set under EM~~ ✅ `openset_under_em_20260628/`

## Open-set under EM (`openset_under_em_20260628`)

Clean-trained Ours，3 seeds，Prototype / Mahalanobis。

| Condition | Proto AUROC | Known acc |
|-----------|-------------|-----------|
| clean | 0.917±0.059 | 81.7% |
| AWGN 30 dB | 0.896±0.106 | 70.0% |
| CFO 0.003 | 0.492±0.126 | 3.3% |
| NBI 10 dB | 0.908±0.068 | 85.0% |
| Mixed AWGN+CFO | 0.429±0.133 | 8.3% |

CFO / mixed stress 同时摧毁 known acc 与 open-set AUROC；NBI 相对温和。

---

| 扰动 | Moderate（训练） | 禁止（仅测试） |
|------|------------------|----------------|
| AWGN | 30–15 dB | 5 / 0 dB |
| CFO norm | 0.001–0.01 | 0.03 / 0.05 / 0.10 |
| Narrowband SIR | 30–10 dB | 0 dB 作为 severe |
| Phase σ | 0.01–0.05 | 0.10 |
| IQ amp | 1–3 dB | ≥5 dB |
| Filter tilt | 0.1–0.2 | ≥0.4 |
| Mixed stress | — | 全部 severe preset |

---

## 产出路径

- `experiments/em_robustness_openset/results/em_full_20260628/`（CSV + 报告 + RUN_MANIFEST）
- `docs/thesis_chapter5_em_openset/figures/fig5_*.pdf`
