# 实验路线表（Week 1–4）

## Week 1：基准 + 诊断（当前）

| # | 任务 | 状态 |
|---|------|------|
| W1-1 | Chapter 3 结果整理 | done |
| W1-2 | cross-receiver baseline 复验 | done |
| W1-3 | EM 扰动脚本 | done |
| W1-4 | **cross-receiver diagnosis pipeline** | done |

```bash
GPU_ID=1 bash experiments/cross_receiver_diagnosis/run_diagnosis.sh
# 报告：experiments/cross_receiver_diagnosis/CROSS_RECEIVER_DIAGNOSIS_REPORT.md
```

## Week 2：方法（诊断驱动，非盲试）

| # | 任务 | 前提 |
|---|------|------|
| W2-1 | RCPA few-shot calibration | diagnosis 确认 OOB entanglement |
| W2-2 | OOB receiver response equalization | rx2/rx1 ratio > 1.4 |
| W2-3 | TTA 对比（可选） | collapse 严重时仅作 negative result |

## Week 3：电磁扰动（创新点 3 前半）

| 任务 | 扰动类型 | x 轴 | y 轴 |
|------|----------|------|------|
| 鲁棒性曲线 ×6 | AWGN, CFO, phase noise, NBI, IQ imbalance, filter drift | 扰动强度 | file-level accuracy |
| 一致性训练（可选） | KL(p(y\|x) \| p(y\|A(x))) | λ sweep | clean vs perturbed acc |

## Week 4：开放集认证（创新点 3 后半）

| 设置 | 训练 | 测试 | 指标 |
|------|------|------|------|
| hold-out 4 unknown devices | 20 known | known + unknown | AUROC, EER, FPR@95TPR |
| 打分函数 | — | max-softmax / energy / prototype distance | FAR / FRR |

---

## 创新点 2 实验表模板

| 设置 | 目标标签 | 方法 | RX1→RX2 | RX2→RX1 |
|------|----------|------|---------|---------|
| Source-only | 否 | CNN-IQ | (baseline) | (baseline) |
| Source-only | 否 | Ours | (baseline) | (baseline) |
| Unsupervised TTA | 否 | Ours + entropy_min | TBD | TBD |
| Few-shot (K=5) | 少量 | Ours + prototype | TBD | TBD |

## 创新点 3 鲁棒性曲线模板

| 扰动 | 强度范围 | 模型 | 备注 |
|------|----------|------|------|
| AWGN | SNR 30→5 dB | F (cross-day ckpt) | 主曲线 |
| CFO | 0→10 kHz | F + CNN | 对比 |
| Phase noise | 0→0.2 rad | F | |
| Narrowband | A 0→0.2 | F | f_i ≈ 0.8×BW |
| IQ imbalance | α 1.0→1.2 | F | β=1/α |
| Filter tilt | 0→8 dB | F | |

---

## 快速命令

```bash
# 复验 cross-receiver baseline（不重训）
bash experiments/cross_receiver_adaptation/verify_baseline.sh

# 完整重训 baseline（~12 runs × 80 ep，耗时较长）
GPU_ID=2 bash experiments/cross_receiver_adaptation/run_baseline.sh

# EM 鲁棒性曲线（全部 6 类扰动）
GPU_ID=3 bash experiments/em_robustness_openset/run_robustness_curves.sh

# 单条曲线 smoke test
python experiments/em_robustness_openset/eval_robustness_curves.py \
  --manifest data/paper/cross_day_day1to5_source_only.csv \
  --checkpoint outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt \
  --perturb-type awgn_snr_db \
  --strengths 30 20 10 \
  --out-csv experiments/em_robustness_openset/results/awgn_smoke.csv
```
