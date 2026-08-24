# 硕士毕设扩展路线（thesis-rffi-extension）

**题目（建议）：** 面向复杂电磁环境的 LoRa 设备射频指纹鲁棒认证方法研究

**分支：** `thesis-rffi-extension`（不影响 IoTJ 投稿分支 `iotj-submit-final-v1`）

---

## 三创新点与章节对应

| 创新点 | 论文章节 | 状态 | 代码/结果路径 |
|--------|----------|------|---------------|
| 1. OOB-guided cross-attentive RF-HSTU | 第 3 章 | **已完成**（IoTJ 主体） | `outputs/paper_ready_v3/step1_phase7_clean/` |
| 2. 跨接收机校准与适配 | 第 4 章 | 进行中 | `experiments/cross_receiver_adaptation/` |
| 3. 电磁扰动鲁棒性 + 开放集认证 | 第 5 章 | 进行中 | `experiments/em_robustness_openset/` |

---

## 研究主线（答辩叙事）

```text
LoRa 设备认证需求
  → 射频指纹物理层认证
  → CNN-IQ 跨日期不稳定
  → OOB-cross-attn RF-HSTU（创新点 1）
  → strict source-only cross-receiver 接近 chance（limitation）
  → 接收机校准 / TTA / few-shot（创新点 2）
  → 复杂电磁扰动 + 未知设备检测（创新点 3）
```

---

## 第一周任务（当前）

- [x] 创建 `thesis-rffi-extension` 分支
- [x] 整理 Chapter 3 IoTJ 结果
- [x] cross-receiver baseline 复验（Table IV 对齐）
- [x] EM 扰动脚本
- [x] **cross-receiver failure diagnosis pipeline**（优先于 TTA/few-shot）

## 第二周任务（诊断完成后）

- [ ] 基于诊断结论选择主方法：**RCPA few-shot + OOB equalization**
- [ ] 暂不盲跑 TTA，等 diagnosis report 确认 collapse 程度

---

## 目录结构

```text
experiments/
├── THESIS_ROADMAP.md              # 本文件
├── EXPERIMENT_SCHEDULE.md         # 周计划与实验表模板
├── thesis/
│   └── chapter3_iotj_results/     # 第 3 章结果整理
├── cross_receiver_adaptation/
│   ├── README.md
│   ├── run_baseline.sh            # 完整重训 baseline
│   ├── verify_baseline.sh         # 复验已有 Phase5 checkpoint
│   └── results/
└── em_robustness_openset/
    ├── README.md
    ├── eval_robustness_curves.py
    ├── run_robustness_curves.sh
    └── results/
```

---

## 关键约束（答辩时必须遵守）

**可以声称：**
- cross-attn OOB fusion 提升 same-receiver cross-day 鲁棒性
- cross-receiver 是 stress test，暴露 receiver calibration 问题
- few-shot / TTA 是合理部署路径（待实验验证）

**不能声称：**
- 跨接收机 invariant / 已解决 cross-receiver
- chirp embedding 是主要创新
- 任意 deployment shift 均 robust

详见 `outputs/paper_ready_v3/PAPER_RESULTS_SUMMARY.md` 第 7 节。
