# 跨接收机适配实验（创新点 2）

## 目标

在 IoTJ 已暴露的 **strict source-only cross-receiver limitation** 基础上，研究：

1. receiver-aware normalization（零目标标签）
2. test-time entropy adaptation（无标签目标流量）
3. few-shot prototype calibration（每设备少量校准样本）

## 协议

- Manifest：`data/paper/rx1_to_rx2_source_only.csv`，`rx2_to_rx1_source_only.csv`
- Val：源接收机；Test：目标接收机
- Hybrid OOB norm：**ratio**（与 Phase5-clean 一致）
- Chance：4.17% file-acc（24 类）

## 脚本

| 脚本 | 用途 |
|------|------|
| `verify_baseline.sh` | 复验 Phase5 checkpoint，不重训 |
| `run_baseline.sh` | 完整重训 CNN + Hybrid（3 seeds × 2 directions） |

## 已有 baseline（IoTJ Table IV）

| Direction | CNN-IQ | Ours |
|-----------|--------|------|
| RX1→RX2 | 4.2±0.0% | 18.1±3.9% |
| RX2→RX1 | 23.6±15.3% | 15.3±7.1% |

## 下一步

- [ ] TTA：`evaluate.py --adapt-mode entropy_min`
- [ ] Few-shot：target RX 每类 K 样本更新 prototype
- [ ] CORAL/MMD：参考 `scripts/paper/phase5_cross_receiver.sh`

## 输出目录

```text
experiments/cross_receiver_adaptation/results/
├── verify_baseline_YYYYMMDD/
└── baseline_YYYYMMDD/
```
