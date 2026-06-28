# 创新点 3：复杂电磁扰动 + 开放集认证

**分支：** `thesis-em-openset`

## 定位

- Paper 1：OOB RF-HSTU（same-receiver / cross-day）
- Paper 2：Cross-receiver 诊断 + RCPA-T
- **Chapter 5 / 创新点 3：EM 鲁棒 + open-set authentication**

## 快速开始

```bash
cd /data1/hcc/llm4RF
bash experiments/em_robustness_openset/run_em_smoke.sh
```

## 目录

| 脚本 | 作用 |
|------|------|
| `build_em_perturbation_benchmark.py` | 扰动配置 + 物理含义 audit |
| `eval_robustness_curves.py` | Closed-set EM 曲线 |
| `build_openset_splits.py` | 20 known + 4 unknown manifests |
| `eval_openset_auth.py` | MSP/Energy/Proto/Maha + AUROC/EER |
| `train_em_consistency.py` | EM-CR 训练（Phase C） |
| `run_em_smoke.sh` | Phase A smoke |
| `run_em_full.sh` | Phase B full curves |

## 文档

`docs/thesis_chapter5_em_openset/CHAPTER5_OUTLINE.md`

## 不修改

- `docs/iotj_paper/`
- `docs/paper2_rcpa/`
- `experiments/cross_receiver_calibration/` 冻结结果
