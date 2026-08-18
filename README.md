# new_phase — 从 0 到最终主线的提纯包

本目录是可复现快照：保留原仓库相对路径，从数据协议 → 核心模型 → 训练评测 → 三章终局实验，一条线走完。不是全仓库备份。

基线：`thesis-em-openset` / `f9dbe1c`  
主模型：`F_cross_attn_chirp_plain` = CNN-stem + RF-HSTU + OOB cross-attn + chirp  
运行时把本目录当作仓库根：`export PYTHONPATH=/data1/hcc/llm4RF/new_phase/src:$PYTHONPATH`

原始 IQ `.dat` **不拷入**（体积大）。manifest 指向原 `data/raw/`；评测时数据仍读原路径。

---

## 从 0 到终局（按顺序读/跑）

```text
0  数据协议     data/paper/*.csv + scripts/check_manifest.py
               scripts/paper/generate_paper_manifests.py
1  特征/模型    src/rfhstu/{data,features,cnn_stem,oob_fusion,models,cnn_baseline}.py
2  训练/评测    scripts/finetune.py  scripts/evaluate.py
3  第3章        scripts/paper/  +  outputs/paper_ready_v3/
               同 RX 跨日：75.0±5.3% vs CNN 54.2±14.2%
4  第4章        experiments/cross_receiver_diagnosis/
               experiments/cross_receiver_calibration/
               诊断纠缠 → RCPA-T K=20 → 75.0%
5  第5章        experiments/em_robustness_openset/
               src/rfhstu/em_perturbations.py
               Clean 83.3%；CFO 失效；Proto AUROC 0.917
```

协议文档：`docs/experiment_protocol.md`、`docs/paper_experiment_pipeline.md`、`docs/thesis/THESIS_EXPERIMENT_MAP.md`。  
第 3 / 4 / 5 章协议数字禁止混用。

---

## 目录对照

| 路径 | 阶段 | 说明 |
|------|------|------|
| `src/rfhstu/` | 1 | 核心库（含 CNN 基线、OOB 融合、EM 扰动） |
| `scripts/finetune.py` `evaluate.py` | 2 | 统一训练/评测入口 |
| `scripts/check_manifest.py` `generate_manifest_*.py` | 0 | manifest 生成与检查 |
| `scripts/paper/` | 3 | IoTJ / 跨日 / 部署 / 跨 RX stress |
| `scripts/bootstrap_eval_ci.py` `paired_compare_models.py` | 3 | 小样本 file_acc 统计 |
| `data/paper/` | 0 | source_only 主协议；`*_oracle_*` 仅诊断，禁入主表 |
| `experiments/cross_receiver_diagnosis/` | 4 | 探针 / 谱 / 几何 |
| `experiments/cross_receiver_calibration/` | 4 | RCPA-T + 同协议 baseline |
| `experiments/em_robustness_openset/` | 5 | EM + open-set（含 EM-CR 负面） |
| `outputs/paper_ready_v3/` | 3 | 终局表与报告，无权重 |

---

## 明确不收录

- `data/raw/**/*.dat`、`*.pt` / `*.pth` / `*.npy` embedding、`logs/` / `nohup`
- `shiyaner/`（RAOF 未收口）
- 旧 Day1→Day2 / `*.ps1`、SupCon / multiscale / TTA 主线脚本
- oracle target-val 结果树（manifest 留下作反例）
- EM-CR full training（只留 smoke/debug 负面）

---

## 终局数字（验收用）

| 章 | 指标 | 值 |
|----|------|-----|
| 3 | File-Acc Ours vs CNN | 75.0±5.3% vs 54.2±14.2% |
| 4 | RCPA-T K=5/10/20 | 58.3 / 69.4 / 75.0% |
| 5 | Clean / CFO 0.003 / Proto AUROC | 83.3% / ~4.2% / 0.917 |

写论文只引用本包内 CSV/MD；改数字必须回源实验重跑。
