# Cross-Receiver Failure Diagnosis

**目标：** 在开发任何 cross-receiver 适配方法之前，先回答「为什么 OOB-cross-attn RF-HSTU 在 strict source-only cross-receiver 下失败」。

## 诊断实验清单

| # | 实验 | 脚本 | 输出 |
|---|------|------|------|
| 1 | Embedding 提取 | `extract_embeddings.py` | `file_embeddings.npz` |
| 2 | 距离比诊断 | `analyze_distances.py` | `distance_summary.csv` |
| 3 | Receiver/Device probe | `train_probes.py` | probe accuracy CSV |
| 4 | UMAP/t-SNE | `plot_embeddings.py` | PNG/PDF |
| 5 | Main/OOB/Fused 对比 | `analyze_path_ablation.py` | path ablation CSV |
| 6 | OOB 频谱 profile | `analyze_oob_spectrum.py` | spectrum figures |
| 7 | Confusion matrix | `plot_confusion.py` | collapse_summary.csv |
| 8 | 诊断报告 | `generate_report.py` | `CROSS_RECEIVER_DIAGNOSIS_REPORT.md` |

## 一键运行

```bash
cd /data1/hcc/llm4RF
GPU_ID=1 bash experiments/cross_receiver_diagnosis/run_diagnosis.sh
```

## 数据与模型

- Manifest: `data/manifest_rx1_to_rx2.csv`（RX1 + RX2 全部 24 设备）
- Checkpoints: Phase5-clean source-only（RX1 训练）
  - CNN: `outputs/paper_ready_v3/phase5_clean_cross_receiver/runs/A_cnn_iq/rx1_to_rx2/seed_0/best.pt`
  - Ours: `outputs/paper_ready_v3/phase5_clean_cross_receiver/runs/F_cross_attn_chirp_plain/rx1_to_rx2/seed_0/best.pt`

## 注意

- 大型 `.npz` 文件在 `results/` 下，已 gitignore，不提交。
- 只提交脚本、CSV summary、小图、报告。

## 第二篇论文主线

```text
Receiver-induced feature entanglement diagnosis
  → OOB evidence receiver-entangled under shift
  → lightweight receiver calibration (RCPA)
```
