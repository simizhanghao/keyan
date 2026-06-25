# Step1 Phase7-Clean 完整实验报告（paper_ready_v3）

> 生成时间：2026-06-25  
> Git commit：`b030d34`  
> 可直接转发给 GPT 做 Step2 决策与论文叙事

---

## 1. 实验目的

在 **统一 clean 协议** 下，对比 Cross-day（Day1–3 train / Day4 val / Day5 test）主结构消融，选定论文主方法（winner），为 Step2 recipe 消融与 deployment 实验定方向。

**本 Step 明确不做：** domain-robust full stack（focal / SWA / MixStyle / OOB-dropout 等），旧 Phase7 M7 数字已废弃。

---

## 2. 协议与超参（Step1 全 job 统一）

| 项 | 值 |
|---|---|
| Manifest | `data/paper/cross_day_day1to5_source_only.csv` |
| 划分 | train=Day1–3, val=Day4, test=Day5 |
| lr | 3e-3 |
| batch | 128 |
| epochs | 80 |
| oob_norm | **zscore**（Step1 不用 ratio） |
| loss | CE |
| checkpoint_metric | **val acc** |
| weight_decay | 5e-4 |
| label_smoothing | 0 |
| eval_samples_per_file | 256 |
| file vote | mean_logits |
| dim / depth | 64 / 2 |
| input_norm / fft_norm | iq_rms / log_zscore |

---

## 3. 模型矩阵（24 jobs）

| model_id | 架构 | seeds | n |
|---|---|---:|---:|
| A_cnn_iq | OSU CNN-IQ baseline | 0–4 | 5 |
| D_concat_oob_plain | CNN-stem + RF-HSTU + concat OOB，无 chirp | 0–4 | 5 |
| F_cross_attn_chirp_plain | CNN-stem + RF-HSTU + cross-attn OOB + chirp | 0–4 | 5 |
| H_gated_chirp_plain | CNN-stem + RF-HSTU + gated OOB + chirp | 0–4 | 5 |
| B_linear_no_oob | RF-HSTU linear patch，无 OOB（诊断） | 0–2 | 3 |
| C_cnn_stem_chirp_no_oob | CNN-stem + RF-HSTU + chirp，无 OOB（诊断） | 0 | 1 |

---

## 4. 运行状态

| 项 | 结果 |
|---|---|
| Jobs 成功 | **24/24** |
| Launcher | `All jobs finished successfully.` |
| failed_jobs | 无 |
| metrics.json | 24/24 |
| best.pt | 24/24 |
| file_predictions.csv | 24/24 |
| 训练日志 Traceback | 0 |
|  wall time | ~40 min（10:25–11:05，6×A100 GPU 1–6） |

**路径：**
- Runs: `outputs/paper_ready_v3/step1_phase7_clean/runs/`
- Outputs: `outputs/paper_ready_v3/step1_phase7_clean/outputs/`
- Logs: `outputs/paper_ready_v3/step1_phase7_clean/logs/train_jobs/`
- 汇总 CSV: `outputs/paper_ready_v3/step1_phase7_clean/metrics_all.csv`

---

## 5. 主表：多 seed 汇总（mean ± std）

| model_id | File-Acc | File-Macro-F1 | Window-Acc | Window-Macro-F1 |
|---|---:|---:|---:|---:|
| **F_cross_attn_chirp_plain** | **75.0 ± 5.3%** | **67.9 ± 6.8%** | 41.5 ± 2.4% | 39.4 ± 2.1% |
| B_linear_no_oob | 66.7 ± 3.4% | 59.1 ± 4.5% | 39.4 ± 13.7% | 36.8 ± 14.3% |
| A_cnn_iq | 54.2 ± 14.2% | 45.6 ± 14.8% | 43.5 ± 5.2% | 38.7 ± 5.7% |
| H_gated_chirp_plain | 19.2 ± 24.2% | 14.5 ± 23.3% | 13.5 ± 15.6% | 9.4 ± 15.9% |
| D_concat_oob_plain | 18.3 ± 18.6% | 12.3 ± 15.7% | 11.8 ± 8.7% | 8.4 ± 9.2% |
| C_cnn_stem_chirp_no_oob | 8.3% (n=1) | 1.8% (n=1) | 5.2% | 1.2% |

---

## 6. 逐 seed 明细（Day5 test）

### A_cnn_iq (OSU CNN)
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62.5% | 50.7% | 41.4% | 36.5% | 45 | 45.8% | 55.2% |
| 1 | 41.7% | 30.1% | 40.6% | 32.7% | 37 | 42.2% | 48.8% |
| 2 | 62.5% | 54.2% | 49.8% | 45.6% | 79 | 50.0% | 59.8% |
| 3 | 33.3% | 27.1% | 36.6% | 33.1% | 51 | 44.7% | 53.9% |
| 4 | 70.8% | 66.0% | 49.3% | 45.5% | 73 | 51.0% | 59.8% |

### B_linear_no_oob（诊断）
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62.5% | 54.4% | 23.2% | 19.9% | 63 | 23.4% | 33.7% |
| 1 | 70.8% | 65.3% | 56.7% | 55.0% | 76 | 55.1% | 66.2% |
| 2 | 66.7% | 57.6% | 38.3% | 35.6% | 74 | 37.8% | 50.7% |

### C_cnn_stem_chirp_no_oob（诊断，崩溃）
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **8.3%** | **1.8%** | 5.2% | 1.2% | 6 | 5.0% | 6.4% |

### D_concat_oob_plain
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.2% | 21.5% | 22.1% | 19.8% | 78 | 20.6% | 30.4% |
| 1 | **4.2%** | 0.3% | 5.0% | 1.3% | 29 | 5.0% | 7.1% |
| 2 | **4.2%** | 0.3% | 4.2% | 0.4% | 59 | 4.2% | 4.3% |
| 3 | 50.0% | 39.2% | 22.8% | 19.7% | 79 | 22.3% | 32.6% |
| 4 | **4.2%** | 0.3% | 4.7% | 1.0% | 12 | 4.6% | 5.5% |

### F_cross_attn_chirp_plain ⭐
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **83.3%** | **79.2%** | 38.7% | 36.9% | 67 | 36.1% | 50.6% |
| 1 | 70.8% | 62.5% | 42.0% | 39.9% | 58 | 39.6% | 53.5% |
| 2 | 70.8% | 61.8% | 40.2% | 38.1% | 76 | 39.0% | 52.4% |
| 3 | 79.2% | 72.2% | 40.8% | 38.9% | 80 | 39.4% | 53.4% |
| 4 | 70.8% | 63.9% | 45.8% | 43.1% | 80 | 44.0% | 56.9% |

### H_gated_chirp_plain
| seed | File-Acc | File-F1 | Win-Acc | Win-F1 | ckpt_ep | val_acc | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **4.2%** | 0.3% | 4.5% | 0.7% | 10 | 4.4% | 5.2% |
| 1 | 16.7% | 10.2% | 8.4% | 2.7% | 25 | 8.4% | 9.4% |
| 2 | **4.2%** | 0.3% | 5.1% | 0.9% | 5 | 5.4% | 6.2% |
| 3 | **4.2%** | 1.0% | 4.7% | 1.4% | 11 | 4.8% | 7.1% |
| 4 | 66.7% | 60.4% | 44.6% | 41.1% | 77 | 43.8% | 55.5% |

---

## 7. 核心对比：D vs F vs H（File-Acc 按 seed）

| seed | D_concat | F_cross_attn | H_gated |
|---:|---:|---:|---:|
| 0 | 29.2% | **83.3%** | 4.2% |
| 1 | 4.2% | **70.8%** | 16.7% |
| 2 | 4.2% | **70.8%** | 4.2% |
| 3 | 50.0% | **79.2%** | 4.2% |
| 4 | 4.2% | **70.8%** | 66.7% |

- **F**：5/5 seed ≥ 70.8%，std=5.3%，**唯一稳定强模型**
- **D**：3/5 seed 崩溃至 4.2%（≈1/24 随机猜），2 seed 尚可
- **H**：4/5 seed 崩溃，仅 seed4=66.7% 拉高均值

---

## 8. 与旧 Phase3 参考（seed=0，同协议口径）对比

| 模型 | Phase3 (旧) File-Acc / F1 | Step1-clean (新) File-Acc / F1 |
|---|---|---|
| D_concat | 87.5% / 69.5% | 29.2% / 21.5% |
| F_cross_attn | 79.2% / 59.6% | **83.3% / 79.2%** |
| C_no_oob | 4.2% / — | 8.3% / 1.8% |

**注意：** 新 Step1 与旧 Phase3 在 job generator、eval architecture args、seed 复现路径上已统一修复；D 在新 run 下方差极大，需 GPT 判断是 seed 敏感性还是旧 Phase3 D 数字口径不同。

---

## 9. 关键观察

1. **Winner = `F_cross_attn_chirp_plain`**（按 File-Macro-F1 与 File-Acc 均值及稳定性）
2. **OOB 对 hybrid 至关重要**：C（无 OOB）再次 ~8%，与旧 Phase3 一致
3. **Concat / Gated 在 cross-day 下方差极大**，不适合作为论文主方法；最多作消融/负例
4. **B_linear_no_oob** 意外强（66.7% mean），但无 OOB、无 CNN-stem，仅作诊断参考
5. **Val-test gap**：F 的 test File-Acc (70–83%) 显著高于 val acc (~36–44%)，说明 Day4 val 选 ckpt 仍有效但 val 绝对值偏低
6. **4.2% 崩溃模式** = 24 类中 ~1 类正确（1/24），典型训练崩溃/退化到近随机

---

## 10. 建议下一步（供 GPT 决策）

### Step2 winner 锁定
```text
winner = F_cross_attn_chirp_plain
```
Recipe 消融（3 seeds）：R0 CE+val_acc, R1 CE+val_macrof1, R2 classbalanced+val_macrof1, R3 focal+val_macrof1, R4 CE+val_acc+SWA

### 需排查（非阻塞 Step2）
- D / H 多 seed 崩溃原因（初始化？gated 实现？optimizer sensitivity？）
- C 无 OOB 崩溃是否为预期（论文叙事：OOB-aware hybrid 必要性）
- 旧 Phase3 D=87.5% vs 新 D seed0=29.2% 口径差异

### 论文主线建议
继续 **RF-HSTU + cross-attn OOB + chirp** 作为主方法；Phase4 deployment（Location/Distance）已有独立证据链。

---

## 11. artifacts 清单

- `outputs/paper_ready_v3/step0_audit/jobs_preview.tsv`
- `outputs/paper_ready_v3/step0_audit/jobs.tsv`
- `outputs/paper_ready_v3/step1_phase7_clean/metrics_all.csv`
- `outputs/paper_ready_v3/step1_phase7_clean/multiseed_summary.json`
- `logs/phase7_clean_20260625_102503.log`
