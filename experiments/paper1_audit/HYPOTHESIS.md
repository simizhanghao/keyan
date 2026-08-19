# Experiment 1 — Paper 1 Evidence Audit: Pre-registered Hypotheses

**Status:** registered before any new training or OOB code change  
**Does not modify:** `outputs/paper_ready_v3/`  
**Does not start:** Paper 2 / RCOF / RAOF / canonicalizer / utility gate

## Question

Paper 1 观察到的 OOB cross-attention 收益，是真实、稳定的设备特征，还是预处理 artifact、协议偶然性、或 256-window file voting 造成的？

## Legacy frozen numbers (do not overwrite)

These remain the submitted IoTJ numbers. Audit results are reported separately.

| Model | File-Acc | Window-Acc | n seeds | Note |
|-------|---------:|-----------:|--------:|------|
| CNN-IQ `A_cnn_iq` | 54.2±14.2% | 43.5±5.2% | 5 | matched protocol |
| RF-HSTU linear no OOB `B_linear_no_oob` | 66.7±3.4% | 39.4±13.7% | 3 | **not** CNN-stem matched |
| Ours `F_cross_attn_chirp_plain` | 75.0±5.3% | 41.5±2.4% | 5 | legacy zscore OOB |
| CNN-stem no OOB `C_cnn_stem_chirp_no_oob` | 8.3% | 5.2% | 1 | collapsed; not in paper table |

## H1 — Protocol

Primary development protocol is Day1–3 train / Day4 val / Day5 untouched test.  
The CSV `data/paper/cross_day_day1to5_source_only.csv` already implements this, despite older docs saying Day1–4 train.

## H2 — OOB is not only FFT leakage / z-score artifact

After corrected z-score / OOB-only z-score / Hann / guard-band, device-level OOB structure remains on Day1–4 (`ρ_day < 1`).

## H3 — Cross-attention Full-OOB still beats matched Main-only

On Day4 selection, then frozen Day5: Full > Main, and Full > CNN, under identical CNN-stem + RF-HSTU + chirp / optimizer / epochs. The only changed variable is OOB cross-attention.

## H4 — File-Acc gain is cumulative evidence, not a K=256 spike

File accuracy of Full increases smoothly with K in {8,16,32,64,128,256}.

Day4 revision-reserve test (not Day5): first-K by `window_index` on frozen 1C CNN / Main / C'. Primary vote `mean_logits`. **Result: H4_PASS** (`file_vote_k.md`).

## H5 — LODO does not reverse the Day5 conclusion

After method freeze, most LODO folds remain Full > Main.

If H2 fails (RED) or H3 fails (YELLOW/RED), Experiment 2 RCOF is not opened.
