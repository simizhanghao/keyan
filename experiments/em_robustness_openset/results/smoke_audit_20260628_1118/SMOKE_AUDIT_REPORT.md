# Smoke Audit Report

**Date:** 2026-06-28T11:19:13
**Output:** `/data1/hcc/llm4RF/experiments/em_robustness_openset/results/smoke_audit_20260628_1118`

## Verdict

- Clean-equivalent spread (file-acc max-min): **0.0000** (0.0 pp)
- Per-file pred disagree (no_perturb vs cfo_norm_0): **0** files
- Audit passed: **YES**

## Answers

1. **Same checkpoint?** Yes — single load: `outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt`
2. **Same windows/file?** Yes — `32` windows, seed `0`, deterministic offsets (see window_sampling_audit.csv)
3. **Same Day5 test manifest?** Yes — `data/paper/cross_day_day1to5_source_only.csv`, split=`test`, `24` files
4. **Same file-level voting?** Yes — `mean_logits`
5. **Perturbation disabled = clean?** Zero-strength CFO/phase/IQ/filter now skip apply; AWGN uses SNR>=100 as clean
6. **AWGN power?** Target 30 dB, measured mean 32.99 dB (I/Q split fixed)
7. **Normalization?** `input_norm=iq_rms` in dataset; perturb after norm

## Key clarification

- **Clean baseline (mean over clean-equivalent):** 83.3%
- **AWGN 30 dB (perturbed, NOT clean):** 66.7%

Prior smoke compared AWGN **30 dB** to CFO **clean (0.0)** — different conditions.
After audit, compare clean-equivalent rows only for baseline consistency.

## Clean-equivalent results

| Condition | File-Acc | Note |
|-----------|----------|------|
| no_perturb | 83.3% | skip apply_em_perturbation |
| awgn_snr_inf | 83.3% | SNR>=100 dB, no noise |
| cfo_norm_0 | 83.3% | CFO norm=0 skipped |
| phase_noise_0 | 83.3% | sigma=0 skipped |
| iq_amp0_phase0 | 83.3% | IQ imbalance 0 skipped |
| filter_tilt_0 | 83.3% | tilt=0 skipped |
| awgn_30db_ref | 66.7% | NOT clean — reference only |