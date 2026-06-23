# Transmitter Query Bottleneck Attention — Design (Not Implemented)

Status: **design only**. Implement after CORAL+IM 30ep sweep shows partial success.

## Motivation

Current Hybrid pools all RF-HSTU tokens with mean-pooling → **receiver style and
transmitter content share one vector**. OOB-ratio normalization helps, but the
main/IQ path still encodes receiver gain and spectral tilt.

NLP analog: compress long context into **few query latents** (Perceiver, Q-Former)
so the classifier sees a bottleneck, not raw styled tokens.

## Architecture sketch

```
IQ window
  └─ RFPatchEmbedder (unchanged: cnn_stem + oob_ratio views)
       ├─ main_tokens  [B, P, D]   from IQ+FFT+amp via CNN-stem
       ├─ oob_tokens   [B, P, D]   from OOB-ratio patches (cross-attn path)
       └─ cfo_token    [B, 1, D]   projected peak_offset (+ optional centroid)

Token bank T = concat(main_tokens, oob_tokens, cfo_token)  →  [B, P+P+1, D]

Learnable transmitter queries Q ∈ R^{K × D}  (K=4 or 8, << P)

  Attn_out = CrossAttn(Q, T, T)   # queries attend to all tokens
  z = mean(Attn_out) or flatten   →  [B, D] or [B, K·D]

Classifier(z) → 24 classes
```

### Key design choices

| Choice | Recommendation | Rationale |
|---|---|---|
| K (num queries) | 4 or 8 | 24 train files → keep params small |
| Query init | Xavier + small noise | Avoid symmetric collapse |
| OOB in bank | Yes (ratio-normalized) | Hardware cue, now receiver-stable |
| CFO in bank | 1 projected token | Keeps peak_offset in attention mix |
| Mean-pool fallback | Ablation only | Baseline comparison |
| RF-HSTU blocks | Keep depth=2 on **main_tokens only** before bank | Limit overfit |

### What changes vs current Hybrid

| Component | Current | Proposed |
|---|---|---|
| Pooling | mean over RF-HSTU output tokens | K query cross-attn bottleneck |
| OOB fusion | cross-attn into main tokens pre-pool | OOB tokens also in bank |
| CFO | concat after pool | optional token in bank (+ concat ablation) |
| Classifier input | D (+ CFO dims) | D or K·D |

## Training (unchanged constraints)

- Normalization: `input_norm=iq_rms`, `fft_norm=log_zscore`, `oob_norm=ratio`
- No center loss / SupCon / hard margin / multi-scale
- Optional: stack CORAL+IM on bottleneck embedding `z`
- 30ep quick → 80ep confirm if promising

## Evaluation protocol

Same as cross-receiver: RX1→RX2 and RX2→RX1, classifier mean_logits primary.

Success vs current D2 (oob_ratio+CFO source-only avg win/f1 = 0.232 / 0.189):

1. Bidirectional avg window ≥ 0.30 or macro_f1 ≥ 0.23
2. RX2→RX1 no longer loses badly to CNN iq_rms
3. RX1→RX2 does not regress below D0 oob_ratio_only

## Minimal code plan (future PR)

1. **`src/rfhstu/query_bottleneck.py`**
   - `TransmitterQueryBottleneck(nn.Module)`: queries + cross-attn + norm
2. **`models.py`**
   - `DeviceClassifier(..., use_query_bottleneck=False, num_queries=4)`
   - If enabled: RF-HSTU → query bottleneck → classifier (CFO token in bank)
3. **`train_utils.py`**
   - `--use-query-bottleneck`, `--num-transmitter-queries`
4. **`scripts/run_cross_receiver_query_bottleneck_sweep.sh`**
   - D2 baseline vs D2+query_bottleneck vs D2+query_bottleneck+coral_im
5. **Tests**
   - Forward shape smoke: `[B,2,8192] → logits [B,24]`
   - Backward compat: flag off → identical to current Hybrid

## Risks

- Query collapse (all queries attend same token) → monitor attn entropy
- K too large for 24-file training → start K=4
- Double-counting CFO (token + concat) → ablate one path

## Not in scope

- Dual-stream disentangled attention (separate doc / P2)
- Qwen / LLM / teacher forcing
- Pseudo-label adaptation
