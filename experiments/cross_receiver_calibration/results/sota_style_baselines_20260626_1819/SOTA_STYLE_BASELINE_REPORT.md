# SOTA-Style Same-Protocol Baseline Report

> Baseline runs: `experiments/cross_receiver_calibration/results/sota_style_baselines_20260626_1819`
> RCPA reference: `experiments/cross_receiver_calibration/results/full_20260626_1720` (frozen, not re-run)

## 1. RCPA-T vs linear probe / head fine-tuning

- **K=1 pooled file-acc:** RCPA-T 33.1 ± 11.2% | linear probe 33.3 ± 10.4% | head FT (source init) 30.3 ± 10.6% | head FT (random) 31.2 ± 8.5%
- **K=5 pooled file-acc:** RCPA-T 58.3 ± 9.0% | linear probe 59.0 ± 12.2% | head FT (source init) 53.9 ± 12.0% | head FT (random) 52.1 ± 13.1%
- **K=10 pooled file-acc:** RCPA-T 69.4 ± 9.7% | linear probe 65.7 ± 11.2% | head FT (source init) 63.9 ± 10.6% | head FT (random) 61.6 ± 10.1%

## 2. Unlabeled feature alignment vs K-shot RCPA

- **feat_mean_shift_source_classifier** pooled: 19.7 ± 8.1%
- **feat_mean_shift_source_prototype** pooled: 14.1 ± 4.8%
- **feat_coral_source_classifier** pooled: 25.0 ± 10.4%
- **feat_coral_source_prototype** pooled: 14.1 ± 4.8%
- **RCPA-T K=5** pooled: 58.3 ± 9.0%

## 3. TTA reference (RX1→RX2 quick)

- source_classifier: 16.7%
- entropy_min_tta: 20.8%
- pseudo_proto_tta: 8.3%
- RCPA-T_K1: 37.5%
- RCPA-T_K3: 33.3%
- RCPA-T_K5: 45.8%
- RCPA-T_K10: 58.3%

## 4. Answers

1. **RCPA-T vs K-shot baselines:** Linear probe (59.0%) exceeds RCPA-T (58.3%) at K=5; RCPA-T advantage is lightweight/non-parametric stability.
2. **Feature alignment:** Unlabeled CORAL/mean-shift remains far below K-shot RCPA-T; alignment alone does not replace labeled calibration.
3. **Same-protocol coverage:** linear probe + head FT + feature alignment + existing TTA/RCPA ablations constitute representative same-protocol comparison.
4. **Full SCRFFI / adversarial receiver-agnostic:** Not required now; different training/data protocol; discuss in related work only.
