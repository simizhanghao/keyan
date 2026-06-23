# Literature Notes

These notes summarize the literature position used by the current LoRa RFFI experiments. They are intentionally conservative: they separate prior work claims from the current project's actual novelty and avoid over-claiming file-level significance.

## 1. OSU LoRa RFFI and Deployment Variability

The OSU LoRa RFFI dataset studies RF fingerprinting for 25 LoRa devices under deployment variability. The relevant shifts include cross-day, cross-location, cross-configuration, and cross-receiver conditions. These shifts are central to the problem because a device fingerprint must remain recognizable when the channel, receiver, location, configuration, or collection day changes.

The original OSU LoRa RFFI work already discusses out-of-band (OOB) spectrum distortion and hardware impairments as useful device-specific evidence. Therefore, this project should not claim that it is the first work to use OOB information for LoRa RF fingerprinting.

The accurate novelty claim is narrower and stronger: this project uses OOB-guided token-level cross-attention. IQ-derived OOB spectral distortion is represented as an auxiliary token sequence, and the main RF tokens dynamically attend to those OOB tokens. The contribution is not the existence of OOB evidence, but the way OOB evidence is injected into a chirp-aware RF token model.

## 2. Spectrogram / Time-Frequency LoRa RFFI

LoRa uses chirp spread spectrum modulation, so its time-frequency structure is physically meaningful. Existing LoRa RFFI work has used spectrogram-style representations to capture fine-grained time-frequency characteristics of LoRa signals.

A spectrogram branch is therefore a reasonable future direction for improving the upper bound of the current system. It could provide a more physically aligned representation of chirp trajectories and local time-frequency distortions than the current IQ-derived token views alone.

The current multi-scale mean-merge attention should not be described as a physically meaningful multi-scale RF representation. It merges neighboring tokens by averaging and can smooth away local RF fingerprint cues. For this reason, multi-scale token fusion is currently marked as experimental and is not part of the main model line.

## 3. Metric Learning and Channel-Robust RFFI

Channel-robust LoRa RFFI literature includes deep metric learning and RF fingerprint extractor ideas. This supports the use of prototype evaluation, center loss, and embedding compactness objectives for cross-domain RFFI.

Prototype evaluation relies on the assumption that each device has a compact and stable embedding cluster. If embeddings have large intra-class dispersion across days or channels, prototype voting can be unstable even when the classifier head has good window-level accuracy.

In the current experiments, SupCon v2 did not provide reliable gains, so it is not used as the main line. The most practical embedding constraint so far is center loss, because it directly penalizes within-class spread and does not depend as strongly on having enough same-class samples inside each mini-batch. This makes it more directly aligned with prototype stability and macro-F1.

## 4. Test-Time Adaptation and Domain Shift

Day1-to-Day2 evaluation is a domain-shift setting. The target day can differ in signal statistics, channel conditions, and receiver-side distribution. Test-time adaptation methods such as BN Adapt and TENT are useful diagnostic tools for checking whether target-domain statistics shift explains part of the performance gap.

Current TENT experiments only bring small improvements. This suggests that BatchNorm/statistics shift exists, but it is not the only bottleneck. The remaining errors are likely also tied to device confusion, embedding dispersion, limited file-level samples, and cross-day changes that cannot be fixed by simple test-time statistics adaptation.

Future work should not put the main effort into TENT unless leave-one-day-out experiments show that some held-out days exhibit much stronger target-domain statistics shift.

## 5. Statistical Reliability

The strict Day1-to-Day2 test setting has only 24 test files after removing the missing device. One file changes file-level accuracy by `1/24 = 0.0417`. File-level accuracy should therefore not be used alone as a statistically strong conclusion.

The required report should include:

- `window_acc`
- `macro_f1`
- `file_acc`
- bootstrap confidence interval
- paired CNN-vs-Hybrid comparison

`window_acc` and `macro_f1` are the more stable primary metrics in the current setup because they are computed over many deterministic windows. `file_acc` remains important as an authentication-style metric, but it must be interpreted with confidence intervals or paired comparison because the number of files is small.
