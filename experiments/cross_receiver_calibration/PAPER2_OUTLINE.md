# Paper 2 Outline (Draft)

**Working title:** Diagnosing and Mitigating Receiver-Induced OOB Feature Entanglement in LoRa RFFI

**Relationship to Paper 1 (IoTJ):** Uses frozen RF-HSTU backbone from Paper 1. Paper 1 = architecture + source-only same-receiver robustness. Paper 2 = cross-receiver failure diagnosis + post-hoc calibration. **Not a model paper.**

---

## 1. Introduction

- LoRa RFFI for IoT device authentication
- Cross-receiver shift breaks source-only models (~20% file-acc)
- Gap: prior work focuses on adversarial disentanglement / source-free pseudo-label; less on **OOB-specific entanglement diagnosis** under a strong frozen backbone
- Contributions (3 bullets, see below)

## 2. Related Work

- RFFI for IoT / LoRa
- Receiver-agnostic adversarial RFFI (TMC 2023)
- Source-free cross-receiver RFFI (SCRFFI / CSCNet)
- Feature disentanglement / domain adaptation
- Few-shot / prototype RFFI
- **Comparison table** → `RELATED_WORK_COMPARISON.md`

## 3. Cross-Receiver Failure Diagnosis

- OOB spectral profiling (RX2/RX1 energy ratio 1.44)
- Receiver / device linear probes (OOB path highest RX probe)
- Embedding distance ratios (CNN vs Ours fused)
- Prediction collapse (CNN top-1 mass 95.8%)
- **Conclusion:** OOB dual role — gain under fixed RX, entangled under cross-RX

## 4. Receiver-Calibrated Prototype Adaptation (RCPA)

- NOT a new backbone; frozen RF-HSTU
- Block-disjoint split: cal (A) / support (B) / query (C+D)
- K = labeled calibration **windows per device** (single file per device)
- RCPA-T (primary): target-receiver prototype, alpha=0
- RCPA-S, RCPA-B as ablations
- Leakage checks enforced

## 5. Experiments

### 5.1 Setup
- Diff_Receivers_Setup_Indoor_SameTx, 24 classes, RX1↔RX2
- Phase5-clean checkpoints, 3 seeds, 3 split repeats

### 5.2 Main results (frozen)
- Source classifier ~20%
- RCPA-T: K=5 58.3%, K=10 69.4%, K=20 75.0% (pooled)
- RCPA-B ablation: blend harmful

### 5.3 Supplementary
- OOB representation equalization (mechanism validation)
- TTA negative baseline (entropy-min 20.8%, pseudo-proto 8.3%)
- Optional: cross-day generality (if run)

## 6. Discussion and Limitations

- K-shot ≠ source-only; deployment mode differs from Paper 1
- Embedding-level OOB-Eq ≠ waveform equalization
- Single indoor setup; one file per device
- Comparison fairness vs source-free methods

## 7. Conclusion

---

## Three contributions (manuscript wording)

1. Diagnose receiver-induced OOB feature entanglement via OOB spectral profiling, receiver probes, embedding-distance analysis, and collapse analysis.

2. Show source classifier / source prototype non-transferable; target-receiver local prototypes restore separability with K labeled windows per device.

3. Propose RCPA — lightweight post-hoc calibration on frozen RF-HSTU — stable across directions, seeds, block-disjoint splits.
