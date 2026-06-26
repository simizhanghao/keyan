# Related Work Comparison (Paper 2)

| Direction | Representative work | Training | Target labels | Backbone change | Key idea | vs. RCPA (ours) |
|-----------|---------------------|----------|---------------|-----------------|----------|-----------------|
| Receiver-agnostic RFFI | Shen et al., TMC 2023 | Multi-receiver adversarial | Multi-RX train | Retrain NN | Learn RX-independent features | We freeze backbone; diagnose OOB entanglement; K-window target calibration |
| Source-free cross-RX | SCRFFI / CSCNet (Sensors 2025) | Source pre-train | Target unlabeled only | Adapt head | Contrastive + pseudo-label on target | We show pseudo/TTA insufficient; minimal **labeled** windows more reliable |
| Feature disentanglement | Cross-RX generalization (arXiv 2025) | Adversarial + disentangle | Varies | Retrain | Disentangle device vs receiver factors | Post-hoc calibration + OOB mechanism diagnosis, not new disentangle net |
| Federated / collaborative RFFI | TIFS 2024 federated contrastive | Multi-client | Unlabeled across clients | Federated train | Collaborative representation | Single-receiver deploy + local K-shot cal, lighter |
| Few-shot / prototype RFFI | Prototypical / Gaussian proto | Meta or standard | Support set | Varies | Class prototype distance | **Diagnosis-first** + LoRa single-file K-window protocol + block-disjoint split |
| Cross-day UDA (IoTJ 2026) | LoRa/WiSig SEI UDA | Source day labeled | Target day unlabeled | Retrain/adapt | Adversarial + pseudo + contrastive | Different shift (day vs receiver); we target RX mismatch with labeled cal |
| **RCPA (ours)** | This work | Source RX only (frozen RF-HSTU) | **K labeled windows/device on target RX** | **None (post-hoc)** | OOB entanglement diagnosis + target prototype calibration | Diagnosis-first; honest TTA negative; OOB-Eq mechanism validation |

## Safe novelty claims

- ✅ First systematic OOB entanglement diagnosis for LoRa cross-RX under OOB cross-attention backbone
- ✅ Block-disjoint K-window calibration protocol for single-file-per-device LoRa captures
- ✅ Strong empirical result with frozen backbone (58–75% vs ~20% classifier)
- ❌ NOT "first cross-receiver RFFI"
- ❌ NOT "new prototype algorithm"
- ❌ NOT "new RF-HSTU architecture"
