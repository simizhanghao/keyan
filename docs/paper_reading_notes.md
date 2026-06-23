# Paper Reading Notes

This file records the downloaded paper set and project-specific notes for the current LoRa RFFI study. The notes focus on deployment shift, OOB distortion and hardware impairments, spectrogram/time-frequency representations, metric learning/prototypes/center loss, and test-time adaptation.

## Download Status

Nine of ten requested PDFs were downloaded and text-extracted under `docs/papers/`. The IEEE Access PDF for the OSU "LoRa Device Fingerprinting in the Wild" paper could not be downloaded from IEEE Xplore in this environment because IEEE returned HTTP 418 to both DOI and stamp PDF endpoints. Official metadata was captured from the DOI/HBKU/ResearchGate pages, and the PDF should be added manually when available.

## 1. LoRa Device Fingerprinting in the Wild: Disclosing RF Data-Driven Fingerprint Sensitivity to Deployment Variability

- Authors: Abdurrahman Elmaghbub; Bechir Hamdaoui
- Venue/year: IEEE Access, 2021
- DOI/arXiv: DOI `10.1109/ACCESS.2021.3121606`
- PDF path: not downloaded locally; IEEE Xplore blocked command-line download with HTTP 418. Official metadata page: `https://elmi.hbku.edu.qa/en/publications/lora-device-fingerprinting-in-the-wild-disclosing-rf-data-driven/`

Project-relevant notes:

- This is the key OSU LoRa RFFI deployment-variability paper and the proper dataset citation for the 25-device LoRa setup.
- It studies deployment changes including time/day, distance, location, LoRa protocol configuration, and receiver hardware.
- It already proposes exploiting OOB spectrum distortion caused by hardware impairments. Therefore, the current project must not claim to be the first to use OOB information.
- The correct novelty boundary for this project is OOB-guided token-level cross-attention: OOB evidence is represented as auxiliary tokens and dynamically injected into main RF tokens.
- The paper reports that cross-setting evaluation can degrade strongly, especially under LoRa configuration and receiver changes. This supports treating cross-day/cross-receiver/cross-configuration as domain-shift protocols rather than ordinary random splits.

## 2. Deep-Learning-Based Device Fingerprinting for Increased LoRa-IoT Security: Sensitivity to Network Deployment Changes

- Authors: Bechir Hamdaoui; Abdurrahman Elmaghbub
- Venue/year: IEEE Network, 2022; arXiv version 2022
- DOI/arXiv: arXiv `2208.14964`; DOI `10.1109/MNET.001.2100553`
- PDF path: `docs/papers/02_deep_learning_lora_iot_deployment_changes.pdf`

Project-relevant notes:

- This article reinforces the OSU deployment-shift framing: same-setting training/testing can work well, while different settings expose sensitivity.
- It explicitly links OOB distortion information to hardware impairments and uses it to improve LoRa fingerprinting accuracy.
- It reports moderate sensitivity to channel condition changes and severe sensitivity to protocol configuration and receiver hardware changes when IQ data is used.
- It reports poor cross-setting FFT behavior, which warns against treating frequency-domain features as automatically more robust.
- For this project, it supports the need for strict cross-day and cross-receiver protocols, but it also limits novelty claims around OOB usage.

## 3. Radio Frequency Fingerprint Identification for LoRa Using Spectrogram and CNN

- Authors: Guanxiong Shen; Junqing Zhang; Alan Marshall; Linning Peng; Xianbin Wang
- Venue/year: IEEE INFOCOM, 2021; arXiv version 2020/2021
- DOI/arXiv: arXiv `2101.01668`
- PDF path: `docs/papers/03_lora_spectrogram_cnn.pdf`

Project-relevant notes:

- This paper motivates spectrogram/time-frequency representation for LoRa because LoRa is chirp spread spectrum and carries fine-grained time-frequency structure.
- It shows spectrogram-CNN outperforming IQ- and FFT-based alternatives in its own 20-device setup, which makes a spectrogram branch a credible future upper-bound direction.
- It identifies instantaneous CFO drift as a stability problem and uses CFO compensation / hybrid classification to mitigate misclassification.
- This supports the project's chirp-aware positioning: LoRa structure matters, and token positions should respect chirp-scale organization.
- Current mean-merge multi-scale token fusion should not be equated with physical time-frequency multi-scale modeling, because it can average away local fingerprint cues.

## 4. Towards Channel-Robust Radio Frequency Fingerprint Identification Using Contrastive Learning

- Authors: Jie Ma; Junqing Zhang; Guanxiong Shen; Linning Peng; Alan Marshall
- Venue/year: IEEE WCNC, 2025
- DOI/arXiv: not identified in local extraction
- PDF path: `docs/papers/04_channel_robust_lora_contrastive_learning.pdf`

Project-relevant notes:

- This paper frames channel variation as a core RFFI bottleneck and uses spectrogram input plus contrastive learning to improve channel robustness.
- It evaluates public and self-collected LoRa datasets under stationary, mobile, LOS, and NLOS scenarios.
- It treats hardware impairments such as IQ imbalance, CFO, and power amplifier nonlinearity as transmitter-originated fingerprint evidence.
- It supports metric-learning-style reasoning for LoRa RFFI, but the current project's SupCon v2 result was weak, so this paper should be cited as motivation rather than as proof that SupCon must be the main line.
- Its reported high accuracies should not be used as a direct OSU Day1-to-Day2 comparison because datasets, splits, signal representation, and evaluation protocols differ.

## 5. Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for LoRa

- Authors: Guanxiong Shen; Junqing Zhang; Alan Marshall; Joseph Cavallaro
- Venue/year: IEEE Transactions on Information Forensics and Security, 2022; arXiv version 2021
- DOI/arXiv: arXiv `2107.02867`
- PDF path: `docs/papers/05_scalable_channel_robust_lora.pdf`

Project-relevant notes:

- This paper proposes a deep metric learning based RFF extractor for scalable LoRa RFFI, supporting enrollment of previously unseen devices.
- It directly supports prototype/database-style evaluation: a learned extractor can map packets into an embedding space where device identities are stored and compared.
- It treats wireless channel impact as a major issue and mitigates it through channel-independent features and data augmentation.
- It uses 60 commercial LoRa devices, which makes it useful for the scalability motivation in Related Work.
- This should be cited as metric-learning and scalable-RFFI context, not as a direct baseline for the OSU 24/25-device cross-day split.

## 6. Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification

- Authors: Jie Ma; Junqing Zhang; Guanxiong Shen; Linning Peng; Alan Marshall
- Venue/year: arXiv preprint, 2025; presented in part at IEEE WCNC 2025 according to the manuscript
- DOI/arXiv: arXiv `2512.12070`
- PDF path: `docs/papers/06_channel_robust_receiver_independent_rffi.pdf`

Project-relevant notes:

- This paper targets both channel robustness and receiver independence, which aligns with the long-term goal beyond Day1-to-Day2 testing.
- It uses spectrogram representation to decouple transmitter impairments from channel and receiver effects.
- It proposes a three-stage pipeline: contrastive pretraining, Siamese classification network training, and inference.
- It supports the idea that receiver effects are a separate domain-shift factor, not merely another channel condition.
- Its high reported performance should be treated as evidence for possible future directions, not as a direct comparison to the OSU LoRa Day1-Day5 protocol.

## 7. Tent: Fully Test-Time Adaptation by Entropy Minimization

- Authors: Dequan Wang; Evan Shelhamer; Shaoteng Liu; Bruno Olshausen; Trevor Darrell
- Venue/year: ICLR, 2021
- DOI/arXiv/OpenReview: OpenReview `uXl3bZLkr3c`
- PDF path: `docs/papers/07_tent.pdf`

Project-relevant notes:

- TENT adapts at test time using only target data and model parameters, without source data or labels.
- It minimizes prediction entropy while estimating normalization statistics and updating channel-wise affine parameters.
- In this project, TENT/BN Adapt should be interpreted as diagnostics for target-day statistics shift.
- Current local TENT experiments only produced small gains, suggesting BN/statistics shift exists but is not the only bottleneck.
- TENT should not become the main method unless leave-one-day-out results show strong day-specific target statistics shift.

## 8. TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation

- Authors: Hyesu Lim; Byeonggeun Kim; Jaegul Choo; Sungha Choi
- Venue/year: ICLR, 2023; arXiv version 2023
- DOI/arXiv: arXiv `2302.05155`
- PDF path: `docs/papers/08_ttn.pdf`

Project-relevant notes:

- TTN studies the tradeoff between conventional BN using source running statistics and transductive BN using current test-batch statistics.
- It argues that pure test-batch normalization can fail under small or non-i.i.d. test batches, which is relevant to LoRa evaluation where files/windows may be grouped by device/day.
- TTN interpolates BN statistics according to each layer's domain-shift sensitivity.
- For this project, TTN supports careful interpretation of BN Adapt/TENT: test-time normalization can help, but it is sensitive to batch composition.
- This reinforces the need to report evaluation batch/window protocol when using TTA.

## 9. A Discriminative Feature Learning Approach for Deep Face Recognition

- Authors: Yandong Wen; Kaipeng Zhang; Zhifeng Li; Yu Qiao
- Venue/year: ECCV, 2016
- DOI/arXiv: DOI `10.1007/978-3-319-46478-7_31`
- PDF path: `docs/papers/09_center_loss.pdf`

Project-relevant notes:

- This is the center loss paper: it learns a center for each class and penalizes distances between features and their class centers.
- The motivation is to reduce intra-class variation while softmax provides inter-class separation.
- This maps well to prototype-based LoRa RFFI because prototype voting depends on compact and stable per-device embedding clusters.
- Center loss is currently a better fit than SupCon v2 for this project because it directly targets intra-class dispersion and does not depend as strongly on same-class samples in each mini-batch.
- It should be cited as a general discriminative feature learning method, not as an RFFI-specific contribution.

## 10. Supervised Contrastive Learning

- Authors: Prannay Khosla; Piotr Teterwak; Chen Wang; Aaron Sarna; Yonglong Tian; Phillip Isola; Aaron Maschinot; Ce Liu; Dilip Krishnan
- Venue/year: NeurIPS, 2020
- DOI/arXiv: arXiv `2004.11362`
- PDF path: `docs/papers/10_supervised_contrastive_learning.pdf`

Project-relevant notes:

- SupCon extends contrastive learning to labeled data by pulling same-class samples together and pushing different-class samples apart.
- It provides theoretical and empirical motivation for embedding compactness and class separation.
- In LoRa RFFI, SupCon is relevant to prototype stability and cross-domain embedding robustness.
- However, the project's current SupCon v2 ablation did not outperform the main line, so SupCon should remain a related-work and ablation reference rather than the default method.
- SupCon performance depends on batch composition, positive-pair availability, augmentations, and temperature, all of which are nontrivial in per-device RF datasets.

## Project-Level Implications

- OOB evidence is prior art in OSU LoRa RFFI; the current novelty is dynamic token-level OOB-guided cross-attention.
- Spectrogram/time-frequency modeling is well motivated for LoRa and should be the next serious upper-bound direction, but it is not part of the current main line.
- Metric learning and prototype evaluation are strongly supported by scalable/channel-robust RFFI literature, but the current best local regularizer is center loss rather than SupCon v2.
- TTA methods are useful for diagnosing day/domain statistics shift, but current local results do not justify making TTA the main contribution.
- DeepCRF-style or contrastive-learning papers with very high accuracies should not be used as direct comparisons to the OSU LoRa Day1-to-Day2 or Day1-to-Day5 protocols. They are reference points for decision fusion, metric learning, and representation design only.
