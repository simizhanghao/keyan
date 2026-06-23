# Experiment Protocol

This document defines the reproducible evaluation levels for the current LoRa RFFI project. The goal is to separate strict cross-day generalization, multi-source cross-day generalization, leave-one-day-out robustness, and statistical reliability checks.

## Current Optimization Route

The current main line is fixed as:

```text
CNN-stem + RF-HSTU + OOB-guided cross-attention + chirp embedding
```

The immediate goal is not to add new modules. The goal is to make the existing method stable, reproducible, and interpretable on the clean Day1-Day5 24-class manifest.

Use this order:

1. Center-loss sweep on Day1-Day4 -> Day5 with `center_loss_weight` in `{none, 0.001, 0.005, 0.01}`.
2. File-level aggregation diagnosis for CNN classifier/prototype and Hybrid classifier/prototype.
3. Bootstrap confidence intervals and paired CNN-vs-Hybrid file-level comparisons.
4. Leave-one-day-out Day1-Day5 cross-day evaluation with the selected center-loss weight.
5. Final multi-seed evaluation only for CNN-IQ and the selected Hybrid configuration.

Do not use SupCon v2, multi-scale token fusion, hard-margin loss, adversarial training, test-time adaptation, Qwen backbones, or synthetic IQ generation as part of the current main contribution. A spectrogram branch is a future upper-bound direction and should be developed separately only after the current route is stable.

Current scripts:

- `scripts/run_center_loss_sweep_day1to5.sh`
- `scripts/compare_file_aggregation_cnn_vs_hybrid.sh`
- `scripts/run_stat_day1to5.sh`
- `scripts/run_lodo_hybrid_vs_cnn.sh`
- `scripts/run_final_multiseed.sh`

## 1. Single-Source Cross-Day

Train on Day1 and test on Day2.

This is the strictest single-source cross-day setting. The model sees one source day during training and is evaluated on a different held-out day. It is useful for measuring direct day-to-day generalization and for comparison with the reproduced OSU-CNN-IQ baseline.

Recommended reporting:

- classifier `window_acc`
- classifier `macro_f1`
- classifier `file_acc`
- prototype `file_acc` with probability-based or confidence-weighted voting
- paired CNN-vs-Hybrid file-level comparison
- bootstrap confidence interval for file-level metrics

## 2. Multi-Source Cross-Day

Train on Day1, Day2, Day3, and Day4; test on Day5.

This setting evaluates whether multi-day training learns device fingerprints that are stable across day-level changes. It is less strict than Day1-to-Day2 single-source transfer, but it is closer to a practical training scenario where multiple enrollment days are available.

Recommended reporting:

- same cleaned 24-device label space across all days
- train split: Day1-Day4
- test split: Day5
- `window_acc`, `macro_f1`, and `file_acc`
- classifier and prototype results
- CNN baseline under the same manifest, windows, labels, and voting logic

## 3. Leave-One-Day-Out Cross-Day

Run five folds over Day1-Day5:

- train Day2-Day5, test Day1
- train Day1/Day3-Day5, test Day2
- train Day1/Day2/Day4/Day5, test Day3
- train Day1-Day3/Day5, test Day4
- train Day1-Day4, test Day5

Report mean and standard deviation across folds. This is the preferred paper-level cross-day protocol because it avoids drawing a conclusion from one specific held-out day.

Recommended reporting:

- mean +/- std for `window_acc`
- mean +/- std for `macro_f1`
- mean +/- std for `file_acc`
- per-fold table with the held-out day
- paired CNN-vs-Hybrid comparison per fold when possible

## 4. Statistical Reliability

File-level metrics are sensitive because the test split can contain only one file per device. For the 24-device cleaned setting, one file changes `file_acc` by `1/24 = 0.0417`.

Required reliability checks:

- bootstrap confidence interval for file-level accuracy
- paired CNN-vs-Hybrid comparison using matched test files
- evaluation window sensitivity, e.g. different deterministic `eval_samples_per_file`
- evaluation seed sensitivity when random evaluation windows are used
- multi-seed training results for final tables

Interpretation rule:

- Use `window_acc` and `macro_f1` as the primary stable metrics.
- Use `file_acc` as an authentication-style metric.
- Do not claim statistically significant file-level superiority unless the confidence interval or paired comparison supports it.

## 5. P1 Deployment-Shift Protocols (Config / Location / Distance)

These protocols use the OSU LoRa P1 setups documented in `docs/data_expansion_plan.md`. They are **not** replacements for the existing cross-receiver line. Their purpose is to test whether receiver-normalized, OOB-guided representations remain stable under additional deployment shifts (LoRa SF change, room/office/outdoor change, TX-RX distance change).

Shared rules for all P1 manifests:

- exclude raw `Device9`
- remap to 24 classes: `device=1..24`, `label=0..23`
- use the same windowing, voting, and CNN-vs-Hybrid comparison logic as cross-day experiments
- generate manifests with:
  - `scripts/generate_manifest_configs.py`
  - `scripts/generate_manifest_locations_distances.py`
- validate with `scripts/check_manifest.py`

Recommended primary metrics remain `window_acc` and `macro_f1`; report `file_acc` with bootstrap CI because each condition still has one file per device.

### A. Cross-Configuration (Diff_Configurations_Setup)

Manifests:

- `data/manifest_configs_all.csv`
- `data/manifest_configs_leave_one_config.csv`

SF mapping:

- Config1 = SF7
- Config2 = SF8
- Config3 = SF11
- Config4 = SF12

Primary protocol:

- train on Config1, Config2, and Config3
- test on Config4

Alternative protocol (preferred for paper-level reporting):

- leave-one-config-out over Config1-4
- each fold trains on 3 configs and tests on the held-out config
- report mean +/- std across 4 folds

Interpretation:

- this is the strictest non-receiver deployment shift in the OSU release
- compare Hybrid vs OSU-CNN-IQ under the same manifest and normalization (`oob_ratio` from cross-receiver confirmation)

### B. Cross-Location (Diff_Locations_Setup)

Manifests:

- `data/manifest_locations_all.csv`
- `data/manifest_locations_leave_one_location.csv`

Location mapping:

- Location1 = room
- Location2 = office
- Location3 = outdoor

Primary protocol:

- leave-one-location-out over Location1-3
- each fold trains on 2 locations and tests on the held-out location
- report mean +/- std across 3 folds

Interpretation:

- tests channel / geometry change at fixed SF7 and fixed indoor/outdoor collection day
- useful for checking whether OOB-guided attention reduces location-induced confusion without receiver gain confounding

### C. Cross-Distance (Diff_Distances_Setup)

Manifests:

- `data/manifest_distances_all.csv`
- `data/manifest_distances_leave_one_distance.csv`

Distance bins:

- 5m, 10m, 15m, 20m

Primary protocol:

- train on 5m, 10m, and 15m
- test on 20m

Alternative protocol:

- leave-one-distance-out over all 4 distance bins
- report mean +/- std across 4 folds

Interpretation:

- isolates path-loss / SNR change while keeping SF7 and indoor scene fixed
- use as a secondary axis after cross-config and cross-location

### D. Multi-Setup RF-MAE Pretraining + Downstream Fine-Tuning

Purpose:

- increase unlabeled IQ exposure before strict transfer evaluation
- test whether pretraining improves cross-day and cross-receiver fine-tuning, not just in-domain accuracy

Pretrain pool (train split only):

- Diff_Configurations: Config1-3
- Diff_Locations: Location1-2
- Diff_Distances: 5m, 10m, 15m
- Diff_Days Indoor: Day1-4 (`IQ_1` minimum; optional `IQ_2..10` later)

Held-out downstream tasks:

1. cross-day: Day1-4 train -> Day5 test, or LODO over Day1-5
2. cross-receiver: RX1->RX2 and RX2->RX1 on Indoor SameTx
3. optional P1 transfer checks: Config4, held-out location, 20m distance

Training order:

1. RF-MAE pretrain on the multi-setup pool (no labels required for reconstruction objective)
2. fine-tune classifier head on the downstream manifest train split
3. evaluate on the downstream manifest test/val split with frozen evaluation protocol

Do not mix P1 test splits into MAE pretraining. Keep pretrain manifests separate from downstream test files.

Reporting:

- pretrain data list and sample count
- downstream `window_acc`, `macro_f1`, `file_acc`
- paired CNN-vs-Hybrid comparison on each downstream task
- explicitly state that P1 results measure broader deployment robustness, not receiver normalization alone
