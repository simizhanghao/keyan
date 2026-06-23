# DI-RF-HSTU

First implementation pass for domain-invariant generative RF sequence modeling on the OSU LoRa RF fingerprinting dataset.

The project reads SigMF `cf32` IQ `.dat` files directly with `numpy.memmap`, samples 8192-point windows, splits them into 256-point patches, derives FFT and OOB features from IQ, and trains:

- `pretrain_mae.py`: masked RF modeling pretraining.
- `finetune.py`: device classification with optional supervised contrastive and domain-adversarial losses.
- `evaluate.py`: classifier or prototype-head evaluation.

## Install

```powershell
cd D:\llm4RF
D:\miniconda\envs\llm4ee\python.exe -m pip install -r requirements.txt
```

## Quick Smoke Run

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\pretrain_mae.py --manifest data\raw\osu_lora\manifest_all.csv --max-files 4 --samples-per-file 8 --epochs 1 --batch-size 4 --device cpu

D:\miniconda\envs\llm4ee\python.exe scripts\finetune.py --manifest data\raw\osu_lora\manifest_all.csv --max-files 8 --samples-per-file 8 --epochs 1 --batch-size 4 --device cpu --use-contrastive --use-adversarial

D:\miniconda\envs\llm4ee\python.exe scripts\evaluate.py --manifest data\raw\osu_lora\manifest_all.csv --checkpoint runs\finetune\best.pt --samples-per-file 8 --device cpu
```

The default `python` in this workspace did not have PyTorch installed during initial verification. The local `llm4ee` conda environment had PyTorch `2.12.0+cu126` and was used for smoke tests.

## Current Takeaways

The current main line is not "Transformer replacing CNN." It is CNN-enhanced RF-HSTU for LoRa RF fingerprinting.

Current best main model:

```text
CNN-stem + RF-HSTU + OOB-guided cross-attention + chirp embedding
```

Recommended training configuration for the main line:

- `label_smoothing=0.05`
- `weight_decay=5e-4`
- `center_loss_weight=0.01`

Method interpretation:

- CNN-stem provides local RF waveform impairment modeling.
- OOB cross-attention dynamically uses out-of-band hardware distortion evidence.
- Chirp embedding aligns the token representation with LoRa chirp structure.
- Center loss reduces intra-class embedding dispersion and is directly aligned with prototype stability and macro-F1.

Modules that are not part of the current main line:

- SupCon v2
- multi-scale token fusion
- hard-margin loss
- score fusion

Current Day1-to-Day2 conclusion:

- The hybrid improves window-level accuracy and macro-F1 over the reproduced OSU-CNN-IQ baseline.
- File-level accuracy has a positive point-estimate trend, but the Day2 test set has only 24 files after cleanup. Therefore, do not claim statistically significant file-level superiority without bootstrap confidence intervals or paired comparison.

Next evaluation stage:

- Day1-Day5 leave-one-day-out cross-day evaluation.
- Spectrogram branch as an optional upper-bound direction, not as part of the current main line.

Paper-level optimization route:

1. Run a fixed-architecture center-loss sweep on the clean Day1-Day5 24-class manifest:
   `./scripts/run_center_loss_sweep_day1to5.sh`
2. Diagnose file-level aggregation without retraining:
   `OUT_ROOT=outputs/<day1to5_run> ./scripts/compare_file_aggregation_cnn_vs_hybrid.sh`
3. Report bootstrap confidence intervals and paired file-level comparisons:
   `OUT_ROOT=outputs/<day1to5_run> ./scripts/run_stat_day1to5.sh`
4. After the best center-loss weight is selected, run Day1-Day5 leave-one-day-out:
   `CENTER_LOSS_WEIGHT=<best> ./scripts/run_lodo_hybrid_vs_cnn.sh`
5. Only after the final candidate is selected, run multi-seed evaluation:
   `CENTER_LOSS_WEIGHT=<best> ./scripts/run_final_multiseed.sh`

The current main contribution does not use SupCon v2, multi-scale token fusion, TTA, hard-margin loss, adversarial training, Qwen backbones, or synthetic IQ generation.

See also:

- `docs/literature_notes.md`
- `docs/experiment_protocol.md`
- `docs/method_positioning.md`

For the fixed project smoke test, run:

```powershell
.\scripts\smoke_test.ps1
```

For OOB fusion mode smoke tests, run:

```powershell
.\scripts\smoke_test_oob.ps1
```

For multi-scale token fusion smoke tests, run:

```powershell
.\scripts\smoke_test_multiscale.ps1
```

## Manifest Check

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\check_manifest.py --manifest data\manifest_all.csv
```

To make sure large raw IQ files were not accidentally committed:

```powershell
git status --short
git ls-files data/raw | Select-Object -First 20
```

## Formal Evaluation

`evaluate.py` writes reproducible result artifacts:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\evaluate.py --manifest data\manifest_all.csv --checkpoint runs\finetune\best.pt --mode classifier --out-dir outputs\eval\my_run --device cuda
```

Output files:

- `metrics.json`: window/file accuracy, macro F1, run mode, model flags, checkpoint and manifest paths.
- `predictions.csv`: one row per evaluated window with file path, window index, label, prediction, correctness, confidence, split/setup/domain fields.
- `confusion_matrix.csv`: row=true label, column=predicted label.
- `per_device_accuracy.csv`: window and file accuracy per device label.
- `per_domain_accuracy.csv`: window accuracy grouped by setup, split, day, receiver, config, location, distance.
- `run_config.json`: evaluation arguments and checkpoint training arguments.

File-level voting averages logits or prototype similarities over windows from the same `.dat` file before `argmax`.

## Sanity Checks

```powershell
.\scripts\overfit_tiny.ps1
.\scripts\gpu_sanity.ps1
```

`overfit_tiny.ps1` trains on a very small subset for 50 epochs and warns if final train accuracy is below 0.9.
`gpu_sanity.ps1` prints CUDA information, runs a 3-epoch GPU finetune, and evaluates classifier/prototype modes.

## Ablation

Current recommended formal model, based on the latest cross-day ablation, is:

```powershell
--oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding
```

The key finding so far is that OOB-guided cross-attention is the most effective module. `--use-multiscale` is currently experimental and should not be used as the default formal configuration. `full_light` is retained for ablation only, not as the current main model.

For the current Day1 to Day2 setting, prefer evaluating with the same or a larger number of windows per file than training. Use `--eval-samples-per-file` to control this explicitly. For example, `--eval-samples-per-file 128` evaluates 24 Day2 files with 3072 windows, while `--eval-samples-per-file 256` evaluates 6144 windows.

Current main-model candidates:

- `cross_attn_oob_no_chirp`: simpler and tends to give steadier file-level behavior.
- `chirp_cross_attn_oob`: often improves window accuracy and macro F1, but file-level accuracy should be checked carefully.

Current finding after comparing against the reproduced OSU CNN baseline:

- OSU-CNN-IQ is a strong baseline because convolution captures local RF waveform distortions.
- OOB-guided cross-attention improves RF-HSTU over simple OOB concatenation and improves classifier macro-F1, but CNN-IQ remains a strong local-feature baseline.
- RF-HSTU does not yet comprehensively outperform CNN-IQ prototype overall.
- Therefore, the next model direction is CNN-stem + RF-HSTU + OOB-guided cross-attention.

Latest hybrid finding:

- CNN-stem alone is not sufficient.
- OOB cross-attention improves dynamic OOB fusion.
- Chirp embedding becomes effective when combined with CNN-stem and OOB cross-attention.
- The current main model direction is CNN-stem + OOB-guided cross-attention + chirp embedding + RF-HSTU.
- This hybrid improves window accuracy and macro F1 over OSU-CNN-IQ, but file accuracy still needs voting and aggregation analysis.

Current recommended main model:

```powershell
--patch-embed-type cnn_stem `
--cnn-stem-dim 32 `
--oob-fusion-type cross_attn_oob `
--use-oob-cross-attention `
--use-chirp-embedding
```

Current best Day1 to Day2 route:

- This is not "Transformer replaces CNN". The current direction is CNN-enhanced RF-HSTU.
- CNN remains a strong local RF waveform baseline, but the hybrid improves strict Day1 to Day2 window-level accuracy, macro F1, and classifier file-level accuracy.
- The current best main line is CNN-stem + RF-HSTU + OOB-guided cross-attention + chirp embedding, trained with `label_smoothing=0.05`, `weight_decay=5e-4`, `samples_per_file=256`, and `eval_samples_per_file=256`.
- SupCon v2, multi-scale token fusion, score fusion, and hard-margin loss are not the current main line.
- Do not claim file-level authentication is fully solved. The correct statement is that window-level and macro-F1 improve over OSU-CNN-IQ, while file-level behavior is close to or slightly better than the CNN prototype and still needs stronger aggregation or adaptation.

Use classifier evaluation for the primary window-level metrics:

- `window_acc`
- `macro_f1`

Use prototype evaluation with probability-based file voting for file-level authentication:

```powershell
--mode prototype --file-vote-mode mean_prob
```

or:

```powershell
--mode prototype --file-vote-mode confidence_weighted
```

The correct comparison statement is: the hybrid improves `window_acc` and `macro_f1` over OSU-CNN-IQ, while file-level accuracy is currently tied with OSU-CNN-IQ prototype after probability-based voting. Do not claim that file accuracy comprehensively exceeds CNN-IQ yet.

The hybrid front-end is enabled with:

```powershell
--patch-embed-type cnn_stem --cnn-stem-dim 32
```

In this mode, the main IQ-derived channels `[IQ, FFT magnitude, amplitude/phase]` are first processed by a lightweight Conv1d stem, then tokenized with `Conv1d(kernel_size=patch_size, stride=patch_size)` into RF-HSTU patch tokens. OOB tokens remain on the existing OOB patch path; when `--oob-fusion-type cross_attn_oob --use-oob-cross-attention` is used, the CNN-stem main tokens attend to the OOB tokens with the same lightweight OOB-guided cross-attention module.

Run the CNN-stem hybrid SPF256 comparison:

```powershell
.\scripts\run_cross_day_hybrid_cnnstem_spf256.ps1
```

This runs:

- `rfhstu_linear_cross_attn`
- `rfhstu_cnnstem_concat`
- `rfhstu_cnnstem_cross_attn`
- `rfhstu_cnnstem_cross_attn_chirp`

Evaluate file-level voting modes for the current best hybrid checkpoint without retraining:

```powershell
.\scripts\eval_vote_modes_hybrid_spf256.ps1
```

Supported file voting modes:

- `mean_logits`: average file-window logits or prototype similarities before `argmax`. This is the default for backward compatibility.
- `mean_prob`: apply softmax per window, then average probabilities.
- `confidence_weighted`: apply softmax per window, use each window's max probability as confidence, and compute a confidence-weighted file probability.

Compare OSU-CNN-IQ classifier predictions against the hybrid classifier:

```powershell
.\scripts\compare_cnn_iq_vs_hybrid.ps1
```

This writes:

- `outputs\analysis_cnn_vs_hybrid\per_device_delta.csv`
- `outputs\analysis_cnn_vs_hybrid\a_correct_b_wrong.csv`
- `outputs\analysis_cnn_vs_hybrid\b_correct_a_wrong.csv`
- `outputs\analysis_cnn_vs_hybrid\both_wrong.csv`
- `outputs\analysis_cnn_vs_hybrid\both_correct.csv`

Analyze repeated error devices and confusion targets:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\analyze_error_devices.py `
  --pred-a outputs\osu_cnn_day1_day2_spf256\osu_cnn_iq\classifier\predictions.csv `
  --pred-b outputs\cross_day_hybrid_cnnstem_spf256\rfhstu_cnnstem_cross_attn_chirp\classifier\predictions.csv `
  --out-dir outputs\error_device_analysis
```

This writes:

- `outputs\error_device_analysis\confusion_targets_by_device.csv`
- `outputs\error_device_analysis\both_wrong_by_device.csv`
- `outputs\error_device_analysis\cnn_wrong_hybrid_right_by_device.csv`
- `outputs\error_device_analysis\hybrid_wrong_cnn_right_by_device.csv`

Run lightweight regularization experiments for the current best hybrid structure:

```powershell
.\scripts\run_hybrid_regularization_spf256.ps1
```

This script keeps the model structure fixed and tests only label smoothing, weight decay, and dropout variants. It does not enable multi-scale, supervised contrastive learning, or domain-adversarial training.

Evaluate test-time adaptation for the current best hybrid checkpoint without retraining:

```powershell
.\scripts\eval_tta_hybrid_best_spf256.ps1
```

Supported TTA modes in `evaluate.py`:

- `--tta-mode none`: standard evaluation.
- `--tta-mode bn_adapt`: all modules stay in eval mode except BatchNorm layers, which use the current evaluation batch statistics. Dropout remains disabled and no parameters are updated.
- `--tta-mode tent`: freezes the model and updates only BatchNorm affine parameters by entropy minimization on unlabeled evaluation batches.

TENT controls:

```powershell
--tent-steps 1
--tent-lr 1e-4
--tent-episodic
```

Run Center Loss experiments for the current best hybrid structure:

```powershell
.\scripts\run_hybrid_center_loss_spf256.ps1
```

For the current Day1-Day5 clean 24-class protocol, use the Linux runner:

```bash
GPU_ID=1 ./scripts/run_center_loss_sweep_day1to5.sh
```

Center Loss is optional and disabled by default. Enable it with:

```powershell
--use-center-loss --center-loss-weight 0.001
```

It penalizes within-class embedding spread using the pooled classifier embedding before the final classifier. It should not be enabled together with SupCon or hard-margin loss in the first version.

## Statistical Reliability Under Small Day2 Evaluation Set

The strict Day1 to Day2 validation split currently has only 24 Day2 files. A file-level accuracy change of one file is therefore `1/24 = 0.0417`, so values such as `0.6250`, `0.6667`, and `0.7083` can differ by only one or two files. Interpret file-level accuracy with confidence intervals and paired comparisons.

Use `window_acc` and `macro_f1` as primary stability metrics. Use `file_acc` as an authentication-style metric, but report bootstrap confidence intervals and paired model comparisons when possible.

Bootstrap file-level confidence intervals from a prediction CSV:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\bootstrap_eval_ci.py `
  --predictions outputs\hybrid_center_loss_spf256\center_w001\classifier\predictions.csv `
  --out outputs\stat_analysis\center_w001_classifier\bootstrap_ci.csv `
  --n-bootstrap 1000
```

Evaluate sensitivity to the number of deterministic evaluation windows per file:

```powershell
.\scripts\eval_window_sensitivity_hybrid.ps1
```

Evaluate sensitivity to random evaluation window offsets:

```powershell
.\scripts\eval_seed_sensitivity_hybrid.ps1
```

Run a paired file-level comparison between OSU-CNN-IQ and the current best hybrid:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\paired_compare_models.py
```

Run the focused long-epoch OOB main comparison:

```powershell
.\scripts\run_cross_day_oob_main.ps1
```

Run the SPF128 comparison with 128 evaluation windows per file:

```powershell
.\scripts\run_cross_day_oob_main_spf128_eval128.ps1
```

Run the SPF256 comparison with 256 training and evaluation windows per file:

```powershell
.\scripts\run_cross_day_oob_main_spf256.ps1
```

## Reproduced OSU CNN Baseline

This is a PyTorch reproduction of the OSU CNN baseline following the architecture and preprocessing described in the paper:

Abdurrahman Elmaghbub and Bechir Hamdaoui, "LoRa Device Fingerprinting in the Wild: Disclosing RF Data-Driven Fingerprint Sensitivity to Deployment Variability", IEEE Access, 2021.

The reproduced baseline uses the same project dataset and evaluation path as RF-HSTU:

- `window_size=8192`
- non-overlapping deterministic evaluation windows
- unit-energy IQ window normalization from the current `SigMFIQDataset`
- CNN classifier with `Conv1d + BatchNorm1d + LeakyReLU + MaxPool1d`
- final average pooling, FC hidden layer with 25 neurons, dropout `0.5`, and a final classifier layer
- the same Day1 to Day2 manifest, `samples_per_file`, `eval_samples_per_file`, metrics, and file-level voting logic

The CNN baseline is selected with:

```powershell
--model-type osu_cnn
```

CNN input representations:

- `--cnn-input-type iq`: raw normalized IQ window as `[I, Q]`.
- `--cnn-input-type fft_inband`: FFT real/imag with only `|f| <= 62.5 kHz` retained for LoRa BW `125 kHz`.
- `--cnn-input-type fft_oob`: full `1 MHz` FFT real/imag derived online from IQ, including in-band and adjacent OOB spectrum.
- `--cnn-input-type fft`: alias-style full FFT real/imag path.
- `--cnn-input-type amp_phase`: amplitude/phase derived from normalized IQ.

The `fft_oob` input is derived online from the IQ `.dat` files in this repository. It is an in-band plus OOB FFT representation for fair comparison, not a downloaded original FFT file from the OSU MATLAB release.

Run the Day1 to Day2 SPF256 CNN baseline set:

```powershell
.\scripts\run_osu_cnn_baseline_day1_day2_spf256.ps1
```

This runs:

- `osu_cnn_iq`
- `osu_cnn_fft_inband`
- `osu_cnn_fft_oob`
- `osu_cnn_amp_phase`

Each experiment runs finetune, classifier evaluation, prototype evaluation, and writes results under:

```text
runs\osu_cnn_day1_day2_spf256\<experiment>
outputs\osu_cnn_day1_day2_spf256\<experiment>\classifier
outputs\osu_cnn_day1_day2_spf256\<experiment>\prototype
```

Compare OSU CNN baselines with the current RF-HSTU SPF256 main results:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\summarize_compare_osu_cnn_vs_rfhstu.py `
  --cnn-root outputs\osu_cnn_day1_day2_spf256 `
  --rfhstu-root outputs\cross_day_oob_main_spf256 `
  --out outputs\compare_osu_cnn_vs_rfhstu_spf256.csv
```

The comparison CSV includes `method_family`, `experiment`, `window_acc`, `file_acc`, `macro_f1`, `num_windows`, and `num_files`.

Run the lightweight cross-day ablation set:

```powershell
.\scripts\run_ablation_cross_day.ps1
```

Summarize all `metrics.json` files:

```powershell
D:\miniconda\envs\llm4ee\python.exe scripts\summarize_results.py --root outputs\ablation_cross_day
```

The summary is written to `outputs\ablation_cross_day\summary.csv`.

## Defaults

- IQ window: `8192`
- Patch size: `256`
- Number of patches: `32`
- In-band: center `125 kHz` LoRa bandwidth inside `1 MHz` sample rate
- OOB: frequencies outside the LoRa bandwidth but inside the sampled bandwidth
- Optional LoRa chirp embedding: disabled by default. Enable with `--use-chirp-embedding`.
  With the defaults `sample_rate=1 MHz`, `lora_bandwidth=125 kHz`, `spreading_factor=7`, and `patch_size=256`,
  `samples_per_chirp = Fs * 2^SF / BW = 1024`, so each chirp spans 4 patches and an 8192-sample window spans about 8 chirps.
  When enabled, the encoder input becomes `x + patch_pos_emb + chirp_id_emb + patch_in_chirp_emb`.
- OOB fusion modes:
  - `no_oob`: only main IQ/FFT/amplitude-phase patch tokens are used.
  - `concat_oob`: OOB patch features are concatenated with main patch features. This is the default and preserves the original behavior.
  - `cross_attn_oob`: main tokens attend to OOB tokens with a lightweight single-layer cross-attention module, followed by gated residual fusion. Enable it with both `--oob-fusion-type cross_attn_oob` and `--use-oob-cross-attention`.
  Example:
  ```powershell
  D:\miniconda\envs\llm4ee\python.exe scripts\finetune.py --manifest data\manifest_all.csv --oob-fusion-type cross_attn_oob --use-oob-cross-attention --device cuda
  ```
- Multi-scale token fusion: disabled by default. Enable with `--use-multiscale --multiscale-ratios 1,2,4 --multiscale-fusion-type concat_pool`.
  This is a lightweight token-level version:
  - ratio `1` keeps the original 32 patch tokens.
  - ratio `2` mean-merges every 2 adjacent tokens into 16 coarse tokens.
  - ratio `4` mean-merges every 4 adjacent tokens into 8 coarse tokens.
  - each scale is encoded separately, mean pooled, concatenated, and projected back to `dim`.
  This does not implement raw patch sizes 128/256/512, frequency-band-specific multi-scale OOB, multi-layer OOB cross-attention, or hierarchical attention.
  Example:
  ```powershell
  D:\miniconda\envs\llm4ee\python.exe scripts\finetune.py --manifest data\manifest_all.csv --oob-fusion-type concat_oob --use-multiscale --multiscale-ratios 1,2,4 --multiscale-fusion-type concat_pool --device cuda
  ```
- RF-HSTU block:
  1. Linear projection to `Q,K,V,U`
  2. `score = QK^T / sqrt(d)`
  3. Add relative patch bias
  4. `A = sigmoid(score)`
  5. `AV = A @ V`
  6. `out = Linear(LayerNorm(AV) * SiLU(U))`
  7. Residual + layer norm
