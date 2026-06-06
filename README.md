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

## Defaults

- IQ window: `8192`
- Patch size: `256`
- Number of patches: `32`
- In-band: center `125 kHz` LoRa bandwidth inside `1 MHz` sample rate
- OOB: frequencies outside the LoRa bandwidth but inside the sampled bandwidth
- RF-HSTU block:
  1. Linear projection to `Q,K,V,U`
  2. `score = QK^T / sqrt(d)`
  3. Add relative patch bias
  4. `A = sigmoid(score)`
  5. `AV = A @ V`
  6. `out = Linear(LayerNorm(AV) * SiLU(U))`
  7. Residual + layer norm
