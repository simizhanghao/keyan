# OSU LoRa RF Fingerprinting Dataset

Official source:

- https://research.engr.oregonstate.edu/hamdaoui/datasets
- https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/

This project follows the RF-HSTU plan and does not download the full 1.2TB dataset at once.

## Download Order

### 0. Smoke subset

Purpose: verify URLs, file naming, IQ loading, patch slicing, and metadata parsing.

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm4ee\scripts\download_osu_lora.ps1 -Phase smoke
```

Expected content:

- `Diff_Days_Indoor_Setup`: Day1/Day2, Device1/Device2, `IQ_1`
- `Diff_Receivers_Setup`: RX1/RX2, Device1/Device2

### 1. Cross-day first batch

Purpose: first RF-HSTU experiment, Day1 train to Day2 test.

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm4ee\scripts\download_osu_lora.ps1 -Phase days_iq1
```

Expected content:

- Setup 1, Different Days Indoor
- Day1/Day2
- all 25 devices
- first IQ recording per device

Approximate scale: about 7.7GB for IQ `.dat` files plus metadata.

### 1b. Remaining indoor days (Day3-Day5)

Purpose: expand cross-day experiments beyond Day1/Day2.

On Linux:

```bash
bash scripts/download_osu_lora_days345.sh
```

Expected content:

- Setup 1, Different Days Indoor
- Day3/Day4/Day5
- all 25 devices
- first IQ recording per device (`IQ_1.dat` + `IQ_1.sigmf-meta`)
- also retries missing `Day2/Device9/IQ_1` if absent

Approximate scale: about 12GB for IQ `.dat` files plus metadata.

Generate a Day1-Day5 manifest after download:

```bash
python3 scripts/generate_manifest_days.py --days 3,4,5 --split extra
```

Official download base URL:

- https://research.engr.oregonstate.edu/hamdaoui/datasets
- https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup/

### 2. Cross-receiver batch

Purpose: RF-HSTU main robustness experiment, Receiver1 train to Receiver2 test.

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm4ee\scripts\download_osu_lora.ps1 -Phase receivers_iq
```

Expected content:

- Setup 7, Different Receivers
- RX1/RX2
- all 25 devices

### 3. Cross-configuration batch

Purpose: cross LoRa configuration / SF robustness experiment.

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm4ee\scripts\download_osu_lora.ps1 -Phase configs_iq
```

Expected content:

- Setup 5, Different Configurations
- Config1-Config4
- all 25 devices

## Notes

- The script writes a phase manifest to this directory before downloading.
- Downloads use `curl.exe --continue-at -`, so partial files can resume.
- Prefer IQ first. FFT can be added later for CNN-FFT baselines or ablation.
