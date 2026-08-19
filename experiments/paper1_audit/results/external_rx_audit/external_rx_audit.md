# External multi-receiver LoRa audit (Zhang / TMC)

verdict=LOCAL_ABSENT  oob=not_applicable  training=false  gpu=false
OSU 2-RX F0 remains RX_FAIL. Do not retune.

## Paper claims (not yet file-verified)

- page: https://junqing-zhang.github.io/dataset-code/
- dataport: https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers
- 10 DUT, 20 SDR, SF7, BW 125 kHz, fs 1 MHz claimed

## Local hits

- none under /data1/hcc, /data1/datasets, /data1/data
- Human must download the three DataPort zips, then re-run this script

LOCAL_ABSENT → download then re-audit. Do not train.
OOB_OK later → new multi-RX DG protocol, not F0 retune on OSU.
