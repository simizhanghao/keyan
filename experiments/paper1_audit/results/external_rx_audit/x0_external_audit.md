# External multi-receiver X0 audit

verdict=OOB_OK training=false gpu=false official_test_signal_opened=false

## Archive inventory

| Archive | Entries | HDF5 | Compressed GB | Uncompressed GB |
|---|---:|---:|---:|---:|
| `multiple_receiver_test.zip` | 38 | 38 | 11.74 | 12.32 |
| `multiple_receiver_train.zip` | 14 | 14 | 13.98 | 14.68 |
| `receiver_drift_dataset.zip` | 10 | 10 | 3.99 | 4.20 |

## Official split from archive metadata

- source/train receivers (14): `b200_1, b200_mini_1, b210_1, n210_1, pluto_1, rtl_1, rtl_2, rtl_3, rtl_4, rtl_5, rtl_6, rtl_7, rtl_8, rtl_9`
- seen test receivers (14): `b200_1, b200_mini_1, b210_1, n210_1, pluto_1, rtl_1, rtl_2, rtl_3, rtl_4, rtl_5, rtl_6, rtl_7, rtl_8, rtl_9`
- blind test receivers (6): `b200_2, b200_mini_2, b210_2, n210_2, n210_3, pluto_2`
- expected 14/6 mapping matched: **True**
- location files: 18; drift HDF5: 10

The six blind receiver signal files were not extracted or opened. Archive filenames and sizes only were inspected.

## Source/train HDF5 samples

| RX | Type | Shape | DUTs | Packets | OOB/IB RMS dB | IB energy | OOB nonzero |
|---|---|---|---:|---:|---:|---:|---:|
| b200_1 | b200 | [8000, 16384] | 10 | 8000 | -28.55±0.76 | 99.02% | 100.00% |
| b200_mini_1 | b200_mini | [8000, 16384] | 10 | 8000 | -28.73±0.43 | 99.07% | 100.00% |
| b210_1 | b210 | [8000, 16384] | 10 | 8000 | -28.68±0.41 | 99.08% | 100.00% |
| n210_1 | n210 | [8000, 16384] | 10 | 8000 | -28.81±0.40 | 99.14% | 100.00% |
| pluto_1 | pluto | [8000, 16384] | 10 | 8000 | -28.93±0.37 | 99.11% | 100.00% |
| rtl_1 | rtl | [8000, 16384] | 10 | 8000 | -28.76±0.46 | 99.10% | 100.00% |

## Locked interpretation

- IQ reconstruction: first half real, second half imaginary (author loader convention).
- candidate physical mask was fixed before this audit: Fs=1000000 Hz, BW=125000 Hz, in-band `|f| <= BW/2`, OOB otherwise.
- HDF5 has no sampling-rate attribute; Fs/BW remain paper/code claims and must be cited as such.
- `OOB_OK` here means full complex IQ and nonzero full-band OOB across all sampled source receiver types; it does not claim stable transmitter identity.
- Official blind receiver signals remain sealed until X6.

