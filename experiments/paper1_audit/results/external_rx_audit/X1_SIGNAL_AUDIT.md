# X1 signal-level OOB audit

**Status:** source/train + receiver-drift only; no classifier training; official six blind receivers unopened.

## Protocol

- Fixed physical mask from X0: `Fs=1 MHz`, LoRa `BW=125 kHz`, in-band `|f| <= BW/2`, OOB is the complement.
- Feature: `20 log10((RMS(OOB)+eps)/(RMS(IB)+eps))` per packet.
- Data: 14 official source/train receivers plus the two non-blind receiver-drift receivers (`n210_1`, `rtl_6`), days 1--4 where available.
- HDF5 rows have device labels but no capture/session identifiers. Results are descriptive packet-level summaries, not independent-capture inference.

## Result

The combined run contains 24 HDF5 files and 144,000 packets. The standard deviation of group means is:

| Group | Std. of group means (dB) |
|---|---:|
| Device | 0.393 |
| Receiver type | 0.111 |
| Receiver instance | 0.083 |
| Day | 0.015 |

Receiver effects are therefore measurably larger than the observed same-receiver day effect, while device separation is larger than receiver separation. This is **not** evidence that OOB is receiver-only or that receiver variance must exceed device variance. The supported interpretation is:

> OOB relative magnitude contains transmitter-discriminative variation and a systematic receiver-dependent component; the receiver component is larger than ordinary drift in this audit.

The six official blind receiver signals remain sealed. This result does not authorize classifier training, F0 retuning, or opening X6.

Machine-readable output: `x1_source_drift_all_signal.json`.
