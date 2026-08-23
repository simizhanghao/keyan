# Phase 2C — External multi-receiver dataset audit (pre-registered)

**Status:** X0 **OOB_OK** on six sampled source receiver types. Official 14-source/6-blind split matched; blind signals sealed. No train.  
**Does not reopen:** F0 retune, F1, S1, `a` range, Day5, C1/C_fft/D, factorization, utility, oracle.

This audit **must not** change any frozen F0 / RX gate.

---

## Why this beat

OSU Indoor SameTx has only two receivers. F0 killed synthetic OOB-scale on Day4 but did **not** beat C' on real RX1↔RX2 (pooled window Δ −0.52 pp).

That closes “identity-first scale pairing on this backbone + OSU 2-RX” as a Paper 2 method.  
It does **not** close the mechanism story (OOB-scale shortcut is still real on Day4).

A second dataset with **many** receivers is the only protocol-clean pivot. Do not add modules to rescue F0 on two SDRs.

---

## Target corpus

Official:

- Page: https://junqing-zhang.github.io/dataset-code/
- IEEE DataPort: https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers
- DOI: 10.21227/d6vx-r538
- Paper: Shen, Zhang et al., TMC 2024, “Towards Receiver-Agnostic and Collaborative RFFI”

Capture facts from the paper/code, with file verification noted separately:

| Item | Paper claim |
| --- | --- |
| DUTs | 10 LoRa (5× LoPy4 + 5× mbed SX1261) |
| Receivers | 20 SDR, 6 models |
| LoRa BW | 125 kHz, SF7, 868.1 MHz |
| RX sample rate | **1 MHz** |
| Representation in their code | preamble extract + channel-independent spectrogram |
| Zips | `multiple_receiver_train.zip` ~13 GB, `test` ~11 GB, `receiver_drift` ~3.7 GB |

X0 verified the source/train HDF5 representation as 8000 packets by 16384 float64 values, decoded as 8192 real samples plus 8192 imaginary samples. All six sampled source receiver types retain finite, nonzero OOB bins under the pre-existing 1 MHz / 125 kHz physical mask. HDF5 contains no sampling-rate attribute, so Fs/BW remain paper/code facts rather than file metadata.

---

## This beat (no train, no GPU)

```text
script   experiments/paper1_audit/scripts/audit_external_x0.py
input    /data1/hcc/llm4RF/data0820/*.zip + source-only audit samples
report   archive / split / HDF5 schema / OOB verdict
forbid   official blind signal access; training; changing F0/RX gates
```

| Verdict | Meaning |
| --- | --- |
| `LOCAL_ABSENT` | not on disk; Human must download; stop |
| `OOB_UNKNOWN` | files exist but format not enough to judge OOB |
| `OOB_OK` | raw-ish IQ at ≥1 MHz, spectrum not obviously in-band-only |
| `OOB_INSUFFICIENT` | spectrogram / baseband-cropped / no complex IQ → do not adapt F0 |

`OOB_INSUFFICIENT` ⇒ do not rewrite our method to fit the dataset.  
`OOB_OK` ⇒ later Human GO for a **new** multi-receiver DG protocol (not a retune of F0 on OSU).

**Recorded:**

- `results/external_rx_audit/x0_external_audit.md` → **OOB_OK**
- `results/external_rx_audit/OFFICIAL_SPLIT_AUDIT.md` → 14 source + 6 blind matched
- `results/external_rx_audit/X0_GROUPING_AUDIT.md` → no capture/session IDs; audited exact-row overlaps = 0
- `results/external_rx_audit/archive_sha256.txt` → three archive hashes

The six blind receivers are `b200_2`, `b200_mini_2`, `b210_2`, `n210_2`, `n210_3`, and `pluto_2`. Their HDF5 signals were not extracted or opened. They remain sealed until a later X6 Human GO after the source-only mechanism and F0/F0-CT pilot are frozen.

Downloaded archives currently live here (do not copy them into `matched_seed0`):

```text
/data1/hcc/llm4RF/data0820/
  multiple_receiver_train.zip
  multiple_receiver_test.zip
  receiver_drift_dataset.zip   # optional this beat
```

The audited HDF5 files expose packet rows and labels but no capture/session identifiers or attributes. Therefore later work must not invent capture groups or treat packets as proven-independent experimental units. Exact-row overlap was zero for audited N210_1 source-train versus seen-test and source-train versus drift-day1, which rules out exact duplicates only.

**Next beat (awaiting Human GO):** X1 signal-level OOB audit using source/train receivers and receiver-drift data only. No classifier training. The blind six remain sealed.

---

## Forbidden

- Retrain F0/C' on OSU RX after seeing RX_FAIL
- Open F1 / change `a` / add utility or GRL “to save Paper 2”
- Claim “F0 solves cross-receiver” from Day4 synthetic numbers
- Train on Zhang data in this beat
- Extract or inspect signal values from the six official blind receivers before X6
