# Phase 2C — External multi-receiver dataset audit (pre-registered)

**Status:** Local scan **LOCAL_ABSENT**. Human download is the current beat. No train.  
**Does not reopen:** F0 retune, F1, S1, `a` range, Day5, C1/C_fft/D, factorization, utility, oracle.

This audit **must not** change any frozen F0 / RX gate.

---

## Why this beat

OSU Indoor SameTx has only two receivers. F0 killed synthetic OOB-scale on Day4 but did **not** beat C' on real RX1↔RX2 (pooled window Δ −0.52 pp).

That closes “identity-first scale pairing on this backbone + OSU 2-RX” as a Paper 2 method.  
It does **not** close the mechanism story (OOB-scale shortcut is still real on Day4).

A second dataset with **many** receivers is the only protocol-clean pivot. Do not add modules to rescue F0 on two SDRs.

---

## Target corpus (paper facts; verify on disk later)

Official:

- Page: https://junqing-zhang.github.io/dataset-code/
- IEEE DataPort: https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers
- DOI: 10.21227/d6vx-r538
- Paper: Shen, Zhang et al., TMC 2024, “Towards Receiver-Agnostic and Collaborative RFFI”

Claimed capture (from the paper, not yet verified on files):

| Item | Paper claim |
| --- | --- |
| DUTs | 10 LoRa (5× LoPy4 + 5× mbed SX1261) |
| Receivers | 20 SDR, 6 models |
| LoRa BW | 125 kHz, SF7, 868.1 MHz |
| RX sample rate | **1 MHz** |
| Representation in their code | preamble extract + channel-independent spectrogram |
| Zips | `multiple_receiver_train.zip` ~13 GB, `test` ~11 GB, `receiver_drift` ~3.7 GB |

OOB usability is **unknown until files are opened**. 1 MHz / 125 kHz *can* leave OOB if the dump is full-band complex IQ. If the release is already preamble-cropped / filtered / spectrogram-only, **do not force our OOB branch onto it**.

---

## This beat (no train, no GPU)

```text
script   experiments/paper1_audit/scripts/audit_external_lora_rx.py
search   local paths only; do not download DataPort
report   present / absent; IQ vs spectrogram; fs; OOB guess
forbid   training; changing F0/RX gates; IEEE login scrape
```

| Verdict | Meaning |
| --- | --- |
| `LOCAL_ABSENT` | not on disk; Human must download; stop |
| `OOB_UNKNOWN` | files exist but format not enough to judge OOB |
| `OOB_OK` | raw-ish IQ at ≥1 MHz, spectrum not obviously in-band-only |
| `OOB_INSUFFICIENT` | spectrogram / baseband-cropped / no complex IQ → do not adapt F0 |

`OOB_INSUFFICIENT` ⇒ do not rewrite our method to fit the dataset.  
`OOB_OK` ⇒ later Human GO for a **new** multi-receiver DG protocol (not a retune of F0 on OSU).

**Recorded:** `results/external_rx_audit/external_rx_audit.md` → **LOCAL_ABSENT**.

Drop downloaded zips here (do not unpack into `matched_seed0`):

```text
/data1/hcc/llm4RF/new_phase/data/external/zhang_tmc_multi_rx/
  multiple_receiver_train.zip
  multiple_receiver_test.zip
  receiver_drift_dataset.zip   # optional this beat
```

Then re-run `audit_external_lora_rx.py`. IEEE DataPort needs a personal login; the agent cannot fetch it.

---

## Forbidden

- Retrain F0/C' on OSU RX after seeing RX_FAIL
- Open F1 / change `a` / add utility or GRL “to save Paper 2”
- Claim “F0 solves cross-receiver” from Day4 synthetic numbers
- Train on Zhang data in this beat
