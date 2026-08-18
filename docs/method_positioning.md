# Method Positioning

This document records the recommended wording for the current method and the claims that should be avoided.

## Claims to Avoid

Do not write:

- We replace CNN with Transformer.
- We are the first to use OOB.
- We significantly outperform CNN in file-level accuracy.

These claims are either technically inaccurate or not yet statistically supported by the current file-level evaluation size.

## Recommended Claims

Use wording like:

- We propose a chirp-aware OOB-guided hybrid architecture for LoRa RFFI.
- CNN-stem preserves local waveform impairment modeling.
- OOB-guided cross-attention dynamically injects OOB hardware-distortion evidence.
- Chirp embedding aligns token representation with LoRa chirp structure.
- Under strict Day1-to-Day2 evaluation, the method improves window-level accuracy and macro-F1 over reproduced OSU-CNN-IQ.
- File-level accuracy shows a positive trend but should be interpreted with bootstrap confidence intervals and paired comparison due to limited test files.

## Current Model Position

The current main line is not "Transformer replacing CNN." It is CNN-enhanced RF-HSTU:

```text
CNN-stem + RF-HSTU + OOB-guided cross-attention + chirp embedding
```

The CNN-stem captures local RF waveform impairments. RF-HSTU models patch-level sequence structure. OOB-guided cross-attention lets main RF tokens selectively use OOB distortion evidence. Chirp embedding injects a LoRa-specific structural prior into token positions.

Center loss is currently the most directly aligned embedding regularizer for prototype stability because it reduces intra-class dispersion. SupCon v2, multi-scale token fusion, hard-margin loss, score fusion, and test-time adaptation are useful ablation or diagnostic tools, but they are not the main method line.
