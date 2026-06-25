# Cross-Receiver Findings (Diff_Receivers_Setup_Indoor_SameTx, 24 classes)

> **DEPRECATED / DO NOT USE FOR PAPER MAIN RESULTS**
>
> This document contains early cross-receiver diagnostic results based on **target-RX validation manifests** (`data/manifest_rx*_to_rx*.csv` where `val` = target receiver). These runs are **not strict source-only** and may **overestimate** cross-receiver performance (e.g. inflated RX2→RX1 CNN ~58%).
>
> **Final paper results must use:**
> - Manifests: `data/paper/*_source_only.csv` (train/val on source RX only; test on target RX)
> - Output path: `outputs/paper_ready_v3/phase5_clean_cross_receiver/`
>
> 注意：本文档包含早期 target-val 协议结果，不满足 strict source-only，**不得用于论文主表**。

Diagnostic-only report. No training was run, no backbone change, no center loss /
SupCon / multi-scale / hard margin / TTA. Subject model = center_none Hybrid
(`classifier mean_logits`), compared against the OSU-CNN-IQ baseline.

Data: RX1 and RX2, same transmitters, indoor. Raw Device9 excluded, remapped to
24 contiguous classes (device 1..24, label 0..23) — identical convention to the
cross-day (LODO) experiment. One `.dat` per (receiver, device), ~2441 windows of
8192 samples (~20 s) each.

## Headline results (direct transfer, no adaptation)

| Direction | Model | window_acc | file_acc | macro_f1 |
|---|---|---|---|---|
| RX1 -> RX2 | CNN    | 0.0850 | 0.0833 | 0.0213 |
| RX1 -> RX2 | Hybrid | 0.1662 | 0.2083 | 0.1135 |
| RX2 -> RX1 | CNN    | 0.4146 | 0.5833 | 0.3517 |
| RX2 -> RX1 | Hybrid | 0.1571 | 0.2500 | 0.1423 |

Two-direction average (classifier mean_logits): CNN window 0.250 / file 0.333 /
f1 0.187; Hybrid window 0.162 / file 0.229 / f1 0.128. Chance level = 1/24 ≈ 0.042.

## 1. Cross-receiver direct transfer is much harder than cross-day

In the LODO cross-day study, both models stayed well above chance on every held-out
day (Hybrid average clearly usable). Here, with the *same* 24-class definition,
direct RX->RX transfer collapses toward chance: the best Hybrid direction reaches
only window_acc 0.166, and the CNN in RX1->RX2 (0.085) is barely 2x chance.
Changing the receiver induces a far larger distribution shift than changing the
collection day.

## 2. Results are highly asymmetric

RX2->RX1 is dramatically easier than RX1->RX2 for the CNN (window 0.415 vs 0.085,
file 0.583 vs 0.083). The spectrum audit shows why the domains are unequal:
RX2 has systematically higher captured energy than RX1
(iq_power_mean +51.6%, inband_energy +51.6%, oob_energy +43.3%, iq_amp_mean +22.0%).
Training on the higher-energy / richer RX2 and testing on RX1 transfers better than
the reverse. The two receivers are not interchangeable domains, so a single averaged
number hides a strong directional effect that must be reported per-direction.

## 3. Hybrid helps RX1->RX2 but fails RX2->RX1

- RX1->RX2: Hybrid > CNN on all three metrics (window 0.166 vs 0.085, file 0.208
  vs 0.083, f1 0.114 vs 0.021). Paired McNemar: Hybrid-only-correct 4, CNN-only 1
  (p=0.375, not significant; only 24 files).
- RX2->RX1: CNN >> Hybrid (window 0.415 vs 0.157, file 0.583 vs 0.250). Paired
  McNemar: CNN-only-correct 6, Hybrid-only 1 (p=0.125).

So the Hybrid does NOT stably beat the CNN across receivers; it wins the hard
direction and loses the easy one. Net average is below the CNN. This is the
opposite of a receiver-invariance claim.

## 4. Raw OOB cross-attention is not receiver-invariant enough

The OOB-guided cross-attention was intended to capture transmitter hardware
invariants. The spectrum stats argue it is currently picking up receiver-coupled
energy instead of a clean transmitter signature:

- The per-receiver `oob_inband_ratio` is nearly identical (RX1 0.01248 vs RX2
  0.01195, only -4.3%), i.e. the OOB band shape carries little receiver-invariant,
  device-discriminative structure to attend to — yet absolute OOB energy scales
  with the receiver (+43%).
- `peak_offset` (CFO proxy) differs between receivers by only ~804 Hz on average,
  far smaller than the inter-device spread within a receiver (~3000-4000 Hz),
  meaning the genuinely device-specific cues exist but are swamped by the
  receiver-level energy/coloration shift the model has latched onto.
- Confusion is structured, not random: top window-level confusions
  (1->3, 19->16, 11->4, 2->0, 14->4) collapse many true devices onto a few
  receiver-favored classes, the signature of a domain (receiver) shift dominating
  the transmitter identity.

## 5. Next steps: receiver/spectrum normalization, not a bigger backbone

Do not enlarge or restructure the backbone, and do not add center loss / SupCon /
multi-scale / hard margin. The evidence points to a domain-normalization
problem, so the productive directions are:

1. Receiver/spectrum normalization of the input (e.g. per-receiver energy / PSD
   normalization, whitening of the receiver gain and spectral tilt) so the
   transmitter fingerprint is not entangled with receiver energy.
2. Receiver-invariant representation learning that explicitly removes the receiver
   factor (the receiver is already a labeled domain in the manifest), rather than
   relying on raw OOB cross-attention alone.
3. Always report cross-receiver per-direction (RX1->RX2 and RX2->RX1) plus the
   asymmetry, never a single averaged accuracy.

## Artifacts

- `outputs/cross_receiver_analysis/data_audit.csv`
- `outputs/cross_receiver_analysis/receiver_spectrum_stats.csv`
- `outputs/cross_receiver_analysis/receiver_spectrum_summary_by_rx.csv`
- `outputs/cross_receiver_analysis/receiver_spectrum_summary_by_device.csv`
- `outputs/cross_receiver_analysis/{rx1_to_rx2,rx2_to_rx1}_confusion_matrix.csv`
- `outputs/cross_receiver_analysis/{rx1_to_rx2,rx2_to_rx1}_wrong_files.csv`
- `outputs/cross_receiver_analysis/top_confusion_pairs.csv`

## Augmentation + CFO Sweep Update

The follow-up 30-epoch cross-receiver source-only sweep confirms that direct
transfer is not solved by adding more losses or a larger backbone.

- `D0_oob_ratio_only` is already a strong source-only Hybrid anchor:
  two-direction average classifier mean_logits = 0.221 / 0.271 / 0.180
  for window_acc / file_acc / macro_f1.
- `D2_oob_ratio_cfo` gives a small improvement:
  two-direction average classifier mean_logits = 0.232 / 0.375 / 0.189.
- `D3_oob_ratio_rxaug_cfo` fails:
  two-direction average classifier mean_logits = 0.179 / 0.188 / 0.142.
- CNN with `iq_rms` remains clearly stronger than the Hybrid on RX2->RX1,
  showing that source-only direct transfer is still insufficient.

Current interpretation:

- OOB ratio normalization is clearly better than the legacy OOB z-score setup.
- CFO is useful, but only as a modest auxiliary cue.
- Receiver-style augmentation did not produce reliable receiver invariance.
- Source-only Hybrid is not yet stable enough to claim superiority over CNN
  `iq_rms` under cross-receiver transfer.

Therefore the next phase should move from source-only direct transfer to
**target-unlabeled domain alignment and representation bottlenecks**, not more
loss tuning or pseudo-label threshold sweeps.

## Pseudo-Label Adaptation Failed

Target-unlabeled pseudo-label / prototype calibration was attempted and **did not
work** because the source model's target predictions already collapse to a few
receiver-favored classes. High-confidence windows are confidently wrong; tuning
`pseudo_threshold` cannot fix a bad prior. **Do not continue pseudo-label.**

## Current Next Phase: CORAL + IM, Then Query Bottleneck

See `docs/nlp_inspired_solutions.md` and `docs/query_bottleneck_attention_design.md`.

1. **P0 (now):** CORAL + Information Maximization during training  
   `loss = CE_source + λ_coral·CORAL(z_s, z_t) + λ_im·IM(logits_t)`  
   Target unlabeled = val-receiver windows; **no target labels in loss**.

2. **P1 (if P0 helps):** Transmitter Query Bottleneck Attention — learnable queries
   cross-attend to main / OOB-ratio / CFO tokens; reduce receiver style in pooled
   representation.

3. **P2 (defer):** Receiver-disentangled dual-stream + MixStyle-style statistics
   randomization.

Also run an **upper-bound diagnostic**: labeled eval on target receiver to confirm
the gap is domain shift rather than fundamental non-separability.

## Deprecated Direction (do not pursue)

- Plain BN-adapt / TENT-only / pseudo_proto threshold sweeps as main method
- Center loss, SupCon, hard margin, multi-scale, deeper backbone
- More receiver-style augmentation (D3 failed)
- Qwen / LLM / teacher forcing
