# Paper 1 (IoTJ) Revision Risk Plan

> **Status:** Submitted — do **not** modify the submitted Overleaf/version.  
> Use this document only when reviewer comments arrive.

---

## Risk 1: Baseline / SOTA comparison insufficient

**Likely reviewer comment:**
> Why not compare with DeepLoRa, scalable channel-robust LoRa RFFI, contrastive RFFI, or other published LoRa RFFI systems?

**Current strength:**
- Reproduced CNN-IQ under unified Day1–3/Day4/Day5 protocol
- Controlled fusion/chirp ablations on the same backbone family
- Bootstrap CI and multi-seed cross-day evaluation

**Weakness:**
- No reproduced external LoRa RFFI code under our exact protocol
- Public datasets/code (e.g., Junqing Zhang LoRa RFFI resources) use different splits, receivers, and enrollment rules

**Revision plan:**
1. Add **same-protocol reproduced baselines** (e.g., spectrogram CNN, ResNet-on-IQ, Siamese/metric variant) trained/evaluated on OSU Day1–5 split — not leaderboard numbers from other papers.
2. Add a **comparison fairness paragraph**: prior high-accuracy LoRa RFFI results are cited for context but not claimed as direct numerical superiority.
3. If reviewer names a specific method with public code, attempt **limited reproduction** under our Day1–5 protocol only; report honestly if data/split mismatch prevents fair comparison.
4. Optional: add stronger internal baseline (e.g., spectrogram ResNet) if runtime allows — **after** reviewer request, not preemptively changing submitted narrative.

---

## Risk 2: Cross-receiver results too weak

**Likely reviewer comment:**
> Cross-receiver transfer is near chance; practical value is limited.

**Current strength:**
- Cross-receiver reported honestly as **stress test / limitation**, not solved
- Same-receiver cross-day gain (+20.8 pp vs CNN-IQ) is the primary claim

**Revision plan:**
1. **Do not** claim receiver-independent authentication in Paper 1.
2. Emphasize cross-receiver as deployment boundary discovered by strict evaluation.
3. If reviewer asks for mitigation: cite that receiver calibration is an open problem; **do not** merge Paper 2 RCPA results into Paper 1 unless explicitly requested and page limit allows.
4. Optional one-sentence future work: "target-receiver calibration with minimal labeled windows" — avoid detailed overlap with Paper 2 before Paper 1 decision.

---

## Risk 3: Novelty insufficient (architecture / OOB not new enough)

**Likely reviewer comment:**
> OOB evidence and attention fusion are not novel; contribution is incremental.

**Revision plan:**
1. Emphasize **controlled ablation evidence**: concat/gated OOB unstable; cross-attention OOB fusion stable under cross-day small-sample protocol.
2. Highlight **RF-HSTU + chirp-aware tokenization + OOB cross-attention** as an integrated design choice, not isolated components.
3. Reframe claims around **robustness mechanism** and **evaluation rigor**, not "first OOB RFFI."
4. Add related-work table distinguishing concat vs cross-attn vs prior LoRa spectrogram CNN if space allows.

---

## Risk 4: Dataset / evaluation scope limited

**Likely reviewer comment:**
> Single public dataset, 24 devices, one file per test day; generalization unclear.

**Revision plan:**
1. Strengthen deployment-shift subsection (distance, indoor location) with clearer protocol description.
2. Add variance discussion: file-level voting, small test set per fold — already partially addressed via bootstrap CI.
3. If requested: add leave-one-config or leave-one-location analysis using existing manifest splits (check if already in repo).
4. Avoid over-claiming outdoor/multi-vendor robustness.

---

## Risk 5: Overlap with Paper 2 (companion work)

**Likely reviewer comment:**
> Related concurrent/companion manuscript on cross-receiver calibration?

**Revision plan:**
1. If Paper 1 accepted first: cite Paper 2 formally only after its publication/decision; in revision, one sentence on "receiver calibration studied separately."
2. If reviewer discovers overlap during review: offer cover-letter style disclosure; emphasize Paper 1 = architecture + cross-day; Paper 2 = diagnosis + calibration (different contributions).
3. **Do not** auto-merge papers in revision unless rejection explicitly requires unified cross-receiver story.

---

## Risk 6: Figures / presentation

**Likely reviewer comment:**
> More qualitative figures, architecture clarity, or failure case analysis needed.

**Revision plan:**
1. Add confusion-matrix or failure-case panel for cross-day vs cross-receiver (reuse existing stress-test figures).
2. Ensure architecture figure readable at IoTJ column width.
3. No new main results required unless reviewer asks for specific missing experiment.

---

## Priority matrix

| Risk | Probability | Impact | Preemptive action now |
|------|-------------|--------|------------------------|
| Baseline insufficient | **High** | High | Prepare baseline reproduction scripts list; no change to submitted PDF |
| Cross-receiver weak | Medium | Medium | Already framed as limitation — hold line |
| Novelty | Medium | High | Keep ablation narrative strong in rebuttal draft |
| Dataset scope | Medium | Medium | Document protocol limits in rebuttal template |
| Paper 2 overlap | Low–Medium | High | Use `PAPER2_OVERLAP_AUDIT.md`; wait for P1 decision |
| Figures | Low | Medium | Keep v1 figure pack ready |

---

## What NOT to do before reviewer response

- Do not modify submitted IoTJ PDF/Overleaf project
- Do not merge Paper 2 RCPA into Paper 1 proactively
- Do not run large new experiment campaigns without reviewer-specific request
- Do not submit Paper 2 while Paper 1 under review (unless explicit strategy change)

---

## Rebuttal template snippets (draft)

**On baselines:**
> We agree that direct leaderboard comparison with prior LoRa RFFI systems is protocol-sensitive. We reproduced CNN-IQ and controlled fusion variants under a unified Day1–5 split. We will add [X] same-protocol baseline(s) in revision if requested.

**On cross-receiver:**
> We explicitly report cross-receiver transfer as an open limitation under source-only evaluation. Our primary contribution is same-receiver cross-day robustness via OOB cross-attention fusion, not receiver-independent deployment.

**On novelty:**
> Our contribution is the combination of RF token sequence modeling, chirp-aware embeddings, and selective OOB cross-attention, with ablations showing that naive OOB concatenation fails under the same protocol.
