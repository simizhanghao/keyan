# NLP-Inspired Solutions for Cross-Receiver RFFI Failure

This note summarizes why cross-receiver transfer fails in the current DI-RF-HSTU
line, why pseudo-label adaptation collapsed, and which NLP / domain-adaptation /
attention ideas are worth borrowing next. **No Qwen/LLM, no teacher forcing.**

## A. Why cross-receiver is hard

Cross-receiver failure is **not** an epoch-count problem. The dominant shift is
**receiver-induced style bias**, not insufficient training time.

Evidence from our diagnostics:

| Phenomenon | Observation |
|---|---|
| Energy / gain shift | RX2 vs RX1: IQ power +51.6%, in-band energy +51.6%, OOB energy +43% |
| Legacy OOB pollution | `oob_inband_ratio` almost unchanged (-4.3%), but absolute OOB scales with receiver |
| Normalization fix | `oob_ratio` >> `legacy_zscore` (bidirectional avg f1 0.180 vs 0.095) |
| CFO cue | Small but real gain from peak_offset / spectral_centroid auxiliary features |
| Direction asymmetry | RX2→RX1 much easier than RX1→RX2 for CNN; Hybrid wins hard direction only |
| Data scarcity | 1 file / device / receiver → 24 train files; models peak early (epoch 7–16) |

The model sees **receiver coloration** (gain, spectral tilt, noise floor) mixed
with **transmitter hardware cues** (CFO spread, relative OOB shape). Source-only
training on one receiver does not produce a representation that is invariant when
the other receiver re-scales the spectrum.

**Upper-bound diagnostic (recommended before any new method):**

Train/eval on **target receiver labeled data** (oracle / transductive upper bound).
If accuracy jumps to >> source-only, the bottleneck is domain shift, not class
separability. If it stays low, the 24-class task itself may be too hard under
current windows.

## B. Why pseudo-label collapsed

Target-unlabeled pseudo-label / prototype calibration failed because **the source
model's target predictions are already collapsed** before adaptation:

- Wrong predictions concentrate on a few receiver-favored classes.
- High-confidence windows are **confidently wrong**, not uncertain.
- Threshold tuning cannot fix a prior that assigns most target mass to ~3–5 classes.
- RF-specific checks (CFO agreement, OOB-ratio similarity) reduce false accepts
  but cannot create diversity where the classifier has none.

Pseudo-label assumes a **reasonable initial decision boundary on the target
domain**. Our boundary is structurally biased by receiver style. Continuing to
tune `pseudo_threshold` / `topk` is not productive.

## C. Why advanced attention cannot be stacked blindly

Current Hybrid already has:

- CNN-stem local impairment modeling
- OOB-guided cross-attention (main tokens attend OOB tokens)
- Chirp positional embedding

Adding deeper / wider attention without disentangling **content vs style** risks:

1. **More capacity to memorize source-receiver coupling** (especially with 24
   train files).
2. **Attention over polluted absolute-OOB tokens** (partially fixed by `oob_ratio`,
   but main/IQ path still carries receiver gain).
3. **No guarantee of invariance** — attention weights can still route by receiver
   energy rather than transmitter identity.

NLP lesson: **more attention ≠ better generalization**. Useful patterns from NLP /
vision DA:

- **Bottleneck / query compression** (Perceiver, Q-Former): force information
  through a small set of latent queries → reduces spurious high-dimensional style.
- **Content–style disentanglement** (NLP style transfer, image domain adaptation):
  separate streams or losses so style factors do not enter the classifier path.
- **Feature-statistics alignment** (CORAL, MixStyle): match low-order statistics
  across domains without target labels.
- **FiLM / adapter calibration**: lightweight scale/shift on features conditioned
  on domain statistics, not full backbone retrain.

## D. Three recommended directions

### 1. CORAL + Information Maximization (Priority: **P0 — implement now**)

**Idea:** During training, minimize source CE while aligning **second-order feature
statistics** (CORAL) between source-labeled and target-unlabeled embeddings, plus
IM on target logits to sharpen but not collapse predictions.

```
loss = CE_source
     + λ_coral · CORAL(z_source, z_target)
     + λ_im    · IM(logits_target)        # coral_im mode only
```

**Pros:** Lightweight, no target labels, no new backbone, well-studied in DA.
**Cons:** Aligns covariance only — may not fix higher-order receiver effects.
**Complexity:** Low (~100 lines, dual DataLoader).
**Expected gain:** Moderate; best hope to lift RX2→RX1 without collapsing RX1→RX2.
**Risk:** λ too large → erases class structure; needs 30ep sweep first.
**Experiment:** `scripts/run_cross_receiver_coral_im_sweep.sh` (30ep, D2 base model).

### 2. Transmitter Query Bottleneck Attention (Priority: **P1 — design now, implement if CORAL+IM works**)

**Idea:** Replace (or wrap) full-sequence pooling with **K learnable transmitter
queries** that cross-attend to a mixed token bank:

- main RF tokens (CNN-stem)
- OOB-ratio tokens
- optional CFO token(s)

Queries compress the window into **device-centric latents** before classification.
Receiver-style variation lives in many tokens; queries must aggregate **invariant
evidence** to explain the label.

Inspired by: Perceiver IO, Q-Former, query-bottleneck transformers in NLP.

**Pros:** Directly addresses "receiver style enters the pooled vector"; small K
limits overfitting.
**Cons:** New module + tuning; needs careful init so queries don't collapse.
**Complexity:** Medium (new `TransmitterQueryBottleneck` module, ~200–300 lines).
**Expected gain:** High if CORAL+IM shows alignment helps but plateaus below CNN.
**Risk:** K too small → information loss; K too large → same as mean-pool.
**Design doc:** `docs/query_bottleneck_attention_design.md`

### 3. Receiver-Disentangled Dual-Stream Attention (Priority: **P2 — defer**)

**Idea:** Explicit **content stream** (transmitter) and **style stream** (receiver)
with:

- content stream: OOB-ratio + CFO + chirp-aligned tokens → classifier
- style stream: gain / tilt / noise statistics → auxiliary head or adversarial
  gradient reversal
- optional MixStyle: randomize feature mean/var during training on source

Inspired by: NLP style/content disentanglement, MixStyle (ICLR), domain-adversarial
NLP.

**Pros:** Conceptually clean paper story for IoTJ.
**Cons:** Highest implementation and tuning cost; easy to over-engineer with 24
train files.
**Complexity:** High.
**Expected gain:** High ceiling, but only after P0/P1 validate alignment/bottleneck.
**Risk:** Style head may leak into content; needs careful ablation.

## E. Experiment priority roadmap

| Step | Action | Gate to proceed |
|---|---|---|
| 0 | Upper-bound: target-receiver labeled eval | Confirms shift vs separability |
| 1 | **CORAL+IM 30ep sweep** (P0) | RX2→RX1 improves, RX1→RX2 stable |
| 2 | Query Bottleneck Attention (P1) | CORAL+IM helps but avg < CNN iq_rms |
| 3 | Dual-stream disentangle (P2) | P1 + CORAL still insufficient |
| — | ~~Pseudo-label threshold sweep~~ | **Stop** — collapsed prior |
| — | ~~Deeper attention / center loss / SupCon~~ | **Stop** — evidence against |

## F. What we explicitly do **not** do next

- Center loss, SupCon, hard margin, multi-scale, deeper backbone
- More receiver-style augmentation tuning (D3 failed)
- Pseudo-label / TENT-only recipes without fixing representation
- Qwen / LLM / teacher forcing
