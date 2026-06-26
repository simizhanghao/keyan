# OOB Representation Equalization Report (Quick Mode)

> **v1 = embedding-level representation correction**, not waveform/log-spectrum equalization.
> Auxiliary experiment only; **RCPA-T remains the primary method**.

Direction: **rx1_to_rx2** | Seed: **0** | Split: **0** | K: 0, 1, 3, 5

Stats estimated from: **source receiver embeddings + target calibration Block A only**.
No support labels or query windows used for equalization.

---

## 1. Does OOB representation equalization lower receiver probe?

| Repr | Before | After (coral) | Δ |
|------|--------|----------------------|---|
| oob_only RX probe | 75.1% | 35.6% | +39.6 pp |

**Answer:** Yes, receiver probe decreases with coral.

---

## 2. Does it improve device probe?

| Repr | Before | After (coral) | Δ |
|------|--------|----------------------|---|
| oob_only device probe | 27.7% | 28.1% | +0.4 pp |

---

## 3. K=0: better than source-only?

| Method | K=0 file-acc |
|--------|-------------|
| RCPA-S (no eq) | 12.5% |
| OOB-Eq only (coral) | 16.7% |

---

## 4. K=1/3/5: further improvement over RCPA-T?

| K | RCPA-T | OOB-Eq+RCPA-T (coral) |
|---|--------|--------------------------------|
| 1 | 37.5% | 37.5% |
| 3 | 33.3% | 33.3% |
| 5 | 45.8% | 41.7% |

**Does OOB-Eq improve RCPA-T?** No — RCPA-T already absorbs most receiver shift.

---

## 5. Most stable equalization method

Best RX-probe reduction: **coral** (Δ=39.6 pp on oob_only receiver probe).

---

## 6. Receiver probe ↓ but acc ↑?

If receiver probe drops without file-acc gain, device-discriminative evidence may be partially suppressed alongside RX-specific bias. See probe table vs shot curve.

---

## 7. Worth OOB-Eq full mode?

**Yes — investigate on both directions**.

---

## 8. Worth waveform-level equalization?

Current v1 operates on **frozen embeddings**, not the OOB forward path. Waveform/log-spectrum equalization remains a **future direction** if embedding-level correction shows RX-probe reduction but insufficient acc gain.

---

## Positioning for paper 2

- **Main method:** RCPA-T (58.3% / 69.4% / 75.0% full mode)
- **This experiment:** validates that OOB receiver entanglement is partially suppressible at representation level
- **Not claimed as:** waveform-level OOB spectral response equalization
