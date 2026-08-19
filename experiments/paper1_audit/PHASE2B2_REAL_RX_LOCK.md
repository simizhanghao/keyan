# Phase 2B-2 — Source-only real RX1↔RX2 (pre-registered)

**Status:** Real RX **RX_FAIL**. OSU 2-RX method line **STOPPED**. Do not retune.  
**Does not reopen:** S1, F1, `a` range, Day5, C1/C_fft/D, factorization, utility, oracle target-val.

Day4 C' / F0 checkpoints are **not** transferable here. Real RX uses
`Diff_Receivers_Setup_Indoor_SameTx`, not the 5-day indoor setup.

---

## Why this beat

Synthetic line (frozen):

```text
S1 from-scratch        CLEAN_FAIL / SCALE_TAX
F0 identity-first      5/5 clean PASS, mean Δ +3.14 pp
                       D_scale 1.5±0.8  SCALE_STRONG
                       D_full  6.9±2.0  TRACKS_SCALE
```

Question this phase may answer:

> Does identity-first counterfactual OOB-scale training transfer to **unseen real receivers**,
> or only to synthetic `oob_scale`?

Paper 1 Phase5-clean file Acc (frozen, do not rewrite): RX1→RX2 F=18.1±3.9, CNN=4.2;
RX2→RX1 F=15.3±7.1, CNN=23.6. Chance ≈ 4.17%.

---

## Hard protocol locks

| Item | Rule |
| --- | --- |
| Manifests | `data/paper/rx1_to_rx2_source_only.csv`, `rx2_to_rx1_source_only.csv` |
| Train / val | **source receiver only** (existing paper split; same source file listed as train and val) |
| Test / eval | **target receiver only**; `--eval-split test` |
| Oracle | `*_oracle_target_val.csv` **FORBIDDEN** |
| Upper bound | `rx*_to_rx*_upper_bound.csv` **closed this beat** |
| Architecture | Frozen C' (`rf_hstu`, cnn_stem, chirp, cross_attn_oob, `oob_norm=ratio`) |
| Recipe | 80 ep, lr **3e-3**, wd 5e-4, bs 128, CE, ckpt = **source val acc** |
| `a` | `U[0.5, 2.0]` — never change |
| Day4 ckpts | **Do not load** Day4 C'/F0/S1 into RX models |
| Target for ckpt / retune | **Forbidden** |
| Seeds first | **0, 1** (2 seeds). Seed 2 later only after 2-seed GO |
| F1 / S1 / CNN / RCPA-T | **Closed this beat** (CNN / RCPA later if F0 vs C' is GO) |

### Arms (must retrain on RX manifests)

```text
C'_RX : scratch + paired_view=off     on source train/val
F0_RX : --init-checkpoint = matching-seed same-direction C'_RX best.pt
        + paired_view=oob_scale
        same 80 / 3e-3 / source-val ckpt
```

Name: `C_full_ratio` and `C_full_ratio_init_paired_scale` under an RX results tree,
not the Day4 `matched_seed0` tree.

---

## Gates (frozen before any RX train)

Primary metric: **window** Acc on target **test**.  
Also report file Acc (1 file / device; 1 file = 4.17 pp). File Acc is **not** a retune knob.

Per direction: `Δ = Acc_F0 − Acc_C'` (same seed, then mean over seeds).  
Pooled: mean of the two direction means.

| Verdict | Rule |
| --- | --- |
| **STRONG_GO** | both directions F0 > C'; pooled Δ ≥ 8 pp; at least one direction ≥ 10 pp |
| **WEAK_GO** | both directions Δ > 0; 4 ≤ pooled Δ < 8 |
| **FAIL** | pooled Δ < 4 **or** any direction mean Δ ≤ −2 pp |

Do **not** use RX2 numbers to change `a`, lr, epoch, or to open F1.

After FAIL: stop this backbone on OSU 2-RX. Pivot to external multi-receiver audit.
Do not add utility / factorization to rescue.

---

## Unlock ladder

```text
F0_5SEED_GO (Day4 synthetic)
        ↓
RX manifest audit PASS  ← this beat (no GPU)
        ↓
later Human GO: train C'_RX then F0_RX, seeds 0/1, both directions
        ↓
read target test once
        ↓
STRONG/WEAK_GO → later CNN / 3rd seed / external dataset
FAIL            → stop OSU-2RX method line
```

---

## Manifest audit (this beat)

```text
script   experiments/paper1_audit/scripts/audit_rx_manifests.py
check    train/val receivers = source only; test = target only
check    24 devices; files exist under /data1/hcc/llm4RF
forbid   oracle rows; Day5; training
```

AUDIT_FAIL ⇒ do not open RX train.

**Recorded:** `results/rx_manifest_audit/rx_manifest_audit.md` → **RX_MANIFEST_PASS**.
train/val = source only; test = target only; 24 devices; 0 missing files.

Runner: `scripts/run_rx_f0_source_only.sh` (default `GPUS=5,6`).
Tree: `results/real_rx_source_only/` — not `matched_seed0`.

---

## Real RX result — RX_FAIL (recorded; do not retrain / retune)

Artifact: `results/real_rx_source_only/rx_f0_vs_cprime.md`

| Direction | C' win (0/1) | F0 win (0/1) | mean Δ win |
| --- | --- | --- | ---: |
| RX1→RX2 | 15.5 / 16.9 | 13.9 / 16.6 | **−0.95** |
| RX2→RX1 | 15.1 / 15.4 | 14.3 / 16.0 | **−0.10** |

Pooled Δ window = **−0.52 pp**. File Acc is 1-file noise (±4.17 pp) and is not read.

Reading (frozen):

```text
synthetic F0 GO; real-RX F0 transfer NO-GO
OSU 2-RX method line STOP
do not change a / lr / epoch; do not open F1
```

Identity-first scale pairing killed the **synthetic** OOB-scale shortcut. It did **not** improve unseen real receivers under the frozen source-only protocol.

Next: `PHASE2C_EXTERNAL_DATASET_LOCK.md` (no GPU).

---

## Forbidden without a new Human GO

- Train before audit PASS
- Load Day4 checkpoints onto RX
- Use oracle / target-val checkpoint
- Open CNN, RCPA-T, S0, F1, S1, upper-bound as this beat
- Change `a` / lr / epoch after seeing target Acc
- Day5 / LODO / True In-Band Main as blockers
