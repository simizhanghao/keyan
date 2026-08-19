# Phase 2B-1 — Identity-Anchored Counterfactual OOB Training (pre-registered)

**Status:** F0 5-seed **F0_5SEED_GO**. Synthetic line closed. Next: Phase 2B-2 real RX audit (`PHASE2B2_REAL_RX_LOCK.md`). F1 closed.  
**Does not reopen:** S1 retune, `a` range, −2 pp gate, Day5, RX2, C1/C_fft/D, factorization, utility.  
**S1 5-seed CLEAN_FAIL and S0 SCALE_TAX stay frozen** (`PHASE2B0_SCALE_AUG_LOCK.md`).

Working name (internal): **IACT** — Identity-Anchored Counterfactual Training.  
Not a manuscript title. Not “S1 v2”.

---

## Why this beat (protocol-clean Case D)

Protocol audit (C' / S0 / S1 seed 0–4): same manifest, Day1–3/Day4, 80 ep / 3e-3 / bs 128,
`iq_rms` / `log_zscore` / `oob_norm=ratio` / `fft_source=full`, clean Day4 ckpt.
S1 seed 2/3/4 match seed 0/1 recipe; `pretrained=null` (scratch). CLEAN_FAIL is **not** a
protocol drift.

Reading (unchanged):

```text
problem GO; mechanism GO; S1-as-final-method NO-GO; real RX sealed
```

Trade-off: forcing scale invariance from scratch compresses window Acc into ~41.7–43.4
while stronger C' seeds sit at 46.1–46.7 (−4 pp). S0 shows it is not two-forward tax.

Question this phase may answer:

> Can identity-first (then counterfactual scale pairing) keep clean identity **and**
> kill the OOB-scale shortcut — without changing architecture or `a~U[0.5,2.0]`?

---

## Hard protocol locks (F0 and F1)

| Item | Rule |
| --- | --- |
| Architecture | Frozen C' (`rf_hstu`, cnn_stem, chirp, cross_attn_oob, `oob_norm=ratio`) |
| Manifest / split | `cross_day_day1to5_source_only.csv`; Day1–3 train / Day4 val; Day5 unused |
| `a` | `U[0.5, 2.0]` — **never change** |
| Stage-2 recipe | 80 ep, lr **3e-3**, wd 5e-4, bs 128, CE, ckpt = clean Day4 **acc** |
| Optimizer | **Fresh** AdamW (do not load C' optimizer state) |
| Stress for ckpt | **Forbidden** |
| Target / RX2 | **Closed** |
| Seeds first | **2, 3, 4 only** (the CLEAN_FAIL set). 0/1 later only after F0 or F1 GO |
| Init | Matching-seed frozen C' **full** `best.pt` via `--init-checkpoint` |
| `--pretrained` | **Forbidden for F0/F1** (encoder-only; drops classifier) |

Unique legal difference vs S1 for **F0**:

```text
S1:  scratch + paired_view=oob_scale
F0:  --init-checkpoint = C_full_ratio/seed_S/best.pt  + paired_view=oob_scale
     same 80 / 3e-3 / everything else
```

No lr drop, no shorter stage-2, no schedule sweep.

---

## Arms (conditional; do not run both at once)

### F0 — Identity-first paired-scale (authorized after init smoke PASS)

```text
θ ← full load matching-seed C' best.pt
train 0.5 CE(x)+0.5 CE(T_a(x)), a~U[0.5,2.0], lock_inband
val / ckpt: clean Day4 only
```

Runner: `scripts/run_f0_identity_first.sh` (default `GPUS=5`, sequential).  
Name: `C_full_ratio_init_paired_scale`

### F1 — Explicit clean identity anchor (authorized **only if** F0 clean FAIL)

```text
same as F0, plus frozen matching-seed C' teacher (eval, stop-grad)
L = (CE_clean + CE_scale + KL(p_T(x) || p_S(x))) / 3
anchor on clean view only; never on T_a(x)
temperature fixed; λ not swept (equal-weight /3)
```

F1 needs a code change (paired-view currently rejects extra losses). Do not implement
until F0 clean FAIL is recorded.

---

## Gates (frozen before results)

Primary metric: **window** Acc. Collapse: `Δ_clean(arm − same-seed C') ≤ −15 pp`.

### Gate A — clean (hard; before any stress)

Per seed: `Δ ≥ −2 pp` and not collapse.  
F0/F1 on seeds 2/3/4: **3/3 PASS**.  
CLEAN_FAIL ⇒ do not read scale / full RX; do not open F1 from stress; do not retune.

### Gate B — oob_scale (only if Gate A PASS)

`D = Acc_clean − Acc_oob_scale` vs own clean.  
STRONG: mean D < 8 and 3/3 D < 15.  
PASS: mean D < 15 and 3/3 D < 15.  
FAIL: any D ≥ 15 or mean ≥ 15.

### Gate C — full RX (recorded after A+B; not a retune knob)

Report mean±std. Ideal TRACKS_SCALE if mean D_full < 15.

### Unlock ladder

```text
init-checkpoint smoke PASS
        ↓
F0 seeds 2/3/4
        ↓
Gate A PASS?
   /        \
 YES         NO  → F1 (same seeds/gates); if F1 also CLEAN_FAIL or B FAIL
  ↓                → STOP synthetic line (no S2/S3/utility/factorization/retune a)
Gate B not FAIL
  ↓
expand to 5 seeds (same gates; ≥4/5 clean, mean Δ≥−2, 0 collapse; scale as 2B-0)
  ↓
later Human GO → real RX1↔RX2
```

**Termination:** If F0 and F1 both fail to pass A+B on {2,3,4}, stop single-source
synthetic-scale method search. Pivot to multi-receiver DG / external dataset audit.
Do not add modules to “rescue” this backbone under the frozen protocol.

---

## Init smoke — SMOKE_PASS (recorded; do not re-run)

```text
script   experiments/paper1_audit/scripts/smoke_init_checkpoint.py
seed     2
full_load missing=[] unexpected=[] classifier_rel_l2=0
pretrained classifier_rel_l2_vs_ckpt=1.002931 (documents F0 forbid)
Day4     frozen=46.0775  smoke=46.0775  abs_pp=0.0  (tol 0.05)
artifact results/matched_seed0/init_checkpoint_smoke/init_checkpoint_smoke.json
forbid   training during smoke; Day5; RX2; stress
```

F0 train is now authorized. F1 is not.

---

## F0 seeds 2/3/4 — F0_GO (recorded; do not retrain)

Artifact: `results/matched_seed0/f0_identity_first_gate.md`

| Gate | Result |
| --- | --- |
| A clean | **3/3 PASS**, mean Δ **+3.53** pp, 0 collapse |
| B oob_scale | **GATE_B_STRONG**, mean D **1.1±0.8** pp, 3/3 < 15 |
| C full RX | **TRACKS_SCALE**, mean D **5.9±2.1** pp |

Per-seed clean window: C' 46.1/46.5/46.7 → F0 52.9/48.5/48.5.  
Seed 2 +6.8 is large; recorded, not a retune knob.

F1 is **not authorized** (clean did not fail).

---

## 5-seed expand (this beat; seeds 0/1 only)

Runner: `scripts/run_f0_expand_seeds01.sh`  
Does **not** retrain 2/3/4. Same `--init-checkpoint` + `paired_view=oob_scale` recipe.

Pre-registered 5-seed gates (frozen before 0/1 results):

| Gate | Rule |
| --- | --- |
| Clean | ≥4/5 `Δ ≥ −2` pp, mean Δ ≥ −2, **0 collapse** |
| Scale | STRONG: mean D < 8 and 5/5 D < 15. PASS: mean < 15 and 5/5 < 15. FAIL: any ≥ 15 or mean ≥ 15 |
| Full RX | record mean±std; TRACKS_SCALE if mean < 15 (not a retune knob) |

CLEAN_FAIL ⇒ do not read 0/1 stress; do not open RX2; do not open F1; do not retune.  
`F0_5SEED_GO` ⇒ recorded below; real RX is Phase 2B-2.

---

## F0 5-seed — F0_5SEED_GO (recorded; do not retrain)

Artifact: `results/matched_seed0/f0_5seed_stability.md`

| Gate | Result |
| --- | --- |
| Clean | **5/5 PASS**, mean Δ **+3.14** pp, 0 collapse |
| Scale | **SCALE_STRONG**, mean D **1.5±0.8** pp, 5/5 < 15 |
| Full RX | **TRACKS_SCALE**, mean D **6.9±2.0** pp |

F1 stays closed. Do not reopen S1. Real RX: `PHASE2B2_REAL_RX_LOCK.md`.

---

## Forbidden without a new Human GO

- Change `a` range / lr / epoch / wd / batch
- Use `--pretrained` for F0/F1
- Read S1 seed 2/3/4 stress; move −2 pp; open RX2/Day5
- Reopen C1 / C_fft / D / factorization / utility / DCT / GRL / SupCon
- Sweep λ or temperature for F1
- Run F1 before F0 clean FAIL
- True In-Band Main / LODO as blockers of this phase

---

## Paper 1

Mechanism audit closed. H4_PASS / per-device MIXED done. True In-Band Main and LODO
remain queued; they do not block 2B-1.
