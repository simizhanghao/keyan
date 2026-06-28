# Core-Change Audit (thesis-em-openset)

**Date:** 2026-06-28  
**Commit `79bf5d5`:** only adds `src/rfhstu/em_perturbations.py` (new file).  
**Uncommitted working tree:** modifies `data.py`, `losses.py`, `models.py`, `oob_fusion.py`, `train_utils.py` (Paper 2 / thesis extension work).

---

## 1. Per-file summary

| File | Why changed | Default behavior |
|------|-------------|------------------|
| `em_perturbations.py` (committed) | Chapter 5 EM perturbation module | N/A (new); not imported by Paper 1/2 eval unless explicitly used |
| `data.py` | Add optional `fold` filter to `load_manifest` | **Same** when `fold=None` (default) |
| `losses.py` | Add `focal_loss`, `macro_f1_from_logits` helpers | **Same** — new functions; existing losses unchanged |
| `oob_fusion.py` | Add `OOBGatedFusion` for `gated_oob` ablation | **Same** when `oob_fusion_type≠gated_oob` |
| `models.py` | Gated OOB path; optional `oob_dropout`, `mixstyle`; encoder path refactor for mixstyle hook | **Same at eval** when `oob_dropout=0`, `mixstyle=False` (checkpoint defaults) |
| `train_utils.py` | Paper 2 args: `train-split`, `val-split`, `fold`, focal, SWA, mixstyle, oob-dropout; stricter val split | **Risk:** `make_datasets` no longer falls back if val empty — only affects training scripts without val rows |

---

## 2. Paper 1 / Paper 2 checkpoint compatibility

| Check | Result |
|-------|--------|
| `F_cross_attn_chirp_plain/seed_0/best.pt` loads | ✅ |
| Clean Day5 test file-acc (256 win/file) | **83.33%** (matches Phase B full) |
| `cross_attn_oob` forward path | Equivalent to old `encoder()` when mixstyle/oob_dropout disabled |
| `A_cnn_iq/seed_0/best.pt` exists | ✅ path present |

**Conclusion:** Frozen Paper 1 Ours checkpoint **inference unchanged** for default eval flags.

---

## 3. `experiments/cross_receiver_calibration/`

- Scripts use their own eval paths and checkpoint args.
- New `gated_oob` / `mixstyle` only activate with explicit CLI flags (Paper 2 RCPA runs).
- **No change** to frozen result CSVs in `experiments/cross_receiver_calibration/` (not modified).

---

## 4. Optional flags (new features off by default)

| Feature | Flag | Default |
|---------|------|---------|
| Gated OOB | `--oob-fusion-type gated_oob` | `concat_oob` / ckpt-stored type |
| OOB dropout | `--oob-dropout` | `0.0` |
| MixStyle | `--mixstyle` | off |
| Focal loss | `--loss-type focal` | `ce` |
| EM perturbations | only in `em_robustness_openset/` scripts | — |

---

## 5. Should EM-CR move to a wrapper?

**Recommendation:** Keep EM-CR in `experiments/em_robustness_openset/train_em_consistency.py`; do **not** further modify `DeviceClassifier.forward` for CR. Use:

- freeze backbone + head-only FT for debug;
- stop-gradient teacher for KL;
- perturbation sampling in `em_perturbations.py` only.

Uncommitted `models.py` changes are for **Paper 2 gated OOB / mixstyle**, not EM-CR. Consider committing them on `thesis-rffi-extension` separately, not mixing into Chapter 5 freeze commits.

---

## 6. Verification command

```bash
/new_nfs/haiyu/anaconda3/bin/python experiments/em_robustness_openset/eval_robustness_curves.py \
  --manifest data/paper/cross_day_day1to5_source_only.csv \
  --checkpoint outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt \
  --perturb-type awgn_snr_db --strengths 100 \
  --samples-per-file 256 --device cuda \
  --out-csv /tmp/clean_verify.csv
# Expected: file_acc=0.8333
```

---

## 7. Action items

1. ✅ Clean baseline verified on current code.
2. ⚠️ Uncommitted core changes should be **committed on Paper 2 branch** or reverted on `thesis-em-openset` if not needed for Ch.5.
3. Do **not** enable `mixstyle` / `oob_dropout` when reproducing Paper 1 frozen numbers.
