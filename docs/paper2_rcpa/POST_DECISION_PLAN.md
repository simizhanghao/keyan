# Post-Decision Plan (Paper 1 → Paper 2)

> Execute only after Paper 1 IoTJ decision. Do not submit Paper 2 before then unless explicitly approved.

---

## Case 1: Paper 1 accepted / minor revision

**Action:**
- Submit Paper 2 as **independent companion paper**
- Use Cover Letter **Version A**
- Cite Paper 1 formally; update `refs.bib` placeholder
- Target venues (priority order):
  1. **IEEE IoTJ** (companion, with disclosure)
  2. **IEEE TIFS** (physical-layer security angle)
  3. **IEEE TWC** (wireless ML / domain shift)
- Avoid verbatim text overlap (see `PAPER2_OVERLAP_AUDIT.md`)

**Do not:** merge into Paper 1 at this stage.

---

## Case 2: Paper 1 major revision

**Action:**
- **Pause Paper 2 submission**
- Address Paper 1 reviewer comments first
- Keep Paper 2 skeleton and frozen results ready
- Reassess positioning after revision:
  - If reviewers ask for cross-receiver solution → mention Paper 2 as companion/in-progress
  - If reviewers want cross-receiver inside Paper 1 → **do not** auto-merge; evaluate page limit and narrative cost

---

## Case 3: Paper 1 rejected

**Action:** analyze rejection reasons before choosing path.

| Rejection reason | Paper 2 strategy |
|------------------|------------------|
| Cross-receiver gap / incomplete evaluation | Consider **merged resubmission** (architecture + diagnosis + RCPA) to IoTJ or TWC |
| Architecture novelty insufficient | Paper 2 (diagnosis + RCPA) may become **primary thesis paper** |
| Scope / fit mismatch | Split venues: Paper 1 → TWC/TVT; Paper 2 → TIFS/Sensors |
| Experimental depth | Expand Paper 2 experiments per reviewer themes; do not rewrite RCPA core |

**Merge criteria (strict):** only if rejection explicitly requests unified cross-receiver story AND page/narrative allow clean integration without redundant publication concerns.

---

## Checklist before any Paper 2 submission

- [ ] Paper 1 status confirmed
- [ ] Cover letter Version A or B selected
- [ ] `PAPER2_OVERLAP_AUDIT.md` reviewed (risk Medium → mitigations applied)
- [ ] Abstract/intro rewritten (no Paper 1 copy)
- [ ] Architecture figure NOT reused
- [ ] TTA threshold appendix included
- [ ] Human approval on target venue
