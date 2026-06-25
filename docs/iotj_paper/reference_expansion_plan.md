# Reference Expansion Plan

## Goal

Expand IoTJ references from the **current 15 in-text cites** to about **25–35 verified references**, primarily in Related Work and Introduction.

## Current inventory (2025-06-25 audit)

| Pool | Count | Notes |
|------|-------|-------|
| Unique keys cited in `.tex` | **15** | All READ or ABSTRACT_CHECKED |
| Entries in `refs.bib` | **26** | Includes uncited prep entries |
| Rows in `reference_candidates.csv` | **35** (after snowball append) | Includes TO_READ / EXCLUDE |

### Currently cited (15)

`elmaghbub2021lora_wild`, `hamdaoui2022deeplearning`, `elmaghbub2021dataset_release`, `elmaghbub2020widescan`, `shen2021lora_spectrogram_cnn`, `jian2020deeplearning_rf_fingerprinting`, `vaswani2017attention`, `shen2022scalable_lora_rffi`, `ma2025contrastive_lora_rffi`, `ma2025receiver_independent_rffi`, `wen2016center_loss`, `khosla2020supcon`, `wang2021tent`, `lim2023ttn`, `lorawan_spec`

### In `refs.bib` but NOT cited (11) — promotion candidates

| bibkey | csv status | Suggested action |
|--------|------------|------------------|
| `zhou2021robust_rffi_iotj` | TO_READ | Promote after abstract skim; IoTJ RFFI robustness anchor |
| `zhang2019phy_layer_security_iot` | TO_READ | Promote to ABSTRACT_CHECKED; PHY IoT auth intro |
| `yang2017iot_security_survey` | TO_READ | Promote to ABSTRACT_CHECKED; IoT security survey |
| `xu2015device_fingerprinting_survey` | TO_READ | Promote; general RFFI survey |
| `riyaz2018deep_cnn_radio_id` | TO_READ | Promote; CNN radio ID lineage |
| `sankhe2019oracle` | TO_READ | Promote; ORACLE CNN baseline |
| `alshawabka2020exposing_fingerprint` | TO_READ | Promote; channel vs fingerprint confound |
| `cekic2020robust_wireless_fingerprinting` | TO_READ | Promote; domain generalization |
| `hamdaoui2020deep_nn_features` | TO_READ | Promote; OSU feature-design prior art |
| `brik2008wireless_device_id` | TO_READ | Promote; classic radiometric ID |
| `zanella2014iot_smart_cities` | TO_READ | Optional; IoT motivation only |

**Do not cite until status → READ or ABSTRACT_CHECKED.**

## Rules

1. Use existing papers' reference lists only as discovery sources.
2. Do not cite unread papers directly in `.tex`.
3. Each new citation must have: bibkey, title, venue/year, DOI/arXiv/URL, status, why_cite, target section.
4. TO_READ entries must not enter `\cite{}` until verified.

## Needed categories (target density)

| Category | Target in Related Work | Current cited | Gap |
|----------|------------------------|---------------|-----|
| LoRaWAN / IoT security background | 2–3 | 1 (`lorawan_spec`) | +1–2 |
| PHY auth / RFFI surveys | 2–3 | 0 dedicated surveys | +2–3 |
| LoRa RFFI / OSU deployment | 4–6 | 4 | OK; add dataset release in §III if desired |
| CNN-IQ / spectrogram baselines | 4–6 | 2 | +2–4 (`riyaz`, `sankhe`, `oracle`, channel paper) |
| OOB / hardware impairment | 3–5 | 2 | +1–2 (`hamdaoui2020`, `zhou2021`) |
| RF sequence / attention | 2–4 | 1 | +1 optional RF-sequence prior art |
| Domain shift / TTA / metric learning | 4–6 | 6 | OK; avoid stacking more 2025 arXiv |

## Citation density by section (current)

| Section | Cite count | Target |
|---------|------------|--------|
| `01_introduction.tex` | 8 (6 unique) | 4–6 ✓ |
| `02_related_work.tex` | 19 (15 unique) | 15–25 ✓ (lower bound met; thin vs IoTJ norm) |
| `03_system_model.tex` | 0 | 0–2 optional dataset refs |
| `04_method.tex` | 3 | few ✓ |
| `05_experiments.tex` | 3 | few ✓ |
| `06–08` | 0 | no new cites ✓ |

## Recommended promotion order (GPT decision)

**Tier A — high value, metadata verifiable without full PDF:**

1. `yang2017iot_security_survey` → Intro/Related IoT security paragraph
2. `zhang2019phy_layer_security_iot` → Intro PHY authentication
3. `xu2015device_fingerprinting_survey` → Related Work opening survey sentence
4. `sankhe2019oracle` + `riyaz2018deep_cnn_radio_id` → CNN baseline lineage
5. `alshawabka2020exposing_fingerprint` → domain-shift / channel confound

**Tier B — read abstract or skim PDF before cite:**

6. `zhou2021robust_rffi_iotj` (IoTJ venue reference)
7. `hamdaoui2020deep_nn_features` (OSU feature design)
8. `cekic2020robust_wireless_fingerprinting` (generalization framing)
9. `brik2008wireless_device_id` (classic radiometric)

**Tier C — new snowball TO_READ (in csv, not in bib yet):**

10. `abbas2021iot_rf_fingerprinting_security` — IoT RF fingerprinting ID system
11. `lalouani2021countering_radiometric` — radiometric signature / adversarial context (limitation)

**Do NOT cite in Related Work for leaderboard comparison:**

- `ma2025contrastive_lora_rffi`, `ma2025receiver_independent_rffi`, `shen2022scalable_lora_rffi` — already cited as motivation only; keep wording conservative.

## 2025 / arXiv verification notes

| bibkey | Check |
|--------|-------|
| `ma2025contrastive_lora_rffi` | WCNC 2025; verify proceedings entry before camera-ready |
| `ma2025receiver_independent_rffi` | arXiv:2512.12070; preprint; cite as arXiv + note WCNC partial presentation per reading notes |
| `shen2021` / `shen2022` | arXiv notes match published venues |
| `cekic2020` | arXiv:2002.10791 only; prefer noting preprint status |

## Page impact estimate

- Adding **8–12** cites with **0 new paragraphs**: +0.3–0.6 pages (bibliography only).
- Expanding Related Work by **1 short paragraph** + cites: +0.5–1.0 pages.
- Current PDF ~**11 pages** → after refs expansion likely **11.5–12.5 pages** unless layout compression applied.

## Next step (GPT)

Review Tier A/B/C list → approve bibkeys → agent adds to `refs.bib` + `\cite{}` in `02_related_work.tex` / `01_introduction.tex` only → re-run validators → recompile PDF.
