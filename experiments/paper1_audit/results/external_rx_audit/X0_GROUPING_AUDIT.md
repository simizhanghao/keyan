# X0 grouping and exact-overlap audit

training=false; official_blind_signal_opened=false

| File role | Data shape | Labels | File attrs | Dataset attrs | Unique IQ rows |
|---|---:|---:|---:|---:|---:|
| source_train | `[8000, 16384]` | 10 | 0 | 0 | 8000 |
| seen_source_test | `[2000, 16384]` | 10 | 0 | 0 | 2000 |
| drift_day1_train | `[8000, 16384]` | 10 | 0 | 0 | 8000 |

## Exact IQ-row overlap

- source train vs seen-source test: **0**
- source train vs drift day-1 train: **0**

## Locked interpretation

- The audited files contain packet rows and device labels but no capture/session IDs or attributes.
- Therefore capture-level independence cannot be asserted or reconstructed; no synthetic grouping key will be invented.
- Later reporting must aggregate over receiver domains/seeds and clearly label packet-level accuracy/macro-F1; capture/session inference is disallowed unless new metadata is supplied.
- Zero exact-row overlap is a duplicate-leakage check only, not proof that packets are statistically independent.
- Only N210_1 source/seen and source-versus-drift-day1 were checked here; the six official blind receiver signals remain sealed until X6.
