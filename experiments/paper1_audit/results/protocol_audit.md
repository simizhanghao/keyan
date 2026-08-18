# Protocol audit (1A)

- keyan: `/data1/hcc/llm4RF/new_phase`
- data-root: `/data1/hcc/llm4RF`
- gate_1A: **PASS**
- protocol_matches_lock: True
- dat present/missing: 120 / 0

## Issues

- none

## Splits

```json
{
  "train": {
    "n_files": 72,
    "n_unique_paths": 72,
    "days": [
      "1",
      "2",
      "3"
    ],
    "n_labels": 24,
    "n_devices": 24,
    "files_per_day": {
      "1": 24,
      "2": 24,
      "3": 24
    },
    "device9_count": 0,
    "dat_present": 72,
    "dat_missing": 0
  },
  "val": {
    "n_files": 24,
    "n_unique_paths": 24,
    "days": [
      "4"
    ],
    "n_labels": 24,
    "n_devices": 24,
    "files_per_day": {
      "4": 24
    },
    "device9_count": 0,
    "dat_present": 24,
    "dat_missing": 0
  },
  "test": {
    "n_files": 24,
    "n_unique_paths": 24,
    "days": [
      "5"
    ],
    "n_labels": 24,
    "n_devices": 24,
    "files_per_day": {
      "5": 24
    },
    "device9_count": 0,
    "dat_present": 24,
    "dat_missing": 0
  }
}
```

## Oracle (forbidden for development)

```json
{
  "path": "data/paper/cross_day_day1to5_oracle_target_val.csv",
  "exists": true,
  "val_days": [
    "5"
  ],
  "test_days": [
    "5"
  ],
  "leaks_day5_into_val": true,
  "role": "FORBIDDEN for Experiment 1 development"
}
```

## LODO (sealed until 1E)

```json
[
  {
    "path": "data/paper/lodo_source_only/test_day_1.csv",
    "exists": true,
    "train_days": [
      "2",
      "3",
      "4"
    ],
    "val_days": [
      "5"
    ],
    "test_days": [
      "1"
    ],
    "uses_day5_in_val": true,
    "uses_day5_in_train": false,
    "sealed_until": "1E"
  },
  {
    "path": "data/paper/lodo_source_only/test_day_2.csv",
    "exists": true,
    "train_days": [
      "1",
      "3",
      "4"
    ],
    "val_days": [
      "5"
    ],
    "test_days": [
      "2"
    ],
    "uses_day5_in_val": true,
    "uses_day5_in_train": false,
    "sealed_until": "1E"
  },
  {
    "path": "data/paper/lodo_source_only/test_day_3.csv",
    "exists": true,
    "train_days": [
      "1",
      "2",
      "4"
    ],
    "val_days": [
      "5"
    ],
    "test_days": [
      "3"
    ],
    "uses_day5_in_val": true,
    "uses_day5_in_train": false,
    "sealed_until": "1E"
  },
  {
    "path": "data/paper/lodo_source_only/test_day_4.csv",
    "exists": true,
    "train_days": [
      "1",
      "2",
      "3"
    ],
    "val_days": [
      "5"
    ],
    "test_days": [
      "4"
    ],
    "uses_day5_in_val": true,
    "uses_day5_in_train": false,
    "sealed_until": "1E"
  },
  {
    "path": "data/paper/lodo_source_only/test_day_5.csv",
    "exists": true,
    "train_days": [
      "1",
      "2",
      "3"
    ],
    "val_days": [
      "4"
    ],
    "test_days": [
      "5"
    ],
    "uses_day5_in_val": false,
    "uses_day5_in_train": false,
    "sealed_until": "1E"
  }
]
```

## Hashes

See `/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/manifest_hashes.json`.

Frozen `outputs/paper_ready_v3/` was not written.

