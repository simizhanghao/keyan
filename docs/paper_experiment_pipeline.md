# IoTJ Paper Experiment Pipeline

Git commit at setup: `b030d34`

## Quick Start

```bash
cd /data1/hcc/llm4RF
export PY=/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python
export CUDA_VISIBLE_DEVICES=0

# Phase 1 only (manifests + audit)
bash scripts/paper/phase1_manifests.sh

# All phases (long-running)
bash scripts/paper/run_all_phases.sh

# Or individual phases
PHASES=2 bash scripts/paper/run_all_phases.sh
```

## Protocol: Source-Only vs Oracle

| Protocol | Manifest suffix | train | val (checkpoint) | test (report) |
|----------|-----------------|-------|------------------|---------------|
| **source_only** (paper main) | `*_source_only.csv` | source domain | source domain (Day4 / RX1) | target (Day5 / RX2) |
| **oracle_target_val** (diagnostic) | `*_oracle_target_val.csv` | source | **target labels** | target |

Cross-day source-only Day1-4→Day5:
- train: Day1,2,3 (72 files)
- val: Day4 (24 files) — early stopping only
- test: Day5 (24 files) — **paper metrics**

Cross-receiver source-only RX1→RX2:
- train: RX1 (24 files)
- val: RX1 (24 files, deterministic windows)
- test: RX2 (24 files)

## Phase Scripts

| Phase | Script | Output |
|-------|--------|--------|
| 1 | `phase1_manifests.sh` | `data/paper/`, `manifest_audit.csv` |
| 2 | `phase2_cross_day.sh` | `outputs/paper_runs/phase2_*` |
| 3 | `phase3_ablation.sh` | ablation A–G |
| 4 | `phase4_deployment.sh` | Config/Location/Distance |
| 5 | `phase5_cross_receiver.sh` | source-only, upper bound, CORAL+IM |
| 6 | `phase6_edge_benchmark.py` | `edge_deployment_summary.csv` |

Aggregate: `python scripts/paper/aggregate_paper_ready.py`

## Code Changes for Strict Protocol

- `train_utils.py`: `--train-split`, `--val-split`, `--eval-split`
- `evaluate.py`: evaluates `--eval-split test` by default for paper manifests
- `scripts/paper/generate_paper_manifests.py`: all paper manifests

## Per-Run Outputs (evaluate.py)

- `metrics.json`, `predictions.csv`, `confusion_matrix.csv`
- `per_device_accuracy.csv`, `per_domain_accuracy.csv`, `run_config.json`

## Edge Deployment (Phase 6 preliminary)

| Model | Params | Latency bs1 (ms) | Peak GPU MB |
|-------|--------|------------------|-------------|
| CNN-IQ | 47,705 | ~1.1 | ~66 |
| Hybrid | 1,156,022 | ~2.5 | ~124 |

## Do NOT Use for Paper Main Tables

- `outputs/cross_receiver_*` with oracle target-val checkpoint selection
- Mixed 30ep / 80ep without labeling
- Day1→Day2 mislabeled as Day1-4→Day5
