# Cross-Receiver Calibration (RCPA)

Receiver-Calibrated Prototype Adaptation for LoRa RFFI cross-receiver transfer.

## Quick mode

```bash
cd /data1/hcc/llm4RF
GPU_ID=1 bash experiments/cross_receiver_calibration/run_calibration_grid.sh --quick
```

## Protocol

- **256 deterministic windows** per target `.dat` file
- **Block-disjoint split:** A=calibration, B=support, C+D=query
- **K-shot** = K labeled **windows** per device (not K files)
- Overlap checks enforced at split build time

## Methods

| Method | Description |
|--------|-------------|
| source_classifier | Phase5 classifier head |
| RCPA-S | alpha=1.0, source prototype only |
| RCPA-T | alpha=0.0, K-shot target prototype |
| RCPA-B | alpha=0.5, fixed blend |

## Outputs

```text
results/quick_*/
├── support_query_split.csv
├── summary_quick.csv
├── shot_curve_rx1_to_rx2_quick.csv
├── fig_shot_curve_quick.pdf
└── embeddings_fused.npz  (gitignored)
```

See `CALIBRATION_REPORT.md` after quick run.
