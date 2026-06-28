# RUN_MANIFEST — Phase B/D EM robustness + open-set full

Generated: 2026-06-28T13:07:04

## Git
- branch: `thesis-em-openset`
- commit: `9ff8b47`

## Environment
- python: `/new_nfs/haiyu/anaconda3/bin/python`
- cuda: `True 2.9.1+cu128`

## Model & data
- checkpoint: `outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt`
- model: RF-HSTU / F_cross_attn_chirp_plain
- model seed: 0
- windows per file: 256
- eval split: test (Day5)
- manifest: `data/paper/cross_day_day1to5_source_only.csv`

## GPU allocation (2026-06-28 run)
- GPU1: AWGN curve
- GPU2: CFO curve
- GPU3: narrowband
- GPU4: phase noise + IQ amp
- GPU5: filter drift
- GPU6: open-set full (3 seeds)

## Perturbation grids
- AWGN SNR (dB): 100, 40, 30, 25, 20, 15, 10, 5, 0
- CFO norm: 0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10
- Narrowband SIR (dB): 30, 20, 10, 5, 0
- Phase noise σ: 0, 0.01, 0.03, 0.05, 0.10
- IQ amp (dB): 0, 1, 3, 5
- Filter tilt norm: 0, 0.1, 0.2, 0.4

## Open-set
- known devices: 20
- unknown devices: 4
- split seeds: 0, 1, 2
- output: `experiments/em_robustness_openset/results/openset_full_20260628_1123`

## Logs
- `experiments/em_robustness_openset/results/em_full_20260628/logs/awgn_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/awgn_snr_db.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/cfo_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/cfo_norm.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/filter_mixed_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/filter_tilt_norm.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/iq_amp_db.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/narrowband_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/narrowband_sir_db.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/openset_full_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/phase_iq_nohup.log`
- `experiments/em_robustness_openset/results/em_full_20260628/logs/phase_noise_std.log`
