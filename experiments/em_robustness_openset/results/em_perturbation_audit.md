# EM Perturbation Audit

Physical interpretation of LoRa IQ perturbations for thesis Chapter 5.

| Perturbation | Physical meaning | Strength range | Train | Test |
|--------------|------------------|----------------|-------|------|
| awgn_snr_db | Thermal/environmental noise (AWGN). | [30.0, 20.0, 15.0, 10.0, 5.0, 0.0] | yes (moderate) | yes (full sweep) |
| narrowband_sir_db | Adjacent-channel or co-channel narrowband interference. | [30.0, 20.0, 10.0, 5.0, 0.0] | yes (moderate) | yes (full sweep) |
| cfo_norm | Carrier frequency offset from oscillator / sync error (normalized to LoRa BW). | [0.0, 0.01, 0.03, 0.05, 0.1] | yes (moderate) | yes (full sweep) |
| phase_noise_std | Oscillator phase jitter (random-walk phase noise). | [0.0, 0.01, 0.03, 0.05, 0.1] | yes (moderate) | yes (full sweep) |
| iq_amp_db | I-branch gain mismatch in the receiver IQ chain. | [0.0, 1.0, 3.0, 5.0] | yes (moderate) | yes (full sweep) |
| iq_phase_deg | I/Q orthogonality error in the receiver front-end. | [0.0, 2.0, 5.0, 10.0] | yes (moderate) | yes (full sweep) |
| filter_tilt_norm | Receiver filter / cable spectral tilt across the observed band. | [0.0, 0.1, 0.2, 0.4] | yes (moderate) | yes (full sweep) |

## Mixed stress presets

- **awgn_cfo**: {'awgn_snr_db': 10.0, 'narrowband_amplitude': None, 'narrowband_freq_hz': None, 'narrowband_sir_db': None, 'cfo_hz': None, 'cfo_norm': 0.05, 'phase_noise_std': None, 'iq_imbalance_alpha': None, 'iq_imbalance_beta': None, 'iq_amp_db': None, 'iq_phase_deg': None, 'filter_tilt_db': None, 'filter_tilt_norm': None, 'sample_rate': 1000000.0, 'lora_bandwidth': 125000.0}
- **awgn_nbi**: {'awgn_snr_db': 10.0, 'narrowband_amplitude': None, 'narrowband_freq_hz': None, 'narrowband_sir_db': 10.0, 'cfo_hz': None, 'cfo_norm': None, 'phase_noise_std': None, 'iq_imbalance_alpha': None, 'iq_imbalance_beta': None, 'iq_amp_db': None, 'iq_phase_deg': None, 'filter_tilt_db': None, 'filter_tilt_norm': None, 'sample_rate': 1000000.0, 'lora_bandwidth': 125000.0}
- **cfo_iq**: {'awgn_snr_db': None, 'narrowband_amplitude': None, 'narrowband_freq_hz': None, 'narrowband_sir_db': None, 'cfo_hz': None, 'cfo_norm': 0.05, 'phase_noise_std': None, 'iq_imbalance_alpha': None, 'iq_imbalance_beta': None, 'iq_amp_db': 3.0, 'iq_phase_deg': 5.0, 'filter_tilt_db': None, 'filter_tilt_norm': None, 'sample_rate': 1000000.0, 'lora_bandwidth': 125000.0}

## Notes

- AWGN: thermal / environmental noise.
- Narrowband interference: adjacent-channel or co-channel interferer.
- CFO: oscillator frequency offset / sync error.
- Phase noise: oscillator phase jitter.
- IQ imbalance: receiver front-end orthogonality and gain mismatch.
- Filter drift: receiver filter / cable frequency response tilt.