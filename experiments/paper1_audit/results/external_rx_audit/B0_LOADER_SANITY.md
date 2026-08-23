# B0 loader sanity

**Verdict: PASS.** No training and no official blind receiver access.

All 14 source/train HDF5 files were checked on 8 packets per file against the
released `LoadDataset` / `ChannelIndSpectrogram` convention:

- IQ storage `[real(8192), imag(8192)]` reconstructed as complex IQ;
- per-packet RMS normalization;
- boxcar STFT, `win_len=128`, overlap `64`, two-sided FFT;
- adjacent-frame channel-independent spectrum ratio and `crop_ratio=0.3`;
- output shape `[8, 52, 126]`, finite for every file;
- SNR/CFO metadata finite for every file;
- all 14 source receiver files have the expected 10-device label set.

The raw HDF5 labels are `31..40`; the released loader subtracts one and the
X2 implementation must explicitly remap them to contiguous `0..9` labels.

Machine-readable result: `b0_loader_sanity.json`.
