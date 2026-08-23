# B1 late-fusion CNN smoke

**Verdict: implementation contract PASS.** This is not an accuracy result.

The B1 control is a plain multi-view late-fusion CNN. IQ, FFT, amplitude/phase,
and OOB are encoded by separate convolutional branches and concatenated only
after global embedding. It has no sequence-attention, HSTU block, gating,
receiver-adversarial head, or F0 augmentation.

Static checks passed for 14 source receivers, no official blind receiver paths,
four required views, late embedding fusion, and a 10-class output contract.
Runtime forward requires the project's PyTorch environment and is deferred to
the nominal P1 seed-0 training job.
