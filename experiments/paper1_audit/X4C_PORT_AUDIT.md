# X4-C Shen-RA Adaptation Audit

The released reference is TensorFlow/Keras. This implementation must be called
**Shen-RA (Shen-style PyTorch adaptation)**, never an exact official
reproduction.

| Component | Reference | Port requirement |
|---|---|---|
| Input | 52 x 126 Channel-Independent Spectrogram | exact CIS preprocessing |
| Backbone | Conv2D 32, residual 32/64 blocks, average pool | layer-for-layer equivalent |
| Feature | Dense 512 + L2 normalization | exact |
| Heads | 10-class TX + receiver head | exact class roles |
| Adversarial objective | TX CE + RX CE, GRL, weights 1:1 | exact; no lambda sweep |
| Optimizer | SGD, lr 1e-3, momentum .9 | exact nominal recipe |
| Schedule | max 500, early stop 20, LR reduction | preserve, while using leakage-safe source validation |
| Protocol | author split | replace only with receiver-held-out source validation |

`Shen-CIS` (no receiver head) and `Shen-RA` (GRL receiver head) are the only
two planned arms. Source receiver labels are allowed and must be disclosed.

## Port preflight

The PyTorch adaptation produced a valid 52 x 126 CIS tensor and completed a two-epoch
Shen-RA smoke on `rtl_2`, seed 0. This smoke is an implementation check only,
not a reported baseline result. The registered formal arms are now running
under `X4C_PROTOCOL_LOCK.md`.
