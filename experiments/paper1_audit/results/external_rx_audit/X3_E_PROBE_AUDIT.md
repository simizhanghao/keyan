# X3-E frozen embedding probes

Embeddings were averaged over 256 packets per `receiver x device` cell. Device
probe splits receiver groups; receiver probe splits device groups, so the probe
does not treat packets as independent experiments.

| model | device probe mean | receiver probe mean | receiver chance |
|---|---:|---:|---:|
| B1 | **94.4%** | 16.2% | 7.7% |
| C' | **97.9%** | 14.7% | 7.7% |

The strong device probe and modest receiver probe support the interpretation
that OOB contains device-discriminative evidence while the shortcut is not
simply a receiver classifier. X3 is therefore `MECHANISM GO`; F0/F0-CT can now
be opened, but official blind receivers remain sealed.
