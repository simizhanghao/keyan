# Final model manifest

Status: schema and roles locked after X4-C formal and before X6.

| Model | Role | Input | Blind status |
|---|---|---|---|
| B1-OOB | matched convolutional OOB probe | multi-view IQ/FFT/AP/OOB | headline |
| C'-OOB | attention OOB probe | IQ + audited OOB fusion | headline |
| C'-TrueIB | strict in-band control | raw-IQ band-limited regenerated views | headline |
| Shen-CIS | prior CIS baseline | native 52x126 CIS | headline |
| Shen-RA | receiver-agnostic prior port | native 52x126 CIS + GRL RX head | headline |
| B1-TrueIB | supplementary control | raw-IQ band-limited regenerated views | supplement |

RSA/F0 is excluded from blind evaluation.
