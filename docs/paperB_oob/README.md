# Paper B: IEEE Manuscript Package

Title: *Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification*

This package is an IEEEtran double-column English manuscript. The author order is fixed as Ziyang Wang (first author) and Chengcheng Han (corresponding author). The funding statement and biographies use the supplied author information.

## Evidence boundary

The paper reports the frozen 20-SDR Paper-B protocol. It does not claim authentication, open-set rejection, universal receiver invariance, or a new backbone architecture. The official blind X6 results are stored under `new_phase/experiments/paper1_audit/results/x6_blind/` and were aggregated with receiver as the top-level unit.

## Build

Use a TeX installation with `IEEEtran`, `booktabs`, `multirow`, `cite`, `url`, and `hyperref`:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current container has no LaTeX engine, so PDF compilation must be run in the author's TeX/Overleaf environment. Figure source generation is reproducible with:

```text
python3 scripts/make_figures.py
```

## Result allocation

Main text retains the shortest evidence chain: signal-level OOB/device structure, development interventions, architecture and strong-baseline controls, mitigation boundary, and six-receiver blind confirmation. Per-seed predictions and full intervention outputs remain in the frozen X6 result directory as source data. The receiver, not the packet, is the inferential unit.

## Nature-skill audit applied

The installed Nature writing skills were used for claim-evidence alignment, terminology consistency, main-text compression, statistical-unit discipline, figure planning, citation hygiene, and reviewer-risk checks. IEEE formatting and IEEE reference style take precedence over Nature layout conventions.
