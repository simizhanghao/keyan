#!/usr/bin/env python3
"""Audit PDF-source layout rules: no draft appendix, no internal paths, required figure labels."""

from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
texts: list[tuple[Path, str]] = []
for p in [root / "main.tex"] + list((root / "sections").glob("*.tex")) + list((root / "tables").glob("*.tex")):
    if p.exists():
        texts.append((p, p.read_text(encoding="utf-8", errors="ignore")))
all_text = "\n".join(t for _, t in texts)

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append("ERROR: " + msg)


def warn(msg: str) -> None:
    warnings.append("WARN: " + msg)


# Hard bans anywhere in PDF source
for pat, msg in [
    ("draft block diagram", "draft block diagram remains"),
    ("paper-ready-v3", "branch name paper-ready-v3 should not appear in PDF source"),
    ("outputs/paper_ready_v3", "internal output path should not appear in PDF source"),
    ("PAPER_RESULTS_SUMMARY", "internal results summary path should not appear in PDF source"),
    ("fig:cross_day_seed_bars", "old seed figure label remains"),
    ("fig:fusion_chirp_ablation", "old ablation figure label remains"),
    ("fig:distance_shift", "old distance figure label remains"),
    (r"\resizebox{0.98\columnwidth}", "resizebox 0.98 columnwidth remains"),
]:
    if pat in all_text:
        err(msg)

# Appendix reproducibility block (not Data and Code Availability)
if re.search(r"\\appendices[\s\S]*?\\section\{Reproducibility\}", all_text):
    err("Appendix Reproducibility section must be removed")
if re.search(r"\\section\{Reproducibility\}", all_text):
    err("Raw Reproducibility section must be replaced by Data and Code Availability")

# Required labels
for label in ["fig:architecture", "fig:results_summary", "fig:cross_receiver_stress"]:
    if label not in all_text:
        err(f"required figure label missing: {label}")

if "Data and Code Availability" not in all_text:
    warn("Data and Code Availability section not found in PDF source")

print("LAYOUT VISUAL AUDIT")
for w in warnings:
    print(" - " + w)
for e in errors:
    print(" - " + e)

if errors:
    print("LAYOUT VISUAL AUDIT: FAIL")
    sys.exit(1)
print("LAYOUT VISUAL AUDIT: PASS")
