#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
tex_files = list((root / "tables").glob("*.tex")) + list((root / "sections").glob("*.tex")) + [root / "main.tex"]

errors = []
warnings = []


def add_error(msg):
    errors.append("ERROR: " + msg)


def add_warn(msg):
    warnings.append("WARN: " + msg)


for p in tex_files:
    s = p.read_text(encoding="utf-8", errors="ignore")
    if "\\begin{table" not in s:
        continue

    table_blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", s, flags=re.S)
    for idx, block in enumerate(table_blocks, 1):
        label = re.search(r"\\label\{([^}]+)\}", block)
        label = label.group(1) if label else f"{p.name}#{idx}"

        begins = len(re.findall(r"\\begin\{table\*?\}", block))
        ends = len(re.findall(r"\\end\{table\*?\}", block))
        if begins != 1 or ends != 1:
            add_error(f"{label}: nested or malformed table env in {p}")

        is_star = "\\begin{table*}" in block
        if "\\resizebox" not in block and "\\begin{tabularx}" not in block:
            add_warn(f"{label}: no resizebox/tabularx guard; check column overflow manually")

        if "Window-Macro-F1" in block:
            add_warn(f"{label}: long header Window-Macro-F1 should be shortened to W-F1")

        if "Label smoothing" in block:
            add_warn(f"{label}: long header Label smoothing should be shortened to LS")

        if "Held-out condition" in block:
            add_warn(f"{label}: long header Held-out condition may overflow; consider Held-out")

        if not is_star and "\\resizebox{0.98\\columnwidth}" not in block and "\\resizebox{0.90\\columnwidth}" not in block:
            if "\\resizebox{\\columnwidth}" in block:
                add_warn(f"{label}: uses full \\columnwidth; prefer 0.98\\columnwidth margin")

print("TABLE LAYOUT AUDIT")
for w in warnings:
    print(" - " + w)
for e in errors:
    print(" - " + e)

if errors:
    print("AUDIT RESULT: FAIL")
    sys.exit(1)

print("AUDIT RESULT: PASS")
