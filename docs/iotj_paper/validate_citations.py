#!/usr/bin/env python3
"""Validate IoTJ paper citations against reference_candidates.csv."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT
CAND = PAPER / "reference_candidates.csv"
BIB = PAPER / "refs.bib"

ALLOWED = {"READ", "ABSTRACT_CHECKED"}


def load_allowed_keys() -> dict[str, str]:
    out = {}
    with CAND.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["bibkey"]] = row["status"]
    return out


def load_bib_keys() -> set[str]:
    text = BIB.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,]+),", text))


def load_cite_keys() -> set[str]:
    keys: set[str] = set()
    for tex in PAPER.rglob("*.tex"):
        if tex.name == "reference_candidates.csv":
            continue
        text = tex.read_text(encoding="utf-8")
        for block in re.findall(r"\\cite\{([^}]+)\}", text):
            for k in block.split(","):
                keys.add(k.strip())
    return keys


def main() -> int:
    status = load_allowed_keys()
    bib = load_bib_keys()
    cites = load_cite_keys()

    errors = []
    for k in sorted(cites):
        if k not in bib:
            errors.append(f"MISSING in refs.bib: {k}")
        elif k in status and status[k] not in ALLOWED:
            errors.append(f"DISALLOWED status {status[k]}: {k}")
        elif k not in status:
            errors.append(f"MISSING in reference_candidates.csv: {k}")

    unused_allowed = [k for k, s in status.items() if s in ALLOWED and k not in cites]
    print(f"Citations in tex: {len(cites)}")
    print(f"refs.bib entries: {len(bib)}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(" ", e)
        return 1
    print("OK: all \\cite keys are in refs.bib and READ/ABSTRACT_CHECKED")
    print(f"Unused allowed keys ({len(unused_allowed)}): {', '.join(unused_allowed[:10])}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
