#!/usr/bin/env python3
"""Lightweight LaTeX structure checks for docs/iotj_paper/."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def audit_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []

    pairs = [
        ("figure", r"\begin{figure}", r"\end{figure}"),
        ("table", r"\begin{table}", r"\end{table}"),
        ("table*", r"\begin{table*}", r"\end{table*}"),
    ]
    for name, begin, end in pairs:
        b, e = text.count(begin), text.count(end)
        if b != e:
            issues.append(f"{name} env mismatch: {b} begin vs {e} end")

    for block in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", text, flags=re.S):
        if r"\input{tables" in block.group(0):
            issues.append("table \\input nested inside figure environment")

    for block in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, flags=re.S):
        if r"\begin{figure}" in block.group(0):
            issues.append("figure nested inside table environment")

    return issues


def audit_sections_inputs() -> list[str]:
    issues: list[str] = []
    for path in sorted((ROOT / "sections").glob("*.tex")):
        in_figure = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if r"\begin{figure}" in line:
                in_figure = True
            if r"\end{figure}" in line:
                in_figure = False
            if r"\input{tables" in line and in_figure:
                issues.append(f"{path.name}:{lineno}: table input inside figure")
    return issues


def audit_forbidden_subsections() -> list[str]:
    issues: list[str] = []
    method = (ROOT / "sections" / "04_method.tex").read_text(encoding="utf-8")
    if r"\subsection{Implementation Details}" in method:
        issues.append("04_method.tex must not contain Implementation Details")
    if r"\subsection{Overview}" not in method:
        issues.append("04_method.tex missing Overview subsection")
    overview_pos = method.find(r"\subsection{Overview}")
    section_pos = method.find(r"\section{Proposed Method}")
    if overview_pos == -1 or overview_pos < section_pos:
        issues.append("Overview must appear after Proposed Method section heading")
    return issues


def main() -> int:
    failed = False

    for path in sorted(ROOT.rglob("*.tex")):
        issues = audit_file(path)
        if issues:
            failed = True
            print(f"FAIL {path.relative_to(ROOT)}: {issues}")
        else:
            print(f"OK   {path.relative_to(ROOT)}")

    for msg in audit_sections_inputs() + audit_forbidden_subsections():
        failed = True
        print(f"FAIL {msg}")

    if failed:
        print("STRUCTURE AUDIT: FAIL")
        return 1

    print("STRUCTURE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
