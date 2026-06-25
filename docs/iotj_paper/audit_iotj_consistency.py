#!/usr/bin/env python3
from pathlib import Path
import re
import sys
import csv

ROOT = Path(__file__).resolve().parent
SECTIONS = ROOT / "sections"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
MAIN = ROOT / "main.tex"
REF_CSV = ROOT / "reference_candidates.csv"
REF_BIB = ROOT / "refs.bib"

tex_files = [MAIN] + sorted(SECTIONS.glob("*.tex")) + sorted(TABLES.glob("*.tex"))
all_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in tex_files if p.exists())

errors = []
warnings = []

def err(msg):
    errors.append("ERROR: " + msg)

def warn(msg):
    warnings.append("WARN: " + msg)

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))

# ---------------------------------------------------------------------
# 1. Forbidden / risky claims
# ---------------------------------------------------------------------
forbidden_patterns = [
    (r"\breceiver[- ]invariant\b", "Forbidden claim: receiver-invariant"),
    (r"\bsolves? cross[- ]receiver\b", "Forbidden claim: solves cross-receiver"),
    (r"\brobust to all\b", "Forbidden overclaim: robust to all shifts"),
    (r"\bfully robust\b", "Forbidden overclaim: fully robust"),
    (r"\bchirp.*primary\b", "Risky claim: chirp as primary gain"),
    (r"\bchirp.*main gain\b", "Risky claim: chirp as main gain"),
    (r"\bconcat.*final\b", "Forbidden: concat as final method"),
    (r"\bH_gated_chirp_plain\b", "Forbidden: old gated baseline H_gated_chirp_plain mentioned as paper result"),
    (r"\bgated\s+OOB\s+hybrid\s+.*proposed\s+final\b", "Forbidden: gated OOB hybrid described as proposed final method"),
    (r"\bgated\s+fusion\s+.*proposed\s+final\b", "Forbidden: gated fusion described as proposed final method"),
    (r"\bM7\b", "Deprecated model M7 mentioned"),
    (r"\btarget[- ]val\b", "Deprecated target-val protocol mentioned"),
    (r"\bLODO\b", "Old LODO protocol mentioned"),
    (r"\b87\.5\%", "Old Phase3 concat 87.5% appears"),
    (r"\b58\.3\%", "Old target-val cross-receiver number 58.3% appears"),
    (r"\b74\.6\%", "Old single-run 74.6% number appears"),
    (r"\b60\.7\%", "Old LODO 60.7% number appears"),
]

for pat, msg in forbidden_patterns:
    matches = []
    for m in re.finditer(pat, all_text, flags=re.IGNORECASE):
        span_start = max(0, m.start() - 80)
        context = all_text[span_start:m.end()].lower()
        # Final method uses gated residual injection; not the old gated fusion baseline.
        if "gated residual injection" in context:
            continue
        if pat == r"\breceiver[- ]invariant\b":
            negation_markers = (
                "not ",
                "rather than",
                "does not claim",
                "should not be interpreted",
                "no evidence of",
                "not evidence of",
                "not interpreted as",
                "solved receiver-invariant",
            )
            if any(marker in context for marker in negation_markers):
                continue
        matches.append(m)
    if matches:
        if "Risky" in msg:
            warn(f"{msg}; occurrences={len(matches)}")
        else:
            err(f"{msg}; occurrences={len(matches)}")

safe_negation_terms = [
    "not solve",
    "does not solve",
    "do not solve",
    "not solved",
    "unsolved",
    "remain challenging",
    "remains challenging",
    "remain difficult",
    "remains difficult",
    "open challenge",
    "limitation",
]

for sent in split_sentences(all_text):
    low = sent.lower()

    if "configuration" in low and "solv" in low:
        if not any(term in low for term in safe_negation_terms):
            err("Possible overclaim about configuration shift being solved: " + sent[:240])

    if "outdoor" in low and "solv" in low:
        if not any(term in low for term in safe_negation_terms):
            err("Possible overclaim about outdoor shift being solved: " + sent[:240])

gated_safe_negation_terms = [
    "not used as the final",
    "not the final",
    "not used as final",
    "baseline",
    "fusion baseline",
]

for sent in split_sentences(all_text):
    low = sent.lower()
    if "gated oob hybrid" in low and "final" in low:
        if not any(term in low for term in gated_safe_negation_terms):
            err("Possible forbidden gated final-method claim: " + sent[:240])

# ---------------------------------------------------------------------
# 2. Required key numbers
# ---------------------------------------------------------------------
required_numbers = {
    "F cross-day acc": r"75\.0\s*\$?\\pm\$?\s*5\.3|75\.0\s*±\s*5\.3",
    "CNN cross-day acc": r"54\.2\s*\$?\\pm\$?\s*14\.2|54\.2\s*±\s*14\.2",
    "Bootstrap CI lower": r"\+9\.2",
    "Bootstrap CI upper": r"\+32\.5",
    "F_no_chirp": r"75\.0\s*\$?\\pm\$?\s*3\.4|75\.0\s*±\s*3\.4",
    "D_chirp": r"9\.7\s*\$?\\pm\$?\s*2\.0|9\.7\s*±\s*2\.0",
    "RX1_to_RX2 F": r"18\.1\s*\$?\\pm\$?\s*3\.9|18\.1\s*±\s*3\.9",
    "RX2_to_RX1 F": r"15\.3\s*\$?\\pm\$?\s*7\.1|15\.3\s*±\s*7\.1",
    "CNN params": r"47\.7K",
    "Hybrid params": r"1\.16M",
    "CNN latency": r"1\.12",
    "Hybrid latency": r"2\.45",
}

for name, pat in required_numbers.items():
    if not re.search(pat, all_text):
        warn(f"Required number not found or formatted differently: {name}")

# ---------------------------------------------------------------------
# 3. Required protocol statements
# ---------------------------------------------------------------------
required_protocol_terms = [
    ("cross-day train split", r"Day1--Day3|Day1–Day3|Day1-+3"),
    ("cross-day validation split", r"Day4"),
    ("cross-day test split", r"Day5"),
    ("cross-day z-score", r"z-score|zscore"),
    ("batch 128", r"128"),
    ("deployment ratio", r"Deployment.*ratio|ratio.*Deployment"),
    ("deployment batch 256", r"256"),
    ("label smoothing 0.05", r"0\.05"),
    ("cross-receiver ratio", r"Cross-receiver.*ratio|ratio.*cross-receiver"),
    ("source-only", r"source-only|source domain|source-domain"),
]

for name, pat in required_protocol_terms:
    if not re.search(pat, all_text, flags=re.IGNORECASE | re.DOTALL):
        warn(f"Protocol statement may be missing: {name}")

# ---------------------------------------------------------------------
# 4. Figure / table references and files
# ---------------------------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", all_text))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", all_text))
missing_refs = sorted(r for r in refs if r not in labels)
if missing_refs:
    err(f"Undefined labels referenced: {missing_refs}")

# includegraphics files
graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text)
for g in graphics:
    candidates = [
        ROOT / g,
        FIGS / g,
        ROOT / (g + ".pdf"),
        FIGS / (g + ".pdf"),
        ROOT / (g + ".png"),
        FIGS / (g + ".png"),
    ]
    if not any(c.exists() for c in candidates):
        err(f"Missing figure file for includegraphics: {g}")

# table inputs
inputs = re.findall(r"\\input\{([^}]+)\}", all_text)
for inp in inputs:
    if inp.startswith("tables/"):
        if not (ROOT / inp).exists():
            err(f"Missing table input file: {inp}")

required_labels = [
    "tab:cross_day_main",
    "tab:fusion_chirp",
    "tab:deployment_shift",
    "tab:cross_receiver",
    "tab:edge_deployment",
    "fig:architecture",
    "fig:results_summary",
    "fig:cross_receiver_stress",
]
for lab in required_labels:
    if lab not in labels:
        warn(f"Expected label not found: {lab}")

# ---------------------------------------------------------------------
# 5. Citation status check
# ---------------------------------------------------------------------
cite_keys = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", all_text):
    for k in m.group(1).split(","):
        cite_keys.add(k.strip())

bib_keys = set(re.findall(r"@\w+\{([^,]+),", REF_BIB.read_text(encoding="utf-8", errors="ignore"))) if REF_BIB.exists() else set()
missing_bib = sorted(k for k in cite_keys if k not in bib_keys)
if missing_bib:
    err(f"Cite keys missing from refs.bib: {missing_bib}")

allowed_status = {}
if REF_CSV.exists():
    with REF_CSV.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("bibkey") or "").strip()
            status = (row.get("status") or "").strip()
            if key:
                allowed_status[key] = status

bad_status = []
for k in sorted(cite_keys):
    st = allowed_status.get(k)
    if st not in {"READ", "ABSTRACT_CHECKED"}:
        bad_status.append((k, st))
if bad_status:
    err(f"Cite keys not READ/ABSTRACT_CHECKED in reference_candidates.csv: {bad_status}")

# ---------------------------------------------------------------------
# 6. PDF-source placeholder detection (main + sections + tables + bib only)
# ---------------------------------------------------------------------
pdf_source_files = [MAIN] + sorted(SECTIONS.glob("*.tex")) + sorted(TABLES.glob("*.tex")) + ([REF_BIB] if REF_BIB.exists() else [])
placeholder_patterns = [
    r"Author~One",
    r"Author One",
    r"Author~Two",
    r"Author Two",
    r"Affiliations TBD",
    r"Manuscript prepared",
    r"\\section\*\{Acknowledgment\}",
    r"draft block diagram",
    r"TODO",
    r"FIXME",
    r"placeholder",
    r"待补",
    r"占位",
    r"A\. Author",
    r"Lorem ipsum",
]
for p in pdf_source_files:
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8", errors="ignore")
    for pat in placeholder_patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            err(f"PDF-source placeholder in {p.relative_to(ROOT)}: pattern={pat}")

# Section III length check
sys_model = SECTIONS / "03_system_model.tex"
if sys_model.exists():
    s = sys_model.read_text(encoding="utf-8", errors="ignore")
    word_count = len(re.findall(r"[A-Za-z]+", s))
    if word_count < 500:
        err(f"Section III system model appears too short / skeletal: {word_count} English words")
else:
    err("Missing sections/03_system_model.tex")

# ---------------------------------------------------------------------
# 7. Print report
# ---------------------------------------------------------------------
print("=== IoTJ CONSISTENCY AUDIT ===")
print(f"TeX files checked: {len(tex_files)}")
print(f"Citations used: {len(cite_keys)}")
print(f"Labels defined: {len(labels)}")
print(f"Refs used: {len(refs)}")
print(f"Graphics used: {len(graphics)}")
print()

if warnings:
    print("WARNINGS:")
    for w in warnings:
        print(" - " + w)
    print()

if errors:
    print("ERRORS:")
    for e in errors:
        print(" - " + e)
    print()
    print("AUDIT RESULT: FAIL")
    sys.exit(1)

print("AUDIT RESULT: PASS")
