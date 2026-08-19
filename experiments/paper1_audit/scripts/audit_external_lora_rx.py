#!/usr/bin/env python3
"""2C: local usability audit of Zhang/TMC multi-receiver LoRa data. No GPU. No download."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/external_rx_audit")
SEARCH_ROOTS = [
    Path("/data1/hcc/llm4RF/data"),
    Path("/data1/hcc/llm4RF/new_phase/data"),
    Path("/data1/hcc/llm4RF/new_phase/data/external/zhang_tmc_multi_rx"),
    Path("/data1/hcc/datasets"),
    Path("/data1/datasets"),
    Path("/data1/data"),
]
NAME_HINTS = (
    "multiple_receiver_train",
    "multiple_receiver_test",
    "receiver_drift_dataset",
    "d6vx-r538",
    "lora_dataset_multiple_receivers",
)
PAPER = {
    "official_page": "https://junqing-zhang.github.io/dataset-code/",
    "dataport": "https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers",
    "doi": "10.21227/d6vx-r538",
    "dut": 10,
    "sdr": 20,
    "bw_hz": 125000,
    "fs_hz_claimed": 1000000,
    "sf": 7,
    "carrier_mhz": 868.1,
    "zips": [
        "multiple_receiver_train.zip (~13 GB)",
        "multiple_receiver_test.zip (~11 GB)",
        "receiver_drift_dataset.zip (~3.7 GB)",
    ],
}


def looks_relevant(path: Path) -> bool:
    n = path.name.lower()
    return any(h in n for h in NAME_HINTS)


def scan() -> list[dict]:
    hits = []
    seen: set[str] = set()
    patterns = ["*", "*/*", "*/*/*"]
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for pat in patterns:
            try:
                for p in root.glob(pat):
                    if not looks_relevant(p):
                        continue
                    key = str(p.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(
                        {
                            "path": key,
                            "is_dir": p.is_dir(),
                            "is_file": p.is_file(),
                            "size_gb": round(p.stat().st_size / 1e9, 2) if p.is_file() else None,
                        }
                    )
            except (PermissionError, OSError):
                continue
    return hits


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hits = scan()
    if not hits:
        verdict = "LOCAL_ABSENT"
        oob = "not_applicable"
    else:
        verdict = "OOB_UNKNOWN"
        oob = "files_found_but_not_parsed"
    payload = {
        "training": False,
        "gpu": False,
        "download": False,
        "osu_rx_fail_frozen": True,
        "paper": PAPER,
        "hits": hits,
        "oob_guess": oob,
        "verdict": verdict,
        "note": (
            "IEEE DataPort needs a login; this script does not download. "
            "1 MHz / 125 kHz can preserve OOB only if dumps are full-band complex IQ. "
            "Preamble-only spectrograms are OOB_INSUFFICIENT. Do not retune F0 from this file."
        ),
    }
    out_json = OUT / "external_rx_audit.json"
    out_md = OUT / "external_rx_audit.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# External multi-receiver LoRa audit (Zhang / TMC)",
        "",
        f"verdict={verdict}  oob={oob}  training=false  gpu=false",
        "OSU 2-RX F0 remains RX_FAIL. Do not retune.",
        "",
        "## Paper claims (not yet file-verified)",
        "",
        f"- page: {PAPER['official_page']}",
        f"- dataport: {PAPER['dataport']}",
        f"- 10 DUT, 20 SDR, SF7, BW 125 kHz, fs 1 MHz claimed",
        "",
        "## Local hits",
        "",
    ]
    if hits:
        for h in hits:
            lines.append(f"- `{h['path']}` file={h['is_file']} dir={h['is_dir']} size_gb={h['size_gb']}")
    else:
        lines.append("- none under /data1/hcc, /data1/datasets, /data1/data")
        lines.append("- Human must download the three DataPort zips, then re-run this script")
    lines.extend(
        [
            "",
            "LOCAL_ABSENT → download then re-audit. Do not train.",
            "OOB_OK later → new multi-RX DG protocol, not F0 retune on OSU.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("VERDICT", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
