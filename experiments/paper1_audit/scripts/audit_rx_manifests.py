#!/usr/bin/env python3
"""2B-2: source-only RX manifest audit. No training. No GPU. No Day5.

Checks both paper source-only manifests:
  train/val = source receiver only
  test      = target receiver only
  24 devices, files exist, no oracle protocol
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
DATA_ROOT = Path("/data1/hcc/llm4RF")
OUT = KEYAN / "experiments/paper1_audit/results/rx_manifest_audit"
MANIFESTS = {
    "rx1_to_rx2": {
        "path": KEYAN / "data/paper/rx1_to_rx2_source_only.csv",
        "source": 1,
        "target": 2,
    },
    "rx2_to_rx1": {
        "path": KEYAN / "data/paper/rx2_to_rx1_source_only.csv",
        "source": 1,
        "target": 2,
    },
}
MANIFESTS["rx2_to_rx1"]["source"] = 2
MANIFESTS["rx2_to_rx1"]["target"] = 1


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def audit_one(name: str, spec: dict) -> dict:
    path: Path = spec["path"]
    if not path.is_file():
        raise SystemExit(f"missing manifest: {path}")
    rows = load_rows(path)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    issues: list[str] = []
    split_info = {}
    for split in ("train", "val", "test"):
        part = by_split.get(split, [])
        recs = sorted({int(r["receiver"]) for r in part})
        devices = sorted({int(r["device"]) for r in part})
        protocols = sorted({r["protocol"] for r in part})
        missing = []
        for r in part:
            p = DATA_ROOT / r["path"]
            if not p.is_file():
                missing.append(r["path"])
        expected_rx = spec["source"] if split in ("train", "val") else spec["target"]
        if recs != [expected_rx]:
            issues.append(f"{name} {split} receivers={recs} expected=[{expected_rx}]")
        if len(devices) != 24:
            issues.append(f"{name} {split} n_device={len(devices)} expected=24")
        if protocols != ["cross_receiver_source_only"]:
            issues.append(f"{name} {split} protocol={protocols}")
        if "oracle" in "".join(protocols):
            issues.append(f"{name} {split} oracle leaked")
        if missing:
            issues.append(f"{name} {split} missing_files={len(missing)} e.g. {missing[0]}")
        split_info[split] = {
            "n_rows": len(part),
            "receivers": recs,
            "n_devices": len(devices),
            "protocols": protocols,
            "n_missing": len(missing),
        }

    train_paths = {r["path"] for r in by_split["train"]}
    test_paths = {r["path"] for r in by_split["test"]}
    if train_paths & test_paths:
        issues.append(f"{name} train/test path overlap")

    return {
        "manifest": str(path),
        "source": spec["source"],
        "target": spec["target"],
        "splits": split_info,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {name: audit_one(name, spec) for name, spec in MANIFESTS.items()}
    ok = all(r["ok"] for r in reports.values())
    payload = {
        "training": False,
        "gpu": False,
        "day5_used": False,
        "oracle_used": False,
        "day4_ckpt_used": False,
        "reports": reports,
        "verdict": "RX_MANIFEST_PASS" if ok else "RX_MANIFEST_FAIL",
    }
    out_json = OUT / "rx_manifest_audit.json"
    out_md = OUT / "rx_manifest_audit.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# RX source-only manifest audit",
        "",
        f"verdict={payload['verdict']}  training=false  gpu=false  oracle=false",
        "",
        "| direction | train RX | val RX | test RX | devices | missing | ok |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, r in reports.items():
        s = r["splits"]
        lines.append(
            f"| {name} | {s['train']['receivers']} | {s['val']['receivers']} | "
            f"{s['test']['receivers']} | {s['test']['n_devices']} | "
            f"{s['train']['n_missing']+s['val']['n_missing']+s['test']['n_missing']} | "
            f"{r['ok']} |"
        )
        for issue in r["issues"]:
            lines.append(f"- FAIL {issue}")
    lines.extend(
        [
            "",
            "Day4 C'/F0 checkpoints must not be loaded onto these manifests.",
            "Oracle target-val manifests are forbidden.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("VERDICT", payload["verdict"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
