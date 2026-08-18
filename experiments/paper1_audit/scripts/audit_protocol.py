#!/usr/bin/env python3
"""Paper 1 Audit 1A: lock manifests, hashes, Device9, splits, .dat existence.

Does not train. Does not write to outputs/paper_ready_v3/.
Day5 is counted as the sealed test split only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

EXCLUDED_RAW = {9}
N_CLASSES = 24
WINDOW = 8192
DEVICE_RE = re.compile(r"Device(\d+)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def raw_device(row: dict[str, str]) -> int | None:
    for key in ("relative_path", "path"):
        m = DEVICE_RE.search(row.get(key, ""))
        if m:
            return int(m.group(1))
    return None


def resolve_dat(row: dict[str, str], data_root: Path) -> Path:
    p = Path(row["path"])
    if p.is_absolute():
        return p
    return data_root / p


def audit_primary(rows: list[dict[str, str]], data_root: Path) -> dict:
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    issues: list[str] = []
    split_info: dict[str, dict] = {}
    missing_dat: list[str] = []
    present = 0
    absent = 0

    for split in ("train", "val", "test"):
        srows = by_split.get(split, [])
        paths = [r["path"] for r in srows]
        path_set = set(paths)
        days = sorted({r["day"] for r in srows})
        labels = sorted({int(r["label"]) for r in srows})
        devices = sorted({int(r["device"]) for r in srows})
        raws = [raw_device(r) for r in srows]
        raw_hit = [x for x in raws if x is not None]
        device9 = [r["path"] for r in srows if raw_device(r) == 9]

        per_day: dict[str, int] = defaultdict(int)
        per_dev_day: dict[tuple[str, str], int] = defaultdict(int)
        for r in srows:
            per_day[r["day"]] += 1
            per_dev_day[(r["day"], r["device"])] += 1

        dat_ok = 0
        dat_missing = 0
        for r in srows:
            dp = resolve_dat(r, data_root)
            if dp.is_file() and dp.stat().st_size > 0:
                dat_ok += 1
                present += 1
            else:
                dat_missing += 1
                absent += 1
                missing_dat.append(str(dp))

        if device9:
            issues.append(f"{split}: Device9 present ({len(device9)})")
        if len(path_set) != len(paths):
            issues.append(f"{split}: duplicate paths")
        if set(labels) != set(range(N_CLASSES)):
            issues.append(f"{split}: labels {labels}")
        if len(devices) != N_CLASSES:
            issues.append(f"{split}: devices {devices}")
        if any(n != 1 for n in per_dev_day.values()):
            issues.append(f"{split}: not exactly one file per device-day")

        split_info[split] = {
            "n_files": len(srows),
            "n_unique_paths": len(path_set),
            "days": days,
            "n_labels": len(labels),
            "n_devices": len(devices),
            "files_per_day": dict(per_day),
            "device9_count": len(device9),
            "dat_present": dat_ok,
            "dat_missing": dat_missing,
        }

    train_p, val_p, test_p = (
        {r["path"] for r in by_split.get(s, [])} for s in ("train", "val", "test")
    )
    ov_tv = sorted(train_p & val_p)
    ov_tt = sorted(train_p & test_p)
    ov_vt = sorted(val_p & test_p)
    if ov_tv:
        issues.append(f"overlap train-val: {len(ov_tv)}")
    if ov_tt:
        issues.append(f"overlap train-test: {len(ov_tt)}")
    if ov_vt:
        issues.append(f"overlap val-test: {len(ov_vt)}")

    expected = {
        "train": (["1", "2", "3"], 72),
        "val": (["4"], 24),
        "test": (["5"], 24),
    }
    protocol_ok = True
    for split, (days, n) in expected.items():
        info = split_info.get(split, {})
        if info.get("days") != days or info.get("n_files") != n:
            protocol_ok = False
            issues.append(f"{split}: expected days={days} n={n}, got {info.get('days')} n={info.get('n_files')}")

    return {
        "protocol_matches_lock": protocol_ok and not issues,
        "issues": issues,
        "splits": split_info,
        "overlaps": {
            "train_val": len(ov_tv),
            "train_test": len(ov_tt),
            "val_test": len(ov_vt),
        },
        "dat_root": str(data_root),
        "dat_present": present,
        "dat_missing": absent,
        "dat_missing_examples": missing_dat[:12],
        "window_size_locked": WINDOW,
        "eval_windows_per_file_locked": 256,
        "file_aggregation_locked": "mean_logits",
    }


def audit_forbidden(keyan: Path) -> dict:
    oracle = keyan / "data/paper/cross_day_day1to5_oracle_target_val.csv"
    out = {"path": str(oracle.relative_to(keyan)), "exists": oracle.is_file()}
    if not oracle.is_file():
        out["note"] = "missing (ok if unused)"
        return out
    rows = read_rows(oracle)
    val_days = sorted({r["day"] for r in rows if r["split"] == "val"})
    test_days = sorted({r["day"] for r in rows if r["split"] == "test"})
    out["val_days"] = val_days
    out["test_days"] = test_days
    out["leaks_day5_into_val"] = val_days == ["5"]
    out["role"] = "FORBIDDEN for Experiment 1 development"
    return out


def audit_lodo(keyan: Path) -> list[dict]:
    folder = keyan / "data/paper/lodo_source_only"
    rows_out = []
    if not folder.is_dir():
        return [{"error": "lodo dir missing"}]
    for day in range(1, 6):
        p = folder / f"test_day_{day}.csv"
        rec: dict = {"path": str(p.relative_to(keyan)), "exists": p.is_file()}
        if p.is_file():
            rows = read_rows(p)
            rec["train_days"] = sorted({r["day"] for r in rows if r["split"] == "train"})
            rec["val_days"] = sorted({r["day"] for r in rows if r["split"] == "val"})
            rec["test_days"] = sorted({r["day"] for r in rows if r["split"] == "test"})
            rec["uses_day5_in_val"] = "5" in rec["val_days"]
            rec["uses_day5_in_train"] = "5" in rec["train_days"]
            rec["sealed_until"] = "1E"
        rows_out.append(rec)
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyan-root", default="/data1/hcc/llm4RF/new_phase")
    parser.add_argument("--data-root", default="/data1/hcc/llm4RF")
    parser.add_argument(
        "--out-dir",
        default="/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results",
    )
    args = parser.parse_args()
    keyan = Path(args.keyan_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = keyan / "data/paper/cross_day_day1to5_source_only.csv"
    frozen_table = Path(
        "/data1/hcc/llm4RF/outputs/paper_ready_v3/final_tables/table1_cross_day_main.csv"
    )
    freeze_paths = [
        primary,
        keyan / "data/paper/cross_day_day1to5_oracle_target_val.csv",
        *[keyan / f"data/paper/lodo_source_only/test_day_{d}.csv" for d in range(1, 6)],
        frozen_table,
        keyan / "src/rfhstu/features.py",
        keyan / "scripts/paper/lib/v3_job_defs.py",
    ]

    hashes: dict[str, dict] = {}
    for p in freeze_paths:
        rec = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0}
        if p.is_file():
            rec["sha256"] = sha256_file(p)
        hashes[str(p)] = rec

    if not primary.is_file():
        raise SystemExit(f"missing primary manifest: {primary}")

    primary_audit = audit_primary(read_rows(primary), data_root)
    report = {
        "experiment": "paper1_audit_1A",
        "keyan_root": str(keyan),
        "data_root": str(data_root),
        "primary_manifest": str(primary),
        "frozen_outputs_untouched": True,
        "day5_development_forbidden": True,
        "primary": primary_audit,
        "oracle_manifest": audit_forbidden(keyan),
        "lodo": audit_lodo(keyan),
        "legacy_frozen": {
            "ours_file_acc": "75.0±5.3",
            "cnn_file_acc": "54.2±14.2",
            "linear_no_oob_file_acc": "66.7±3.4",
            "matched_cnn_stem_no_oob_seed0": "8.3 (n=1, collapsed)",
            "table": str(frozen_table),
            "do_not_overwrite": True,
        },
        "gate_1A": "PASS" if primary_audit["protocol_matches_lock"] else "FAIL",
    }

    (out_dir / "protocol_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Protocol audit (1A)",
        "",
        f"- keyan: `{keyan}`",
        f"- data-root: `{data_root}`",
        f"- gate_1A: **{report['gate_1A']}**",
        f"- protocol_matches_lock: {primary_audit['protocol_matches_lock']}",
        f"- dat present/missing: {primary_audit['dat_present']} / {primary_audit['dat_missing']}",
        "",
        "## Issues",
        "",
    ]
    if primary_audit["issues"]:
        lines.extend(f"- {x}" for x in primary_audit["issues"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## Splits",
        "",
        "```json",
        json.dumps(primary_audit["splits"], indent=2),
        "```",
        "",
        "## Oracle (forbidden for development)",
        "",
        "```json",
        json.dumps(report["oracle_manifest"], indent=2),
        "```",
        "",
        "## LODO (sealed until 1E)",
        "",
        "```json",
        json.dumps(report["lodo"], indent=2),
        "```",
        "",
        "## Hashes",
        "",
        f"See `{out_dir / 'manifest_hashes.json'}`.",
        "",
        "Frozen `outputs/paper_ready_v3/` was not written.",
        "",
    ]
    (out_dir / "protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"gate_1A={report['gate_1A']}")
    print(f"wrote {out_dir / 'protocol_audit.json'}")
    print(f"wrote {out_dir / 'manifest_hashes.json'}")
    print(f"wrote {out_dir / 'protocol_audit.md'}")
    print(f"dat_present={primary_audit['dat_present']} dat_missing={primary_audit['dat_missing']}")
    if primary_audit["issues"]:
        print("ISSUES:")
        for x in primary_audit["issues"]:
            print(" -", x)
    return 0 if report["gate_1A"] == "PASS" and primary_audit["dat_missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
