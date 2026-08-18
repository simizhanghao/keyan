#!/usr/bin/env python3
"""Day4 OOB identity-shuffle audit.

--check-donors: CPU only, no training, no Day5.
Default: compare frozen 1C C' vs shuffled Full on Day4 window/file acc.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase")
RESULTS = ROOT / "experiments/paper1_audit/results/matched_seed0"
TRUE_FULL = "C_full_ratio"
SHUFFLE_FULL = "C_full_ratio_oob_shuffle"
SEEDS = [0, 1, 2, 3, 4]
DROP_PP = 5.0
MANIFEST = ROOT / "data/paper/cross_day_day1to5_source_only.csv"
DATA_ROOT = Path("/data1/hcc/llm4RF")


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def mean_std(values: list[float]) -> str:
    if not values:
        return "?"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def load_metrics(model: str, seed: int) -> dict | None:
    path = RESULTS / "eval_val" / model / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def check_donors(max_items: int = 256) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from rfhstu.data import SigMFIQDataset, load_manifest

    report = {"day5_used": False, "no_training": True, "checks": []}
    failed = False
    for split, random_windows, samples in (("train", True, 8), ("val", False, 8)):
        rows = load_manifest(MANIFEST, root=DATA_ROOT, split=split)
        ds = SigMFIQDataset(
            rows,
            window_size=8192,
            samples_per_file=samples,
            random_windows=random_windows,
            seed=0,
            input_norm="iq_rms",
            oob_identity_shuffle=True,
        )
        n = min(max_items, len(ds))
        mismatch = 0
        same_day = 0
        donors_e0 = []
        for i in range(n):
            item = ds[i]
            if int(item["oob_donor_label"]) != int(item["label"]):
                mismatch += 1
            if int(item["domains"][0]) == int(ds.rows[ds._donor_row_index(i, i // ds.samples_per_file)].domains["day"]):
                same_day += 1
            donors_e0.append(int(item["oob_donor_label"]))
            if item["oob_iq"].shape != item["iq"].shape:
                raise SystemExit(f"{split} oob_iq shape {tuple(item['oob_iq'].shape)} != iq {tuple(item['iq'].shape)}")
        rate = mismatch / n
        day_rate = same_day / n
        changed = None
        if random_windows:
            ds.set_epoch(1)
            donors_e1 = [int(ds[i]["oob_donor_label"]) for i in range(n)]
            changed = sum(a != b for a, b in zip(donors_e0, donors_e1)) / n
            if changed == 0.0:
                failed = True
        if rate != 1.0 or day_rate != 1.0:
            failed = True
        check = {
            "split": split,
            "n": n,
            "donor_mismatch_rate": rate,
            "same_day_rate": day_rate,
            "train_epoch_donor_change_rate": changed,
        }
        report["checks"].append(check)
        print(
            f"{split}: n={n} mismatch={rate:.3f} same_day={day_rate:.3f}"
            + (f" epoch_change={changed:.3f}" if changed is not None else "")
        )
        if not ds.oob_identity_shuffle:
            failed = True
    off = SigMFIQDataset(
        load_manifest(MANIFEST, root=DATA_ROOT, split="val"),
        window_size=8192,
        samples_per_file=4,
        random_windows=False,
        seed=0,
        input_norm="iq_rms",
        oob_identity_shuffle=False,
    )
    if "oob_iq" in off[0]:
        failed = True
        print("FAIL: shuffle-off dataset still has oob_iq")
    else:
        print("shuffle-off: no oob_iq (default path unchanged)")
    out = RESULTS / "oob_shuffle_donor_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report["ok"] = not failed
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)
    if failed:
        raise SystemExit("donor check failed")
    print("donor check PASS")
    return 0


def audit_results() -> int:
    per_seed = []
    missing = []
    window_drops = []
    file_drops = []
    for seed in SEEDS:
        true_m = load_metrics(TRUE_FULL, seed)
        shuf_m = load_metrics(SHUFFLE_FULL, seed)
        if true_m is None or shuf_m is None:
            missing.append(seed)
            continue
        mismatch = shuf_m.get("oob_donor_mismatch_rate", "")
        if mismatch not in ("", None) and float(mismatch) != 1.0:
            raise SystemExit(f"seed {seed} donor mismatch_rate={mismatch}, expected 1.0")
        true_w = 100.0 * true_m["window_acc"]
        shuf_w = 100.0 * shuf_m["window_acc"]
        true_f = 100.0 * true_m["file_acc"]
        shuf_f = 100.0 * shuf_m["file_acc"]
        drop_w = true_w - shuf_w
        drop_f = true_f - shuf_f
        window_drops.append(round(drop_w, 1))
        file_drops.append(round(drop_f, 1))
        per_seed.append(
            {
                "seed": seed,
                "true_window_pct": pct(true_m["window_acc"]),
                "shuffle_window_pct": pct(shuf_m["window_acc"]),
                "drop_window_pp": round(drop_w, 1),
                "true_file_pct": pct(true_m["file_acc"]),
                "shuffle_file_pct": pct(shuf_m["file_acc"]),
                "drop_file_pp": round(drop_f, 1),
                "below_5pp_window": drop_w < DROP_PP,
                "n_files": shuf_m["num_files"],
                "donor_mismatch_rate": mismatch,
            }
        )
    payload = {
        "day5_used": False,
        "primary": f"{TRUE_FULL} vs {SHUFFLE_FULL}",
        "delta_drop_pp": DROP_PP,
        "note": "Window drop = true Full − shuffled Full. Mean drop < 5pp shrinks the identity claim. Threshold is not moved. C zscore shuffle is not this step. RX is not opened here.",
        "missing_seeds": missing,
        "per_seed": per_seed,
        "window_drop_pp": window_drops,
        "file_drop_pp": file_drops,
    }
    if window_drops:
        payload["window_drop_mean_pp"] = round(statistics.mean(window_drops), 1)
        payload["identity_claim_shrinks"] = statistics.mean(window_drops) < DROP_PP
    out_json = RESULTS / "oob_identity_shuffle.json"
    out_md = RESULTS / "oob_identity_shuffle.md"
    lines = [
        "# Day4 OOB identity shuffle",
        "",
        "Primary: C' Full ratio vs shuffled C'. Main IQ/label kept. OOB from a same-day different device.",
        "Train donors reshuffle each epoch. Eval donors are frozen per window. Day5 unused.",
        "",
        f"Frozen rule: mean window drop < {DROP_PP:.0f}pp → identity claim shrinks. Gate is not moved.",
        "",
        "| seed | C' win | shuffle win | drop pp | C' file | shuffle file | drop pp |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {row['true_window_pct']:.1f} | {row['shuffle_window_pct']:.1f} | "
            f"{row['drop_window_pp']:.1f} | {row['true_file_pct']:.1f} | {row['shuffle_file_pct']:.1f} | "
            f"{row['drop_file_pp']:.1f} |"
        )
    if window_drops:
        lines.extend(
            [
                "",
                f"window drop all-5: {window_drops}  mean {mean_std(window_drops)}",
                f"identity claim shrinks (mean drop < {DROP_PP:.0f}pp): {payload['identity_claim_shrinks']}",
            ]
        )
    if missing:
        lines.extend(["", f"missing seeds (not a verdict): {missing}"])
    lines.extend(["", "Utility gate / RCOF / RX-style are not opened here.", ""])
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-donors", action="store_true")
    parser.add_argument("--max-items", type=int, default=256)
    args = parser.parse_args()
    if args.check_donors:
        return check_donors(args.max_items)
    return audit_results()


if __name__ == "__main__":
    raise SystemExit(main())
