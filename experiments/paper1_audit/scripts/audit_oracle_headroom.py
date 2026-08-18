#!/usr/bin/env python3
"""Day4 label-oracle headroom from frozen 1C predictions. No training, no Day5."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0")
SEEDS = [0, 1, 2, 3, 4]
MAIN = "B_exact_main_no_oob"
PRIMARY_FULL = "C_full_ratio"
SECONDARY_FULL = "C_full_zscore"
DELTA_DROP_PP = 5.0
MAIN_TRAINED_SEEDS = [0, 1]


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def mean_std(values: list[float]) -> str:
    if not values:
        return "?"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def load_rows(model: str, seed: int, name: str) -> list[dict]:
    path = ROOT / "eval_val" / model / f"seed_{seed}" / name
    with path.open() as handle:
        return list(csv.DictReader(handle))


def window_key(row: dict) -> tuple[str, int]:
    return row["file_path"], int(row["window_index"])


def file_key(row: dict) -> str:
    return row["file_path"]


def aligned_correct(left: list[dict], right: list[dict], key_fn) -> tuple[list[int], list[int]]:
    right_map = {key_fn(row): int(row["correct"]) for row in right}
    left_c = []
    right_c = []
    missing = 0
    for row in left:
        key = key_fn(row)
        if key not in right_map:
            missing += 1
            continue
        left_c.append(int(row["correct"]))
        right_c.append(right_map[key])
    if missing:
        raise SystemExit(f"alignment missing {missing} keys")
    if len(left_c) != len(right):
        raise SystemExit(f"alignment size mismatch {len(left_c)} vs {len(right)}")
    return left_c, right_c


def pair_stats(main_c: list[int], full_c: list[int]) -> dict:
    n = len(main_c)
    both = sum(a and b for a, b in zip(main_c, full_c))
    main_only = sum(a and not b for a, b in zip(main_c, full_c))
    full_only = sum((not a) and b for a, b in zip(main_c, full_c))
    neither = sum((not a) and (not b) for a, b in zip(main_c, full_c))
    main_acc = sum(main_c) / n
    full_acc = sum(full_c) / n
    oracle_acc = (both + main_only + full_only) / n
    best = max(main_acc, full_acc)
    delta_pp = 100.0 * (oracle_acc - best)
    return {
        "n": n,
        "main_acc_pct": pct(main_acc),
        "full_acc_pct": pct(full_acc),
        "best_expert_pct": pct(best),
        "oracle_acc_pct": pct(oracle_acc),
        "delta_pp": round(delta_pp, 1),
        "below_5pp": delta_pp < DELTA_DROP_PP,
        "both_correct": both,
        "main_only": main_only,
        "full_only": full_only,
        "neither": neither,
        "main_only_pct": pct(main_only / n),
        "full_only_pct": pct(full_only / n),
    }


def analyze_pair(full_name: str) -> dict:
    per_seed = []
    for seed in SEEDS:
        main_w = load_rows(MAIN, seed, "predictions.csv")
        full_w = load_rows(full_name, seed, "predictions.csv")
        main_f = load_rows(MAIN, seed, "file_predictions.csv")
        full_f = load_rows(full_name, seed, "file_predictions.csv")
        if len(main_w) != 6144 or len(full_w) != 6144:
            raise SystemExit(f"{full_name} seed {seed}: expected 6144 windows")
        if len(main_f) != 24 or len(full_f) != 24:
            raise SystemExit(f"{full_name} seed {seed}: expected 24 files")
        mw, fw = aligned_correct(main_w, full_w, window_key)
        mf, ff = aligned_correct(main_f, full_f, file_key)
        per_seed.append(
            {
                "seed": seed,
                "main_collapsed": seed not in MAIN_TRAINED_SEEDS,
                "window": pair_stats(mw, fw),
                "file": pair_stats(mf, ff),
            }
        )
    return {"full_model": full_name, "per_seed": per_seed}


def subset_deltas(pair: dict, seeds: list[int], level: str) -> list[float]:
    wanted = set(seeds)
    return [row[level]["delta_pp"] for row in pair["per_seed"] if row["seed"] in wanted]


def main() -> int:
    primary = analyze_pair(PRIMARY_FULL)
    secondary = analyze_pair(SECONDARY_FULL)
    payload = {
        "day5_used": False,
        "no_training": True,
        "oracle_type": "label_oracle",
        "delta_drop_pp": DELTA_DROP_PP,
        "primary_pair": f"{MAIN} vs {PRIMARY_FULL}",
        "secondary_pair": f"{MAIN} vs {SECONDARY_FULL}",
        "main_trained_seeds": MAIN_TRAINED_SEEDS,
        "note": (
            "Label oracle is an upper bound on any utility gate. "
            "Collapsed Main seeds make max(Main,Full)≈Full; report them but do not "
            "let them alone DROP the 5pp gate. Threshold stays 5pp."
        ),
        "primary": primary,
        "secondary": secondary,
        "window_delta_pp": {
            "primary_all5": subset_deltas(primary, SEEDS, "window"),
            "primary_main_trained": subset_deltas(primary, MAIN_TRAINED_SEEDS, "window"),
            "secondary_all5": subset_deltas(secondary, SEEDS, "window"),
            "secondary_main_trained": subset_deltas(secondary, MAIN_TRAINED_SEEDS, "window"),
        },
    }
    out_json = ROOT / "oracle_headroom.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    def block(title: str, pair: dict) -> list[str]:
        lines = [f"## {title}", "", "| seed | Main win | Full win | best | oracle | Δ pp | Main-only | Full-only | collapsed |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in pair["per_seed"]:
            w = row["window"]
            lines.append(
                f"| {row['seed']} | {w['main_acc_pct']:.1f} | {w['full_acc_pct']:.1f} | "
                f"{w['best_expert_pct']:.1f} | {w['oracle_acc_pct']:.1f} | {w['delta_pp']:.1f} | "
                f"{w['main_only_pct']:.1f} | {w['full_only_pct']:.1f} | {row['main_collapsed']} |"
            )
        lines += ["", "File-Acc (same oracle, K=256; not the mechanism gate):", "", "| seed | Main file | Full file | best | oracle | Δ pp |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in pair["per_seed"]:
            f = row["file"]
            lines.append(
                f"| {row['seed']} | {f['main_acc_pct']:.1f} | {f['full_acc_pct']:.1f} | "
                f"{f['best_expert_pct']:.1f} | {f['oracle_acc_pct']:.1f} | {f['delta_pp']:.1f} |"
            )
        wd_all = subset_deltas(pair, SEEDS, "window")
        wd_ok = subset_deltas(pair, MAIN_TRAINED_SEEDS, "window")
        lines += [
            "",
            f"window Δ all-5: {wd_all}  mean {mean_std(wd_all)}",
            f"window Δ Main-trained {{0,1}}: {wd_ok}  mean {mean_std(wd_ok)}",
            f"frozen DROP if window Δ < {DELTA_DROP_PP:.0f}pp; collapsed seeds are diagnostic, not a moved threshold.",
            "",
        ]
        return lines

    md = [
        "# Day4 label-oracle headroom",
        "",
        "No training. Day5 unused. Primary pair B vs C'. Secondary B vs C.",
        "Oracle = Main correct OR Full correct. Δ = oracle − max(Main, Full).",
        "",
    ]
    md += block("Primary window: B Main vs C' Full ratio", primary)
    md += block("Secondary window: B Main vs C Full zscore", secondary)
    md += [
        "Utility gate is not opened here. Shuffle and RX-style are not started.",
        "",
    ]
    out_md = ROOT / "oracle_headroom.md"
    out_md.write_text("\n".join(md))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
