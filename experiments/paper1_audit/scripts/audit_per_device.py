#!/usr/bin/env python3
"""Paper 1 revision-reserve: Day4 per-device table on frozen 1C evals.

No training. No GPU. Reads eval_val/{A,B,C'}/seed_*/per_device_accuracy.csv.
Does not open Day5, LODO, RX2, or S1 5-seed.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0")
FROZEN_EVAL = ROOT / "eval_val"
MODELS = ["A_cnn_iq", "B_exact_main_no_oob", "C_full_ratio"]
MODEL_LABEL = {
    "A_cnn_iq": "A CNN",
    "B_exact_main_no_oob": "B Main",
    "C_full_ratio": "C' Full ratio",
}
SEEDS = [0, 1, 2, 3, 4]
N_DEVICES = 24
BROAD_MIN = 16
CONCENTRATED_MAX = 8


def mean_std(values: list[float]) -> str:
    if not values:
        return "?"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def load_per_device(model: str, seed: int) -> list[dict]:
    path = FROZEN_EVAL / model / f"seed_{seed}" / "per_device_accuracy.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_DEVICES:
        raise SystemExit(f"{path}: expected {N_DEVICES} devices, got {len(rows)}")
    return rows


def load_metrics_file_acc(model: str, seed: int) -> float:
    path = FROZEN_EVAL / model / f"seed_{seed}" / "metrics.json"
    return 100.0 * float(json.loads(path.read_text())["file_acc"])


def main() -> int:
    by_model: dict[str, dict[int, dict[str, list[float]]]] = {}
    sanity = []
    missing = []
    for model in MODELS:
        by_model[model] = {
            label: {"window": [], "file": []} for label in range(N_DEVICES)
        }
        for seed in SEEDS:
            try:
                rows = load_per_device(model, seed)
            except SystemExit as exc:
                missing.append(str(exc))
                continue
            file_mean = 0.0
            for row in rows:
                label = int(row["label"])
                if int(row["num_files"]) != 1 or int(row["num_samples"]) != 256:
                    raise SystemExit(
                        f"{model} seed {seed} label {label}: expected 256 windows / 1 file"
                    )
                win = 100.0 * float(row["window_acc"])
                file_acc = 100.0 * float(row["file_acc"])
                by_model[model][label]["window"].append(win)
                by_model[model][label]["file"].append(file_acc)
                file_mean += file_acc
            file_mean /= N_DEVICES
            frozen = load_metrics_file_acc(model, seed)
            sanity.append(
                {
                    "model": model,
                    "seed": seed,
                    "macro_file_pct": round(file_mean, 4),
                    "metrics_file_pct": round(frozen, 4),
                    "ok": abs(file_mean - frozen) <= 0.05,
                }
            )
    if missing:
        raise SystemExit("missing CSVs:\n" + "\n".join(missing))
    sanity_ok = all(item["ok"] for item in sanity)

    devices = []
    n_file_win = n_file_lose = n_file_tie = 0
    n_win_win = n_win_lose = n_win_tie = 0
    for label in range(N_DEVICES):
        c_file = statistics.mean(by_model["C_full_ratio"][label]["file"])
        a_file = statistics.mean(by_model["A_cnn_iq"][label]["file"])
        b_file = statistics.mean(by_model["B_exact_main_no_oob"][label]["file"])
        c_win = statistics.mean(by_model["C_full_ratio"][label]["window"])
        a_win = statistics.mean(by_model["A_cnn_iq"][label]["window"])
        b_win = statistics.mean(by_model["B_exact_main_no_oob"][label]["window"])
        if c_file > a_file:
            n_file_win += 1
        elif c_file < a_file:
            n_file_lose += 1
        else:
            n_file_tie += 1
        if c_win > a_win:
            n_win_win += 1
        elif c_win < a_win:
            n_win_lose += 1
        else:
            n_win_tie += 1
        devices.append(
            {
                "label": label,
                "device_name": f"Device{label + 1}",
                "window_pct": {
                    "C_full_ratio": mean_std(by_model["C_full_ratio"][label]["window"]),
                    "A_cnn_iq": mean_std(by_model["A_cnn_iq"][label]["window"]),
                    "B_exact_main_no_oob": mean_std(by_model["B_exact_main_no_oob"][label]["window"]),
                },
                "file_pct": {
                    "C_full_ratio": mean_std(by_model["C_full_ratio"][label]["file"]),
                    "A_cnn_iq": mean_std(by_model["A_cnn_iq"][label]["file"]),
                    "B_exact_main_no_oob": mean_std(by_model["B_exact_main_no_oob"][label]["file"]),
                },
                "file_n5": {
                    "C_full_ratio": int(round(c_file / 20.0)),
                    "A_cnn_iq": int(round(a_file / 20.0)),
                    "B_exact_main_no_oob": int(round(b_file / 20.0)),
                },
                "delta_file_cp_minus_a": round(c_file - a_file, 1),
                "delta_window_cp_minus_a": round(c_win - a_win, 1),
            }
        )

    if not sanity_ok:
        spread = "SANITY_FAIL"
    elif n_file_win >= BROAD_MIN:
        spread = "BROAD"
    elif n_file_win <= CONCENTRATED_MAX:
        spread = "CONCENTRATED"
    else:
        spread = "MIXED"

    payload = {
        "day5_used": False,
        "training": False,
        "gpu": False,
        "models": MODELS,
        "seeds": SEEDS,
        "n_devices": N_DEVICES,
        "source": "eval_val/*/per_device_accuracy.csv",
        "device_name_note": (
            "Device{label+1} is the experiment index. raw Device9 is excluded; "
            "experiment Device9 is remapped raw Device10."
        ),
        "sanity": sanity,
        "sanity_ok": sanity_ok,
        "counts_file_cp_vs_cnn": {
            "win": n_file_win,
            "lose": n_file_lose,
            "tie": n_file_tie,
        },
        "counts_window_cp_vs_cnn": {
            "win": n_win_win,
            "lose": n_win_lose,
            "tie": n_win_tie,
        },
        "broad_min": BROAD_MIN,
        "concentrated_max": CONCENTRATED_MAX,
        "spread": spread,
        "devices": devices,
        "note": (
            "Revision reserve. File per seed is 0/1 (one Day4 file). "
            "Do not open Day5 / LODO / RX2 from this table."
        ),
    }
    out_json = ROOT / "per_device_day4.json"
    out_md = ROOT / "per_device_day4.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Paper 1 per-device Day4 (frozen 1C, K=256 mean_logits)",
        "",
        f"spread={spread}  sanity_ok={sanity_ok}  day5=unused",
        f"C' vs CNN file: win {n_file_win} / lose {n_file_lose} / tie {n_file_tie}",
        f"C' vs CNN window: win {n_win_win} / lose {n_win_lose} / tie {n_win_tie}",
        "",
        "Experiment Device9 = remapped raw Device10 (raw Device9 excluded).",
        "File-Acc per seed is 0/1 (one file per device). Mean is seeds correct / 5.",
        "",
        "| Dev | C' win% | CNN win% | Main win% | C' file n/5 | CNN file n/5 | Main file n/5 | Δfile C'−CNN | Δwin C'−CNN |",
        "| ---: | ------: | -------: | --------: | ----------: | -----------: | ------------: | -----------: | ----------: |",
    ]
    for item in devices:
        lines.append(
            "| {dev} | {cw} | {aw} | {bw} | {cf}/5 | {af}/5 | {bf}/5 | {df:+.1f} | {dw:+.1f} |".format(
                dev=item["device_name"],
                cw=item["window_pct"]["C_full_ratio"],
                aw=item["window_pct"]["A_cnn_iq"],
                bw=item["window_pct"]["B_exact_main_no_oob"],
                cf=item["file_n5"]["C_full_ratio"],
                af=item["file_n5"]["A_cnn_iq"],
                bf=item["file_n5"]["B_exact_main_no_oob"],
                df=item["delta_file_cp_minus_a"],
                dw=item["delta_window_cp_minus_a"],
            )
        )
    lines.extend(
        [
            "",
            "## Pre-registered spread",
            "",
            f"BROAD if C' file-mean > CNN on ≥{BROAD_MIN}/24 devices.",
            f"CONCENTRATED if that count is ≤{CONCENTRATED_MAX}/24.",
            "Otherwise MIXED. This does not open Day5, LODO, or RX2.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    print("SPREAD", spread)
    return 0 if sanity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
