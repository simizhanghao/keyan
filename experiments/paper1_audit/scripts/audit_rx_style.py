#!/usr/bin/env python3
"""Day4 RX-style eval audit. Frozen 1C C' checkpoints only. No training, no Day5."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0")
TRUE_FULL = "C_full_ratio"
RX_FULL = "C_full_ratio_rx_style"
SEEDS = [0, 1, 2, 3, 4]
DROP_PP = 5.0


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def mean_std(values: list[float]) -> str:
    if not values:
        return "?"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def load_metrics(model: str, seed: int) -> dict | None:
    path = ROOT / "eval_val" / model / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> int:
    per_seed = []
    missing = []
    window_drops = []
    file_drops = []
    for seed in SEEDS:
        true_m = load_metrics(TRUE_FULL, seed)
        rx_m = load_metrics(RX_FULL, seed)
        if true_m is None or rx_m is None:
            missing.append(seed)
            continue
        if rx_m.get("rx_style_eval") is not True:
            raise SystemExit(f"seed {seed} missing rx_style_eval=true")
        drop_w = 100.0 * true_m["window_acc"] - 100.0 * rx_m["window_acc"]
        drop_f = 100.0 * true_m["file_acc"] - 100.0 * rx_m["file_acc"]
        window_drops.append(round(drop_w, 1))
        file_drops.append(round(drop_f, 1))
        per_seed.append(
            {
                "seed": seed,
                "true_window_pct": pct(true_m["window_acc"]),
                "rx_window_pct": pct(rx_m["window_acc"]),
                "drop_window_pp": round(drop_w, 1),
                "true_file_pct": pct(true_m["file_acc"]),
                "rx_file_pct": pct(rx_m["file_acc"]),
                "drop_file_pp": round(drop_f, 1),
                "below_5pp_window": drop_w < DROP_PP,
                "n_files": rx_m["num_files"],
            }
        )
    payload = {
        "day5_used": False,
        "no_training": True,
        "primary": f"{TRUE_FULL} vs {RX_FULL}",
        "delta_drop_pp": DROP_PP,
        "note": "Eval-only RX-style on frozen 1C C'. In-band scale locked. Window drop = clean C' − RX C'. Mean drop < 5pp → not strongly RX-entangled at inference. Threshold is not moved. RCOF stays closed.",
        "missing_seeds": missing,
        "per_seed": per_seed,
        "window_drop_pp": window_drops,
        "file_drop_pp": file_drops,
    }
    if window_drops:
        payload["window_drop_mean_pp"] = round(statistics.mean(window_drops), 1)
        payload["rx_entangled"] = statistics.mean(window_drops) >= DROP_PP
    out_json = ROOT / "rx_style_eval.json"
    out_md = ROOT / "rx_style_eval.md"
    lines = [
        "# Day4 RX-style eval",
        "",
        "Frozen 1C C' checkpoints. No retraining. Day5 unused.",
        "Operators: tilt / OOB scale / gain / phase / noise. In-band scale locked at 1.",
        "",
        f"Frozen rule: mean window drop < {DROP_PP:.0f}pp → not strongly RX-entangled at inference.",
        "",
        "| seed | C' win | RX win | drop pp | C' file | RX file | drop pp |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {row['true_window_pct']:.1f} | {row['rx_window_pct']:.1f} | "
            f"{row['drop_window_pp']:.1f} | {row['true_file_pct']:.1f} | {row['rx_file_pct']:.1f} | "
            f"{row['drop_file_pp']:.1f} |"
        )
    if window_drops:
        lines.extend(
            [
                "",
                f"window drop all-5: {window_drops}  mean {mean_std(window_drops)}",
                f"RX-entangled (mean drop ≥ {DROP_PP:.0f}pp): {payload['rx_entangled']}",
            ]
        )
    if missing:
        lines.extend(["", f"missing seeds (not a verdict): {missing}"])
    lines.extend(["", "RCOF / Day5 / 1D / Hann/guard are not opened here.", ""])
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
