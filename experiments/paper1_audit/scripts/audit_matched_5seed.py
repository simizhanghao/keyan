#!/usr/bin/env python3
"""Day4 matched 5-seed hypothesis audit. Does not evaluate Day5 or change gates."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0")
MODELS = ["A_cnn_iq", "B_exact_main_no_oob", "C_full_zscore", "C_full_ratio"]
SEEDS = [0, 1, 2, 3, 4]
GREEN_COUNT = 4


def load_metrics(model: str, seed: int) -> dict | None:
    path = ROOT / "eval_val" / model / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def mean_std(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.1f}" if values else "?"
    mu = statistics.mean(values)
    sd = statistics.stdev(values)
    return f"{mu:.1f}±{sd:.1f}"


def main() -> int:
    missing = []
    file_acc: dict[str, list[float]] = {m: [] for m in MODELS}
    window_acc: dict[str, list[float]] = {m: [] for m in MODELS}
    present_seeds = []
    for seed in SEEDS:
        ok = True
        row_file = {}
        row_win = {}
        for model in MODELS:
            m = load_metrics(model, seed)
            if m is None:
                missing.append(f"{model}/seed_{seed}")
                ok = False
                continue
            if int(m.get("num_files", 0)) != 24:
                raise SystemExit(f"{model} seed {seed}: expected 24 Day4 files")
            row_file[model] = pct(m["file_acc"])
            row_win[model] = pct(m["window_acc"])
        if ok:
            present_seeds.append(seed)
            for model in MODELS:
                file_acc[model].append(row_file[model])
                window_acc[model].append(row_win[model])

    def paired(left: str, right: str, table: dict[str, list[float]]) -> list[float]:
        return [a - b for a, b in zip(table[left], table[right])]

    n = len(present_seeds)
    complete = n == 5 and not missing
    delta_file_c_b = paired("C_full_zscore", "B_exact_main_no_oob", file_acc) if n else []
    delta_file_cp_b = paired("C_full_ratio", "B_exact_main_no_oob", file_acc) if n else []
    delta_win_c_b = paired("C_full_zscore", "B_exact_main_no_oob", window_acc) if n else []
    delta_win_cp_b = paired("C_full_ratio", "B_exact_main_no_oob", window_acc) if n else []
    delta_file_c_a = paired("C_full_zscore", "A_cnn_iq", file_acc) if n else []
    delta_file_cp_a = paired("C_full_ratio", "A_cnn_iq", file_acc) if n else []

    def n_pos(xs: list[float]) -> int:
        return sum(1 for x in xs if x > 0)

    def window_not_opposite(file_d: list[float], win_d: list[float]) -> bool:
        if not file_d:
            return False
        file_mean = statistics.mean(file_d)
        win_mean = statistics.mean(win_d)
        if file_mean == 0:
            return win_mean == 0
        return (file_mean > 0 and win_mean >= 0) or (file_mean < 0 and win_mean <= 0)

    count_c_gt_b = n_pos(delta_file_c_b)
    count_cp_gt_b = n_pos(delta_file_cp_b)
    count_c_gt_a = n_pos(delta_file_c_a)
    count_cp_gt_a = n_pos(delta_file_cp_a)

    c_green_count = complete and count_c_gt_b >= GREEN_COUNT
    cp_green_count = complete and count_cp_gt_b >= GREEN_COUNT
    c_window_ok = complete and window_not_opposite(delta_file_c_b, delta_win_c_b)
    cp_window_ok = complete and window_not_opposite(delta_file_cp_b, delta_win_cp_b)

    payload = {
        "day5_used": False,
        "k": 256,
        "file_vote": "mean_logits",
        "recipe_frozen": True,
        "green_count_threshold": GREEN_COUNT,
        "seeds_present": present_seeds,
        "complete_5_seeds": complete,
        "missing": missing,
        "file_acc_pct": file_acc,
        "window_acc_pct": window_acc,
        "delta_file_pct": {
            "C_minus_B": delta_file_c_b,
            "Cp_minus_B": delta_file_cp_b,
            "C_minus_A": delta_file_c_a,
            "Cp_minus_A": delta_file_cp_a,
        },
        "delta_window_pct": {
            "C_minus_B": delta_win_c_b,
            "Cp_minus_B": delta_win_cp_b,
        },
        "counts": {
            "C_gt_B_file": f"{count_c_gt_b}/{n}" if n else "0/0",
            "Cp_gt_B_file": f"{count_cp_gt_b}/{n}" if n else "0/0",
            "C_gt_A_file": f"{count_c_gt_a}/{n}" if n else "0/0",
            "Cp_gt_A_file": f"{count_cp_gt_a}/{n}" if n else "0/0",
        },
        "registered_count_gate": {
            "C_zscore_full_gt_main": c_green_count,
            "Cp_ratio_full_gt_main": cp_green_count,
        },
        "window_sign_not_opposite_mean": {
            "C_zscore": c_window_ok,
            "Cp_ratio": cp_window_ok,
        },
        "verdict": "INCOMPLETE" if not complete else "COMPUTED_COUNTS_ONLY",
        "note": (
            "Do not treat count-gate True as a paper GREEN until window/file "
            "tables are read. Do not move the 4/5 threshold. Experiment 2 closed."
        ),
    }
    out_json = ROOT / "audit_5seed.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    def row(name: str, key: str) -> str:
        by_seed = {s: file_acc[key][i] for i, s in enumerate(present_seeds)}
        cells = " | ".join(f"{by_seed[s]:5.1f}" if s in by_seed else "    ?" for s in SEEDS)
        mean = mean_std([by_seed[s] for s in present_seeds]) if present_seeds else "?"
        return f"| {name:<15} | {cells} | {mean:>8} |"

    lines = [
        "# Day4 matched 5-seed audit",
        "",
        f"complete={complete}  missing={missing or 'none'}  day5=unused",
        "",
        "## File-Acc (%)",
        "",
        "| Model           | seed0 | seed1 | seed2 | seed3 | seed4 | Mean±Std |",
        "| --------------- | ----: | ----: | ----: | ----: | ----: | -------: |",
        row("B Main", "B_exact_main_no_oob"),
        row("C Full zscore", "C_full_zscore"),
        row("C' Full ratio", "C_full_ratio"),
        row("A CNN", "A_cnn_iq"),
        "",
        "## Paired Δ File-Acc (pp)",
        "",
        f"C−B   per seed: {delta_file_c_b}   count C>B = {payload['counts']['C_gt_B_file']}",
        f"C'−B  per seed: {delta_file_cp_b}   count C'>B = {payload['counts']['Cp_gt_B_file']}",
        f"C−A   per seed: {delta_file_c_a}   count C>A = {payload['counts']['C_gt_A_file']}",
        f"C'−A  per seed: {delta_file_cp_a}   count C'>A = {payload['counts']['Cp_gt_A_file']}",
        "",
        "## Paired Δ Window-Acc (pp)",
        "",
        f"C−B   per seed: {delta_win_c_b}",
        f"C'−B  per seed: {delta_win_cp_b}",
        "",
        "Registered count gate is 4/5 and is not moved after seeing data.",
        "verdict remains COMPUTED_COUNTS_ONLY until a human reads window+file together.",
        "Experiment 2 / RCOF closed.",
        "",
    ]
    out_md = ROOT / "audit_5seed.md"
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
