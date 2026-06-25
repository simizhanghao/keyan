#!/usr/bin/env python3
"""Generate final paper tables under outputs/paper_ready_v3/final_tables/."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_ready_v3" / "final_tables"

CONFIG_HELD_OUT = {1: "Config1 (SF7 BW125)", 2: "Config2 (SF8)", 3: "Config3 (SF11)", 4: "Config4"}
LOCATION_HELD_OUT = {1: "room (indoor)", 2: "office", 3: "outdoor"}


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}"


def mean_std(values: list[float], as_pct: bool = True) -> tuple[str, str]:
    if not values:
        return "", ""
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    if as_pct:
        return pct(m), pct(s)
    return f"{m:.4f}", f"{s:.4f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def table1_cross_day_main() -> None:
    rows = [
        {
            "model": "CNN-IQ",
            "model_id": "A_cnn_iq",
            "file_acc_mean_pct": "54.2",
            "file_acc_std_pct": "14.2",
            "file_macro_f1_mean_pct": "45.6",
            "file_macro_f1_std_pct": "14.8",
            "window_acc_mean_pct": "43.5",
            "window_acc_std_pct": "5.2",
            "window_macro_f1_mean_pct": "38.7",
            "window_macro_f1_std_pct": "5.7",
            "n_seeds": "5",
            "protocol": "Step1 cross-day (Day1-3 train / Day4 val / Day5 test)",
            "source": "outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md",
        },
        {
            "model": "RF-HSTU linear no OOB",
            "model_id": "B_linear_no_oob",
            "file_acc_mean_pct": "66.7",
            "file_acc_std_pct": "3.4",
            "file_macro_f1_mean_pct": "59.1",
            "file_macro_f1_std_pct": "4.5",
            "window_acc_mean_pct": "39.4",
            "window_acc_std_pct": "13.7",
            "window_macro_f1_mean_pct": "36.8",
            "window_macro_f1_std_pct": "14.3",
            "n_seeds": "3",
            "protocol": "Step1 cross-day (diagnostic baseline)",
            "source": "outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md",
        },
        {
            "model": "F cross-attn OOB + chirp",
            "model_id": "F_cross_attn_chirp_plain",
            "file_acc_mean_pct": "75.0",
            "file_acc_std_pct": "5.3",
            "file_macro_f1_mean_pct": "67.9",
            "file_macro_f1_std_pct": "6.8",
            "window_acc_mean_pct": "41.5",
            "window_acc_std_pct": "2.4",
            "window_macro_f1_mean_pct": "39.4",
            "window_macro_f1_std_pct": "2.1",
            "n_seeds": "5",
            "protocol": "Step1 cross-day (primary main model)",
            "source": "outputs/paper_ready_v3/step1_phase7_clean/STEP1_REPORT_FOR_GPT.md",
        },
    ]
    fields = list(rows[0].keys())
    write_csv(OUT / "table1_cross_day_main.csv", rows, fields)


def table2_fusion_chirp_ablation() -> None:
    rows = [
        {
            "fusion": "concat",
            "chirp": "no",
            "model_id": "D_concat_oob_plain",
            "file_acc_mean_pct": "18.3",
            "file_acc_std_pct": "18.6",
            "file_macro_f1_mean_pct": "12.3",
            "file_macro_f1_std_pct": "15.7",
            "n_seeds": "5",
            "source": "step1_phase7_clean",
        },
        {
            "fusion": "concat",
            "chirp": "yes",
            "model_id": "D_concat_chirp_plain",
            "file_acc_mean_pct": "9.7",
            "file_acc_std_pct": "2.0",
            "file_macro_f1_mean_pct": "3.1",
            "file_macro_f1_std_pct": "1.8",
            "n_seeds": "3",
            "source": "step1b_chirp_fusion_ablation",
        },
        {
            "fusion": "cross-attn",
            "chirp": "no",
            "model_id": "F_cross_attn_no_chirp_plain",
            "file_acc_mean_pct": "75.0",
            "file_acc_std_pct": "3.4",
            "file_macro_f1_mean_pct": "68.1",
            "file_macro_f1_std_pct": "4.0",
            "n_seeds": "3",
            "source": "step1b_chirp_fusion_ablation",
        },
        {
            "fusion": "cross-attn",
            "chirp": "yes",
            "model_id": "F_cross_attn_chirp_plain",
            "file_acc_mean_pct": "75.0",
            "file_acc_std_pct": "5.3",
            "file_macro_f1_mean_pct": "67.9",
            "file_macro_f1_std_pct": "6.8",
            "n_seeds": "5",
            "source": "step1_phase7_clean",
        },
    ]
    write_csv(OUT / "table2_fusion_chirp_ablation.csv", rows, list(rows[0].keys()))


def _load_deployment_summary() -> list[dict]:
    path = ROOT / "outputs" / "paper_ready_v2" / "deployment_shift_fixed_summary.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table3_deployment_shift() -> None:
    by_key: dict[tuple[str, str], dict] = {}

    def get_row(shift: str, cond: str) -> dict:
        key = (shift, cond)
        if key not in by_key:
            by_key[key] = {
                "shift_type": shift,
                "condition": cond,
                "cnn_file_acc_pct": "",
                "cnn_file_macro_f1_pct": "",
                "cnn_window_acc_pct": "",
                "hybrid_file_acc_pct": "",
                "hybrid_file_macro_f1_pct": "",
                "hybrid_window_acc_pct": "",
                "file_acc_winner": "",
                "notes": "",
                "source": "outputs/paper_ready_v2/deployment_shift_fixed_summary.csv",
            }
        return by_key[key]

    for r in _load_deployment_summary():
        exp = r["experiment"]
        parts = exp.split("_")
        if exp.startswith("config_loco_fold_"):
            fold = int(parts[3])
            shift, cond = "config LOCO", CONFIG_HELD_OUT.get(fold, f"fold_{fold}")
        elif exp.startswith("location_loco_fold_"):
            fold = int(parts[3])
            shift, cond = "location LOCO", LOCATION_HELD_OUT.get(fold, f"fold_{fold}")
        elif exp.startswith("distance_loco_fold_"):
            dist = parts[3]
            shift, cond = "distance LOCO", f"{dist} held-out"
        else:
            continue

        row = get_row(shift, cond)
        fa, ff1, wa = pct(float(r["file_acc"])), pct(float(r["file_macro_f1"])), pct(float(r["window_acc"]))
        if r["model_type"] == "osu_cnn":
            row["cnn_file_acc_pct"] = fa
            row["cnn_file_macro_f1_pct"] = ff1
            row["cnn_window_acc_pct"] = wa
        else:
            row["hybrid_file_acc_pct"] = fa
            row["hybrid_file_macro_f1_pct"] = ff1
            row["hybrid_window_acc_pct"] = wa

    rows = list(by_key.values())
    for row in rows:
        cnn = float(row["cnn_file_acc_pct"]) if row["cnn_file_acc_pct"] else 0.0
        hyb = float(row["hybrid_file_acc_pct"]) if row["hybrid_file_acc_pct"] else 0.0
        if hyb > cnn:
            row["file_acc_winner"] = "Hybrid"
        elif cnn > hyb:
            row["file_acc_winner"] = "CNN"
        else:
            row["file_acc_winner"] = "tie"

    order = {"config LOCO": 0, "location LOCO": 1, "distance LOCO": 2}
    rows.sort(key=lambda x: (order.get(x["shift_type"], 9), x["condition"]))

    notes_map = {
        ("location LOCO", "outdoor"): "Hybrid worse than CNN — do not over-claim location robustness",
        ("distance LOCO", "5m held-out"): "Similar file-acc; distance gains clearer at 10/15/20m",
    }
    for row in rows:
        row["notes"] = notes_map.get((row["shift_type"], row["condition"]), "")

    fields = [
        "shift_type",
        "condition",
        "cnn_file_acc_pct",
        "cnn_file_macro_f1_pct",
        "cnn_window_acc_pct",
        "hybrid_file_acc_pct",
        "hybrid_file_macro_f1_pct",
        "hybrid_window_acc_pct",
        "file_acc_winner",
        "notes",
        "source",
    ]
    write_csv(OUT / "table3_deployment_shift.csv", rows, fields)


def table4_cross_receiver_stress() -> None:
    rows = [
        {
            "direction": "RX1→RX2",
            "model": "CNN-IQ",
            "model_id": "A_cnn_iq",
            "file_acc_mean_pct": "4.2",
            "file_acc_std_pct": "0.0",
            "file_macro_f1_mean_pct": "0.4",
            "file_macro_f1_std_pct": "0.0",
            "n_seeds": "3",
            "oob_norm": "iq_rms (CNN) / n/a",
            "protocol": "Phase5-clean strict source-only",
        },
        {
            "direction": "RX1→RX2",
            "model": "F Hybrid",
            "model_id": "F_cross_attn_chirp_plain",
            "file_acc_mean_pct": "18.1",
            "file_acc_std_pct": "3.9",
            "file_macro_f1_mean_pct": "12.5",
            "file_macro_f1_std_pct": "1.2",
            "n_seeds": "3",
            "oob_norm": "ratio",
            "protocol": "Phase5-clean strict source-only",
        },
        {
            "direction": "RX2→RX1",
            "model": "CNN-IQ",
            "model_id": "A_cnn_iq",
            "file_acc_mean_pct": "23.6",
            "file_acc_std_pct": "15.3",
            "file_macro_f1_mean_pct": "17.8",
            "file_macro_f1_std_pct": "12.9",
            "n_seeds": "3",
            "oob_norm": "iq_rms (CNN) / n/a",
            "protocol": "Phase5-clean strict source-only",
        },
        {
            "direction": "RX2→RX1",
            "model": "F Hybrid",
            "model_id": "F_cross_attn_chirp_plain",
            "file_acc_mean_pct": "15.3",
            "file_acc_std_pct": "7.1",
            "file_macro_f1_mean_pct": "10.9",
            "file_macro_f1_std_pct": "7.1",
            "n_seeds": "3",
            "oob_norm": "ratio",
            "protocol": "Phase5-clean strict source-only",
        },
    ]
    write_csv(OUT / "table4_cross_receiver_stress.csv", rows, list(rows[0].keys()))


def table5_edge_deployment() -> None:
    src = ROOT / "outputs" / "paper_ready" / "edge_deployment_summary.csv"
    rows = []
    with src.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "model": r["model"],
                    "params": r["params"],
                    "latency_ms_bs1": f"{float(r['latency_ms_bs1']):.2f}",
                    "latency_ms_bs32": f"{float(r['latency_ms_bs32']):.2f}",
                    "latency_ms_bs64": f"{float(r['latency_ms_bs64']):.2f}",
                    "peak_gpu_mem_mb_bs32": f"{float(r['peak_gpu_mem_mb_bs32']):.1f}",
                    "source": "outputs/paper_ready/edge_deployment_summary.csv",
                    "notes": "Phase6 edge benchmark; Hybrid ~1.16M params, ~2.45 ms @ bs1",
                }
            )
    write_csv(OUT / "table5_edge_deployment.csv", rows, list(rows[0].keys()))


def main() -> None:
    table1_cross_day_main()
    table2_fusion_chirp_ablation()
    table3_deployment_shift()
    table4_cross_receiver_stress()
    table5_edge_deployment()
    print(f"Wrote final tables to {OUT}")


if __name__ == "__main__":
    main()
