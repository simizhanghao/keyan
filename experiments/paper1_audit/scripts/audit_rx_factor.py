#!/usr/bin/env python3
"""Day4 RX-factor attribution. Frozen 1C C'. No training, no Day5, no R0/R6 rerun."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/matched_seed0")
CLEAN = "C_full_ratio"
ARMS = ("tilt", "oob_scale", "gain", "phase", "noise", "spec", "nonspec")
SEEDS = [0, 1, 2, 3, 4]
D_FULL = 30.3
LARGE_PP = 15.0
FAMILY_PP = 18.2
SPEC_WEAK_PP = 12.1


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


def pred_map(model: str, seed: int) -> dict[tuple[str, int], int]:
    path = ROOT / "eval_val" / model / f"seed_{seed}" / "predictions.csv"
    out: dict[tuple[str, int], int] = {}
    if not path.is_file():
        return out
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["file_path"], int(row["window_index"]))] = int(row["pred"])
    return out


def flip_rate(clean: dict[tuple[str, int], int], corrupt: dict[tuple[str, int], int]) -> float | None:
    keys = set(clean) & set(corrupt)
    if not keys:
        return None
    n_flip = sum(clean[k] != corrupt[k] for k in keys)
    return n_flip / len(keys)


def classify(mean_drop: dict[str, float]) -> dict:
    d_spec = mean_drop["spec"]
    d_nonspec = mean_drop["nonspec"]
    d_phase = mean_drop["phase"]
    d_noise = mean_drop["noise"]
    d_scale = mean_drop["oob_scale"]
    singles = {k: mean_drop[k] for k in ("tilt", "oob_scale", "gain", "phase", "noise")}
    max_single = max(singles.values())
    unique_phase = d_phase >= LARGE_PP and d_phase == max_single and list(singles.values()).count(d_phase) == 1

    if d_spec >= FAMILY_PP and d_nonspec >= FAMILY_PP:
        case = "mixed"
    elif d_spec >= FAMILY_PP and d_nonspec < FAMILY_PP and max(d_phase, d_noise) < LARGE_PP:
        case = "magnitude_candidate"
    elif d_nonspec >= FAMILY_PP and d_spec < SPEC_WEAK_PP and d_noise >= LARGE_PP:
        case = "noise"
    elif d_nonspec >= FAMILY_PP and d_spec < SPEC_WEAK_PP and unique_phase:
        case = "phase_main_path"
    elif max_single < LARGE_PP and d_spec < FAMILY_PP and d_nonspec < FAMILY_PP:
        case = "compound"
    else:
        case = "unclassified"

    canonicalizer_go = case == "magnitude_candidate" and d_scale >= LARGE_PP
    need_oob_tilt = case == "magnitude_candidate" and d_scale < LARGE_PP
    return {
        "case": case,
        "canonicalizer_go": canonicalizer_go,
        "need_oob_only_tilt_localization": need_oob_tilt,
        "note": (
            "Case 1 is magnitude-family candidate only. Canonicalizer GO requires "
            "D_oob_scale >= 15pp. R_spec alone does not authorize DCT. "
            "Non-additivity cannot by itself be read as factor interaction: "
            "arms use independent perturbation draws plus network nonlinearity."
        ),
    }


def main() -> int:
    missing: list[str] = []
    window_rows: dict[str, list[float]] = {arm: [] for arm in ARMS}
    file_rows: dict[str, list[float]] = {arm: [] for arm in ARMS}
    flip_rows: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_arm: dict[str, dict] = {}

    for arm in ARMS:
        model = f"C_full_ratio_rx_{arm}"
        per_seed = []
        for seed in SEEDS:
            clean_m = load_metrics(CLEAN, seed)
            arm_m = load_metrics(model, seed)
            if clean_m is None or arm_m is None:
                missing.append(f"{arm}:{seed}")
                continue
            if arm_m.get("rx_factor") != arm:
                raise SystemExit(f"{model} seed {seed} rx_factor={arm_m.get('rx_factor')}")
            if arm_m.get("day5_used") is not False:
                raise SystemExit(f"{model} seed {seed} used Day5")
            drop_w = 100.0 * clean_m["window_acc"] - 100.0 * arm_m["window_acc"]
            drop_f = 100.0 * clean_m["file_acc"] - 100.0 * arm_m["file_acc"]
            flips = flip_rate(pred_map(CLEAN, seed), pred_map(model, seed))
            window_rows[arm].append(round(drop_w, 1))
            file_rows[arm].append(round(drop_f, 1))
            if flips is not None:
                flip_rows[arm].append(round(100.0 * flips, 1))
            per_seed.append(
                {
                    "seed": seed,
                    "clean_window_pct": pct(clean_m["window_acc"]),
                    "arm_window_pct": pct(arm_m["window_acc"]),
                    "drop_window_pp": round(drop_w, 1),
                    "clean_file_pct": pct(clean_m["file_acc"]),
                    "arm_file_pct": pct(arm_m["file_acc"]),
                    "drop_file_pp": round(drop_f, 1),
                    "flip_rate_pct": None if flips is None else round(100.0 * flips, 1),
                    "rx_factor": arm_m.get("rx_factor"),
                    "checkpoint_sha256": arm_m.get("checkpoint_sha256"),
                    "eval_rng_seed": arm_m.get("eval_rng_seed"),
                }
            )
        per_arm[arm] = {
            "per_seed": per_seed,
            "window_drop_pp": window_rows[arm],
            "file_drop_pp": file_rows[arm],
            "flip_rate_pct": flip_rows[arm],
            "window_drop_mean_pp": round(statistics.mean(window_rows[arm]), 1) if window_rows[arm] else None,
            "window_drop_std_pp": round(statistics.stdev(window_rows[arm]), 1) if len(window_rows[arm]) > 1 else None,
        }

    mean_drop = {arm: per_arm[arm]["window_drop_mean_pp"] for arm in ARMS}
    complete = not missing and all(v is not None for v in mean_drop.values())
    decision = classify({k: float(v) for k, v in mean_drop.items()}) if complete else {"case": "incomplete"}

    payload = {
        "day5_used": False,
        "no_training": True,
        "r0": CLEAN,
        "r6_window_drop_frozen_pp": D_FULL,
        "r6_not_rerun": True,
        "primary": "window drop = clean C' − factor C'",
        "file_acc_not_deciding": True,
        "thresholds_pp": {
            "individually_large": LARGE_PP,
            "family_majority": FAMILY_PP,
            "spec_weak": SPEC_WEAK_PP,
            "d_full": D_FULL,
        },
        "missing": missing,
        "complete": complete,
        "arms": per_arm,
        "mean_window_drop_pp": mean_drop,
        "decision": decision,
    }

    lines = [
        "# Day4 RX-factor attribution",
        "",
        "Frozen 1C C' checkpoints. No retraining. Day5 unused. R0/R6 not rerun.",
        f"Primary: window drop vs clean C'. Frozen D_full = {D_FULL:.1f}±2.0 pp.",
        "File-Acc is recorded and does not decide cases.",
        "Independent RNG per arm: non-additivity ≠ interaction by itself.",
        "",
        "## Window drop (pp)",
        "",
        "| Arm | s0 | s1 | s2 | s3 | s4 | Mean±Std | vs full 30.3 |",
        "| --- | --: | --: | --: | --: | --: | -------: | -----------: |",
    ]
    for arm in ARMS:
        shown = [(f"{window_rows[arm][i]:.1f}" if i < len(window_rows[arm]) else "?") for i in range(5)]
        vs_full = ""
        if per_arm[arm]["window_drop_mean_pp"] is not None:
            vs_full = f"{per_arm[arm]['window_drop_mean_pp'] - D_FULL:+.1f}"
        lines.append(
            f"| {arm} | {' | '.join(shown)} | {mean_std(window_rows[arm])} | {vs_full} |"
        )
    lines.extend(
        [
            "",
            "## File drop (pp, not deciding)",
            "",
            "| Arm | s0 | s1 | s2 | s3 | s4 | Mean±Std |",
            "| --- | --: | --: | --: | --: | --: | -------: |",
        ]
    )
    for arm in ARMS:
        shown = [(f"{file_rows[arm][i]:.1f}" if i < len(file_rows[arm]) else "?") for i in range(5)]
        lines.append(f"| {arm} | {' | '.join(shown)} | {mean_std(file_rows[arm])} |")
    lines.extend(
        [
            "",
            f"Frozen case (computed, does not open training): **{decision.get('case')}**",
            f"Canonicalizer GO: {decision.get('canonicalizer_go', False)}",
            f"Need OOB-only tilt localization: {decision.get('need_oob_only_tilt_localization', False)}",
            "",
            decision.get("note", ""),
            "",
            "PAPER1_AUDIT_REPORT / Day5 / 1D / 1E / RCOF / utility / DCT are not opened here.",
            "",
        ]
    )
    if missing:
        lines.extend([f"missing (not a verdict): {missing}", ""])

    out_json = ROOT / "rx_factor_attribution.json"
    out_md = ROOT / "rx_factor_attribution.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
