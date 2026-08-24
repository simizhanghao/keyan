#!/usr/bin/env python3
"""Frozen receiver-level X6 aggregation and decision rule."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RECEIVERS = ("b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2")
MODELS = ("Shen-CIS", "Shen-RA", "B1-OOB", "C'-OOB", "C'-TrueIB")
SCALES = ("scale_0.5", "scale_0.70710678", "scale_1.41421356", "scale_2.0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows, values = [], {}
    for receiver in RECEIVERS:
        for model in MODELS:
            stem = model.replace("'", "prime")
            conditions = ("clean",) if model not in {"B1-OOB", "C'-OOB"} else ("clean", *SCALES, "shuffle", "neutral", "left_scale_0.5", "right_scale_0.5")
            for seed in range(5):
                for condition in conditions:
                    path = a.root / receiver / stem / f"seed_{seed}" / f"{condition}.json"
                    x = json.loads(path.read_text())
                    key = (receiver, model, seed, condition)
                    values[key] = x["accuracy"]
                    rows.append({"receiver": receiver, "model": model, "seed": seed, "condition": condition,
                                 "accuracy": x["accuracy"], "macro_f1": x["macro_f1"], "n_eval": x["n_eval"]})
    assert len(rows) == 630
    effects, decisions = [], {}
    for model in ("B1-OOB", "C'-OOB"):
        for receiver in RECEIVERS:
            mean = lambda condition: float(np.mean([values[(receiver, model, s, condition)] for s in range(5)]))
            clean = mean("clean")
            disrupt = clean - np.mean([mean("shuffle"), mean("neutral")])
            scale = clean - min(mean(c) for c in SCALES)
            lr = (clean - mean("left_scale_0.5")) - (clean - mean("right_scale_0.5"))
            effects.append({"model": model, "receiver": receiver, "clean_accuracy": clean,
                            "disruption_drop": float(disrupt), "worst_scale_drop": float(scale),
                            "left_minus_right_drop": float(lr)})
        arm = [x for x in effects if x["model"] == model]
        core_positive = sum(x["disruption_drop"] > 0 and x["worst_scale_drop"] > 0 for x in arm)
        med_disrupt = float(np.median([x["disruption_drop"] for x in arm]))
        med_scale = float(np.median([x["worst_scale_drop"] for x in arm]))
        go = core_positive >= 4 and med_disrupt >= 0.05 and med_scale >= 0.05
        lr_positive = sum(x["left_minus_right_drop"] > 0 for x in arm)
        strong = go and lr_positive >= 4 and np.median([x["left_minus_right_drop"] for x in arm]) > 0
        decisions[model] = {"GO": bool(go), "STRONG_GO": bool(strong), "core_positive_receivers": core_positive,
                            "median_disruption_drop": med_disrupt, "median_worst_scale_drop": med_scale,
                            "left_greater_right_receivers": lr_positive}
    overall = "STRONG_GO" if any(x["STRONG_GO"] for x in decisions.values()) else "GO" if any(x["GO"] for x in decisions.values()) else "DOWNGRADE"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"statistical_unit": "receiver", "runs": len(rows), "receiver_effects": effects,
               "model_decisions": decisions, "overall_decision": overall, "rows": rows}
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    with a.out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(json.dumps({"runs": len(rows), "overall_decision": overall, "model_decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
