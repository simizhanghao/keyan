#!/usr/bin/env python3
"""Pseudo-proto TTA threshold sweep for appendix defense."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
CAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CAL_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import build_model, collect_train_embeddings, load_checkpoint, prepare_model_input, pseudo_proto_adapt
from lib.calibration_io import SplitWindowDataset, build_eval_args
from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE
from rfhstu.prototype import build_prototypes
from run_tta_negative_baseline import (
    collapse_stats,
    eval_prototype_query,
    file_level_vote,
    macro_f1,
    make_tta_args,
    rows_by_role,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rcpa-k5-acc", type=float, default=0.4583333333333333)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with Path(args.split_csv).open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    cal_rows = rows_by_role(split_rows, ROLE_CALIBRATION)
    qry_rows = rows_by_role(split_rows, ROLE_QUERY)
    src_rows = rows_by_role(split_rows, ROLE_SOURCE)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    eval_args = build_eval_args(ckpt, args.batch_size)
    ckpt_args = ckpt.get("args", {})
    input_norm = eval_args.input_norm

    cal_loader = DataLoader(SplitWindowDataset(cal_rows, input_norm), batch_size=args.batch_size, shuffle=False, num_workers=2)
    qry_loader = DataLoader(SplitWindowDataset(qry_rows, input_norm), batch_size=args.batch_size, shuffle=False, num_workers=2)
    src_loader = DataLoader(SplitWindowDataset(src_rows, input_norm), batch_size=args.batch_size, shuffle=False, num_workers=2)

    run_args = argparse.Namespace(adapt_steps=3, adapt_lr=1e-4, pseudo_threshold=0.9)
    results = []

    for thr in args.thresholds:
        model = build_model(eval_args, ckpt, device)
        train_z, train_y = collect_train_embeddings(model, src_loader, device, eval_args, ckpt_args)
        src_prototypes, src_labels = build_prototypes(train_z, train_y)
        ra = copy.copy(run_args)
        ra.pseudo_threshold = thr
        pseudo_args = make_tta_args(eval_args, ra, "pseudo_proto")
        adapted_proto, stats = pseudo_proto_adapt(
            model, cal_loader, device, pseudo_args, ckpt_args, src_prototypes, src_labels,
        )
        acc, f1, collapse = eval_prototype_query(
            model, qry_loader, device, eval_args, ckpt_args, adapted_proto, src_labels,
        )
        collapsed = collapse["top1_file_mass"] >= 0.5 or collapse["num_file_classes_predicted"] <= 3
        results.append({
            "threshold": thr,
            "num_pseudo_selected": stats.get("num_pseudo_selected", 0),
            "num_classes_updated": stats.get("num_classes_updated", 0),
            "file_acc": acc,
            "macro_f1": f1,
            "top1_file_mass": collapse["top1_file_mass"],
            "num_file_classes_predicted": collapse["num_file_classes_predicted"],
            "collapse_flag": int(collapsed),
            "gap_to_rcpa_k5_pp": (acc - args.rcpa_k5_acc) * 100,
        })

    fields = list(results[0].keys())
    with (out_dir / "tta_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
