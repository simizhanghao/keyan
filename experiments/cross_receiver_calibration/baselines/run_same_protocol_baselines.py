#!/usr/bin/env python3
"""Same-protocol SOTA-style baselines on frozen RF-HSTU embeddings."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAL))
sys.path.insert(0, str(CAL / "baselines"))

from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE, ROLE_SUPPORT  # noqa: E402
from common import (  # noqa: E402
    align_features,
    eval_linear_classifier_file_acc,
    eval_source_classifier_on_z,
    finetune_linear_head,
    load_classifier_head,
    load_role_data,
    read_split_csv,
    support_subset,
    train_linear_probe,
)
from run_rcpa_prototypes import build_class_prototypes, eval_prototype_file_acc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True)
    p.add_argument("--embeddings-npz", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--shot-ks", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--num-classes", type=int, default=24)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--head-epochs", type=int, default=50)
    p.add_argument("--head-lr", type=float, default=0.05)
    return p.parse_args()


def append_row(rows: list[dict], **kwargs) -> None:
    rows.append(kwargs)


def run_feature_alignment(
    rows: list[dict],
    *,
    args: argparse.Namespace,
    by_role: dict,
    head,
    device: torch.device,
) -> None:
    src = by_role[ROLE_SOURCE]
    cal = by_role[ROLE_CALIBRATION]
    qry = by_role[ROLE_QUERY]

    src_protos, src_labels = build_class_prototypes(src["z"], src["y"], args.num_classes)

    for method in ["mean_shift", "coral"]:
        qry_z = align_features(qry["z"], method, src["z"], cal["z"])
        qry_aligned = {**qry, "z": qry_z}

        acc_cls, f1_cls = eval_source_classifier_on_z(qry_aligned, head, device, args.num_classes)
        append_row(
            rows,
            method=f"feat_{method}_source_classifier",
            calibration_mode="unlabeled_target",
            trainable_params="head frozen (source)",
            direction=args.direction,
            seed=args.seed,
            split_seed=args.split_seed,
            shot_k=0,
            init="source",
            file_acc=acc_cls,
            macro_f1=f1_cls,
        )

        src_z_aligned = align_features(src["z"], method, src["z"], cal["z"])
        protos, plabels = build_class_prototypes(src_z_aligned, src["y"], args.num_classes)
        acc_proto, f1_proto = eval_prototype_file_acc(
            qry_aligned, protos, plabels, "cosine", args.num_classes
        )
        append_row(
            rows,
            method=f"feat_{method}_source_prototype",
            calibration_mode="unlabeled_target",
            trainable_params="none (prototype)",
            direction=args.direction,
            seed=args.seed,
            split_seed=args.split_seed,
            shot_k=0,
            init="source",
            file_acc=acc_proto,
            macro_f1=f1_proto,
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_rows = read_split_csv(Path(args.split_csv))
    by_role = load_role_data(Path(args.embeddings_npz), split_rows)

    src = by_role[ROLE_SOURCE]
    sup = by_role[ROLE_SUPPORT]
    qry = by_role[ROLE_QUERY]

    rows: list[dict] = []
    ckpt = Path(args.checkpoint)
    head = load_classifier_head(ckpt, device)
    head.eval()

    run_feature_alignment(rows, args=args, by_role=by_role, head=head, device=device)

    for k in args.shot_ks:
        sup_z, sup_y = support_subset(sup, k, args.split_seed)
        if len(sup_z) == 0:
            continue

        clf = train_linear_probe(sup_z, sup_y)
        acc, f1 = eval_linear_classifier_file_acc(qry, clf, num_classes=args.num_classes)
        append_row(
            rows,
            method="linear_probe_kshot",
            calibration_mode="k_shot_labeled",
            trainable_params="logistic head",
            direction=args.direction,
            seed=args.seed,
            split_seed=args.split_seed,
            shot_k=k,
            init="n/a",
            file_acc=acc,
            macro_f1=f1,
        )

        for init in ["source", "random"]:
            ft_head = finetune_linear_head(
                sup_z,
                sup_y,
                init=init,
                checkpoint=ckpt if init == "source" else None,
                seed=args.seed * 100 + args.split_seed,
                epochs=args.head_epochs,
                lr=args.head_lr,
                device=device,
                num_classes=args.num_classes,
            )
            acc_h, f1_h = eval_linear_classifier_file_acc(
                qry, ft_head, torch_mode=True, device=device, num_classes=args.num_classes
            )
            append_row(
                rows,
                method="head_finetune_kshot",
                calibration_mode="k_shot_labeled",
                trainable_params="linear head",
                direction=args.direction,
                seed=args.seed,
                split_seed=args.split_seed,
                shot_k=k,
                init=init,
                file_acc=acc_h,
                macro_f1=f1_h,
            )

    fields = [
        "method",
        "calibration_mode",
        "trainable_params",
        "direction",
        "seed",
        "split_seed",
        "shot_k",
        "init",
        "file_acc",
        "macro_f1",
    ]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"out_csv": str(out), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
