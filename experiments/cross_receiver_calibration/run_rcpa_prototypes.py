#!/usr/bin/env python3
"""Run RCPA prototype calibration and source classifier baselines."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE, ROLE_SUPPORT, sample_k_support_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True)
    p.add_argument("--embeddings-npz", required=True)
    p.add_argument("--direction", default="rx1_to_rx2")
    p.add_argument("--model", default="ours_fused")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--shot-ks", nargs="+", type=int, default=[0, 1, 5, 10])
    p.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--out-csv", required=True)
    p.add_argument("--alpha-sensitivity-csv", default=None)
    p.add_argument("--alpha-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--num-classes", type=int, default=24)
    return p.parse_args()


def macro_f1(labels: list[int], preds: list[int], num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for y, p in zip(labels, preds) if y == c and p == c)
        fp = sum(1 for y, p in zip(labels, preds) if y != c and p == c)
        fn = sum(1 for y, p in zip(labels, preds) if y == c and p != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def dist_matrix(z: np.ndarray, protos: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        z_n = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
        p_n = protos / np.clip(np.linalg.norm(protos, axis=1, keepdims=True), 1e-8, None)
        return 1.0 - z_n @ p_n.T
    return np.linalg.norm(z[:, None, :] - protos[None, :, :], axis=-1)


def build_class_prototypes(z: np.ndarray, y: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    protos, labels = [], []
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            protos.append(z[mask].mean(axis=0))
            labels.append(c)
    return np.stack(protos, axis=0), np.array(labels, dtype=np.int64)


def predict_prototype(z: np.ndarray, protos: np.ndarray, proto_labels: np.ndarray, metric: str) -> np.ndarray:
    d = dist_matrix(z, protos, metric)
    return proto_labels[d.argmin(axis=1)]


def blend_prototypes(
    src_protos: np.ndarray,
    src_labels: np.ndarray,
    tgt_protos: np.ndarray,
    tgt_labels: np.ndarray,
    alpha: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(tgt_protos) == 0:
        return src_protos, src_labels
    src_map = {int(l): i for i, l in enumerate(src_labels)}
    tgt_map = {int(l): i for i, l in enumerate(tgt_labels)}
    blend_p, blend_l = [], []
    for c in range(num_classes):
        if c in src_map and c in tgt_map:
            blend_p.append(alpha * src_protos[src_map[c]] + (1.0 - alpha) * tgt_protos[tgt_map[c]])
            blend_l.append(c)
    if not blend_p:
        return None
    return np.stack(blend_p), np.array(blend_l, dtype=np.int64)


def file_level_vote(
    device_ids: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray | None = None,
) -> tuple[list[int], list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    grouped_pred: dict[int, list[int]] = defaultdict(list)
    grouped_score: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, dev in enumerate(device_ids):
        grouped[dev].append(int(labels[i]))
        grouped_pred[dev].append(int(preds[i]))
        if scores is not None:
            grouped_score[dev].append(scores[i])
    file_labels, file_preds = [], []
    for dev in sorted(grouped.keys()):
        file_labels.append(grouped[dev][0])
        if scores is not None:
            mean_score = np.mean(np.stack(grouped_score[dev], axis=0), axis=0)
            file_preds.append(int(mean_score.argmax()))
        else:
            vals, counts = np.unique(grouped_pred[dev], return_counts=True)
            file_preds.append(int(vals[counts.argmax()]))
    return file_labels, file_preds


def eval_prototype_file_acc(
    qry: dict,
    blend_protos: np.ndarray,
    blend_labels: np.ndarray,
    metric: str,
    num_classes: int,
) -> tuple[float, float]:
    file_labels, file_preds = [], []
    for dev in sorted(set(qry["device"].tolist())):
        mask = qry["device"] == dev
        z_mean = qry["z"][mask].mean(axis=0, keepdims=True)
        pred = predict_prototype(z_mean, blend_protos, blend_labels, metric)[0]
        file_labels.append(int(qry["y"][mask][0]))
        file_preds.append(int(pred))
    acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    return acc, macro_f1(file_labels, file_preds, num_classes)


def load_role_data(npz_path: Path, split_rows: list[dict]):
    data = np.load(npz_path, allow_pickle=True)
    z = data["embeddings"]
    logits = data["logits"]
    labels = data["labels"]
    device_ids = data["device_ids"]
    win_idx = data["window_indices"]
    row_indices = data["row_indices"]
    roles = np.array([split_rows[int(i)]["role"] for i in row_indices], dtype=object)

    by_role = {}
    for role in [ROLE_SOURCE, ROLE_CALIBRATION, ROLE_SUPPORT, ROLE_QUERY]:
        mask = roles == role
        by_role[role] = {
            "z": z[mask],
            "logits": logits[mask],
            "y": labels[mask],
            "device": device_ids[mask],
            "win": win_idx[mask],
        }
    return by_role


def append_result(
    results: list[dict],
    *,
    method: str,
    args: argparse.Namespace,
    shot_k: int,
    alpha: float | str,
    file_acc: float,
    macro_f1_val: float,
) -> None:
    results.append({
        "method": method,
        "direction": args.direction,
        "model": args.model,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "shot_k": shot_k,
        "alpha": alpha,
        "distance": args.distance if method != "source_classifier" else "",
        "file_acc": file_acc,
        "macro_f1": macro_f1_val,
    })


def main() -> None:
    args = parse_args()
    with Path(args.split_csv).open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    by_role = load_role_data(Path(args.embeddings_npz), split_rows)
    src = by_role[ROLE_SOURCE]
    qry = by_role[ROLE_QUERY]
    sup = by_role[ROLE_SUPPORT]

    num_classes = args.num_classes
    results: list[dict] = []
    alpha_rows: list[dict] = []

    q_logits = torch.from_numpy(qry["logits"])
    q_probs = F.softmax(q_logits, dim=-1).numpy()
    cls_preds = q_probs.argmax(axis=1)
    file_labels, file_preds = file_level_vote(qry["device"], qry["y"], cls_preds, q_probs)
    cls_acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    cls_f1 = macro_f1(file_labels, file_preds, num_classes)
    append_result(results, method="source_classifier", args=args, shot_k=-1, alpha="", file_acc=cls_acc, macro_f1_val=cls_f1)

    src_protos, src_labels = build_class_prototypes(src["z"], src["y"], num_classes)

    for k in args.shot_ks:
        support_indices = set(sample_k_support_indices(k, args.split_seed))
        support_mask = np.array([int(w) in support_indices for w in sup["win"]], dtype=bool)

        if k > 0:
            sup_z = sup["z"][support_mask]
            sup_y = sup["y"][support_mask]
            tgt_protos, tgt_labels = build_class_prototypes(sup_z, sup_y, num_classes)
        else:
            tgt_protos, tgt_labels = np.zeros((0, src_protos.shape[1])), np.array([], dtype=np.int64)

        method_configs = [
            ("RCPA-S", 1.0, "source"),
            ("RCPA-T", 0.0, "target"),
            ("RCPA-B", 0.5, "blend"),
        ]
        for method, alpha, mode in method_configs:
            if k == 0 and mode == "target":
                continue
            if mode == "source":
                blend_protos, blend_labels = src_protos, src_labels
            elif mode == "target":
                if len(tgt_protos) == 0:
                    continue
                blend_protos, blend_labels = tgt_protos, tgt_labels
            else:
                blended = blend_prototypes(src_protos, src_labels, tgt_protos, tgt_labels, alpha, num_classes)
                if blended is None:
                    continue
                blend_protos, blend_labels = blended

            acc, f1 = eval_prototype_file_acc(qry, blend_protos, blend_labels, args.distance, num_classes)
            append_result(results, method=method, args=args, shot_k=k, alpha=alpha, file_acc=acc, macro_f1_val=f1)

        if k > 0 and args.alpha_sensitivity_csv:
            for alpha in args.alpha_values:
                if alpha == 0.0:
                    blend_protos, blend_labels = tgt_protos, tgt_labels
                elif alpha == 1.0:
                    blend_protos, blend_labels = src_protos, src_labels
                else:
                    blended = blend_prototypes(src_protos, src_labels, tgt_protos, tgt_labels, alpha, num_classes)
                    if blended is None:
                        continue
                    blend_protos, blend_labels = blended
                acc, f1 = eval_prototype_file_acc(qry, blend_protos, blend_labels, args.distance, num_classes)
                alpha_rows.append({
                    "direction": args.direction,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "shot_k": k,
                    "alpha": alpha,
                    "distance": args.distance,
                    "file_acc": acc,
                    "macro_f1": f1,
                })

    fields = [
        "method", "direction", "model", "seed", "split_seed",
        "shot_k", "alpha", "distance", "file_acc", "macro_f1",
    ]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    if args.alpha_sensitivity_csv and alpha_rows:
        ap = Path(args.alpha_sensitivity_csv)
        af = ["direction", "seed", "split_seed", "shot_k", "alpha", "distance", "file_acc", "macro_f1"]
        with ap.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=af)
            w.writeheader()
            w.writerows(alpha_rows)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
