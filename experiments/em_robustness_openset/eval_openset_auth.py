#!/usr/bin/env python3
"""Evaluate open-set authentication scores (MSP, energy, prototype, Mahalanobis)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    compute_auroc,
    compute_eer,
    compute_fpr_at_tpr,
    collect_file_features,
    make_eval_loader,
    populate_args_from_ckpt,
    resolve_device,
    save_json,
)
from evaluate import build_model, load_checkpoint  # noqa: E402
from rfhstu.em_perturbations import EmPerturbConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--openset-manifest", default=None)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--samples-per-file", type=int, default=256)
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--perturb", choices=["clean", "awgn", "cfo", "nbi"], default="clean")
    p.add_argument("--perturb-strength", type=float, default=None)
    return p.parse_args()


def build_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    protos = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos[c] = embeddings[mask].mean(axis=0)
    return protos


def mahalanobis_min_dist(emb: np.ndarray, protos: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    dists = []
    for p in protos:
        diff = emb - p
        d = np.einsum("ij,ij->i", diff, diff @ cov_inv)
        dists.append(d)
    return np.min(np.stack(dists, axis=1), axis=1)


def load_unknown_map(manifest_path: Path, root: Path) -> dict[str, int]:
    unk_map: dict[str, int] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["path"])
            if not p.is_absolute():
                p = root / p
            unk_map[str(p)] = int(r.get("is_unknown_device", "0"))
    return unk_map


def openset_labels(rows: list[dict], unk_map: dict[str, int]) -> np.ndarray:
    return np.array([0 if unk_map.get(r["file_path"], 0) else 1 for r in rows])


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_t = float(np.median(scores))
    best_eer = 1.0
    for t in np.unique(scores):
        pred = scores >= t
        far = float(np.mean(pred[y_true == 0])) if np.any(y_true == 0) else 0.0
        frr = float(np.mean(~pred[y_true == 1])) if np.any(y_true == 1) else 0.0
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer = eer
            best_t = float(t)
    return best_t


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    ckpt_args = populate_args_from_ckpt(args, ckpt)
    model = build_model(args, ckpt, device)

    manifest = args.openset_manifest or args.manifest
    args.manifest = manifest
    unk_map = load_unknown_map(ROOT / manifest, Path(args.root))

    perturb_cfg = None
    if args.perturb == "awgn":
        perturb_cfg = EmPerturbConfig(awgn_snr_db=args.perturb_strength or 10.0)
    elif args.perturb == "cfo":
        perturb_cfg = EmPerturbConfig(cfo_norm=args.perturb_strength or 0.05)
    elif args.perturb == "nbi":
        perturb_cfg = EmPerturbConfig(narrowband_sir_db=args.perturb_strength or 10.0)

    val_loader = make_eval_loader(args, args.val_split)
    test_loader = make_eval_loader(args, args.test_split)
    val_rows = collect_file_features(model, val_loader, device, args, ckpt_args, perturb_cfg)
    test_rows = collect_file_features(model, test_loader, device, args, ckpt_args, perturb_cfg)

    y_val = openset_labels(val_rows, unk_map)
    y_test = openset_labels(test_rows, unk_map)

    train_loader = make_eval_loader(args, "train")
    train_rows = collect_file_features(model, train_loader, device, args, ckpt_args, None)
    train_emb = np.stack([r["embedding"] for r in train_rows])
    train_y = np.array([r["label"] for r in train_rows])
    num_classes = int(ckpt.get("num_classes", ckpt_args.get("num_classes", 24)))
    protos = build_prototypes(train_emb, train_y, num_classes)
    cov = np.cov(train_emb, rowvar=False) + np.eye(train_emb.shape[1]) * 1e-4
    cov_inv = np.linalg.inv(cov)

    test_emb = np.stack([r["embedding"] for r in test_rows])
    msp = np.array([r["msp"] for r in test_rows])
    energy = -np.array([r["energy"] for r in test_rows])
    proto_min = np.linalg.norm(test_emb[:, None, :] - protos[None, :, :], axis=2).min(axis=1)
    maha = mahalanobis_min_dist(test_emb, protos, cov_inv)

    val_emb = np.stack([r["embedding"] for r in val_rows])
    val_msp = np.array([r["msp"] for r in val_rows])
    val_energy = -np.array([r["energy"] for r in val_rows])
    val_proto = np.linalg.norm(val_emb[:, None, :] - protos[None, :, :], axis=2).min(axis=1)
    val_maha = mahalanobis_min_dist(val_emb, protos, cov_inv)

    scorers = {
        "msp": (msp, val_msp, True),
        "energy": (energy, val_energy, True),
        "proto_dist": (-proto_min, -val_proto, True),
        "mahalanobis": (-maha, -val_maha, True),
    }

    out_rows = []
    for name, (ts, vs, higher) in scorers.items():
        s_test = ts if higher else -ts
        s_val = vs if higher else -vs
        thr = pick_threshold(y_val, s_val)
        pred_known = s_test >= thr
        far = float(np.mean(pred_known[y_test == 0])) if np.any(y_test == 0) else 0.0
        frr = float(np.mean(~pred_known[y_test == 1])) if np.any(y_test == 1) else 0.0
        known_idx = [i for i, r in enumerate(test_rows) if y_test[i] == 1]
        known_acc = float(np.mean([test_rows[i]["correct"] for i in known_idx])) if known_idx else float("nan")
        out_rows.append(
            {
                "scorer": name,
                "perturb": args.perturb,
                "perturb_strength": args.perturb_strength,
                "auroc": compute_auroc(y_test, s_test),
                "eer": compute_eer(y_test, s_test),
                "fpr_at_95tpr": compute_fpr_at_tpr(y_test, s_test),
                "far": far,
                "frr": frr,
                "known_acc": known_acc,
                "threshold": thr,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    save_json(out_path.with_suffix(".json"), {"manifest": manifest, "checkpoint": args.checkpoint})
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
