#!/usr/bin/env python3
"""TTA negative baseline: unsupervised adaptation on calibration Block A only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CAL_ROOT))

from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE
from run_rcpa_prototypes import file_level_vote, macro_f1

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import (
    build_model,
    collect_train_embeddings,
    configure_tta,
    load_checkpoint,
    prepare_model_input,
    pseudo_proto_adapt,
    restore_model_state,
    set_bn_train_only,
    tent_adapt_batch,
)
from rfhstu.prototype import build_prototypes

from lib.calibration_io import SplitWindowDataset, build_eval_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTA negative baseline (cal-only adaptation)")
    p.add_argument("--split-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--direction", default="rx1_to_rx2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--adapt-steps", type=int, default=3)
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--pseudo-threshold", type=float, default=0.9)
    p.add_argument("--rcpa-summary", default=None, help="Optional summary_full.csv for RCPA-T reference")
    return p.parse_args()


def make_tta_args(base: argparse.Namespace, run_args: argparse.Namespace, adapt_mode: str) -> argparse.Namespace:
    ns = copy.copy(base)
    ns.adapt_mode = adapt_mode
    ns.adapt_steps = run_args.adapt_steps
    ns.adapt_lr = run_args.adapt_lr
    ns.tent_episodic = False
    ns.tent_steps = run_args.adapt_steps
    ns.tent_lr = run_args.adapt_lr
    ns.tta_mode = "none"
    ns.pseudo_threshold = run_args.pseudo_threshold
    ns.pseudo_topk_per_class = 0
    ns.pseudo_min_per_class = 1
    ns.prototype_momentum = 0.5
    return ns


def rows_by_role(split_rows: list[dict], role: str) -> list[dict]:
    return [r for r in split_rows if r["role"] == role]


@torch.no_grad()
def eval_classifier_query(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_args: argparse.Namespace,
    ckpt_args: dict,
) -> tuple[float, float, dict]:
    model.eval()
    labels, preds, probs_list, devices = [], [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        out = model(prepare_model_input(iq, eval_args, ckpt_args))
        prob = F.softmax(out["logits"], dim=-1).cpu()
        pred = prob.argmax(dim=-1)
        labels.extend(batch["label"].tolist())
        preds.extend(pred.tolist())
        probs_list.append(prob.numpy())
        devices.extend(batch["device_id"])
    probs = np.concatenate(probs_list, axis=0)
    file_labels, file_preds = file_level_vote(
        np.array(devices), np.array(labels), np.array(preds), probs,
    )
    acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    f1 = macro_f1(file_labels, file_preds, 24)
    collapse = collapse_stats(file_preds, preds)
    return acc, f1, collapse


@torch.no_grad()
def eval_prototype_query(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_args: argparse.Namespace,
    ckpt_args: dict,
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> tuple[float, float, dict]:
    model.eval()
    labels, preds, devices = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        out = model(prepare_model_input(iq, eval_args, ckpt_args))
        emb = F.normalize(out["embedding"].detach().cpu(), dim=-1)
        scores = emb @ prototypes.T
        idx = scores.argmax(dim=-1)
        pred = prototype_labels[idx]
        labels.extend(batch["label"].tolist())
        preds.extend(pred.tolist())
        devices.extend(batch["device_id"])
    file_labels, file_preds = file_level_vote(
        np.array(devices), np.array(labels), np.array(preds),
    )
    acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    f1 = macro_f1(file_labels, file_preds, 24)
    collapse = collapse_stats(file_preds, preds)
    return acc, f1, collapse


def collapse_stats(file_preds: list[int], window_preds: list[int]) -> dict:
    fc = Counter(file_preds)
    wc = Counter(window_preds)
    top_file, top_file_n = fc.most_common(1)[0] if fc else (-1, 0)
    top_win, top_win_n = wc.most_common(1)[0] if wc else (-1, 0)
    n_file, n_win = max(len(file_preds), 1), max(len(window_preds), 1)
    return {
        "top1_file_class": top_file,
        "top1_file_mass": top_file_n / n_file,
        "num_file_classes_predicted": len(fc),
        "top1_window_class": top_win,
        "top1_window_mass": top_win_n / n_win,
        "num_window_classes_predicted": len(wc),
    }


def entropy_min_adapt(
    model: torch.nn.Module,
    cal_loader: DataLoader,
    device: torch.device,
    eval_args: argparse.Namespace,
    ckpt_args: dict,
    tta_args: argparse.Namespace,
) -> str:
    tta_mode, optimizer, episodic = configure_tta(model, tta_args)
    if tta_mode != "entropy_min":
        return tta_mode
    set_bn_train_only(model)
    for batch in tqdm(cal_loader, desc="entropy_min adapt (cal only)"):
        iq = batch["iq"].to(device)
        model_input = prepare_model_input(iq, eval_args, ckpt_args)
        restore_model_state(model, episodic)
        tent_adapt_batch(model, model_input, tta_args, optimizer)
    model.eval()
    return tta_mode


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

    cal_loader = DataLoader(SplitWindowDataset(cal_rows, input_norm), batch_size=args.batch_size, shuffle=True, num_workers=2)
    qry_loader = DataLoader(SplitWindowDataset(qry_rows, input_norm), batch_size=args.batch_size, shuffle=False, num_workers=2)
    src_loader = DataLoader(SplitWindowDataset(src_rows, input_norm), batch_size=args.batch_size, shuffle=False, num_workers=2)

    results: list[dict] = []

    def record(method: str, acc: float, f1: float, collapse: dict, extra: dict | None = None):
        row = {
            "method": method,
            "direction": args.direction,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "file_acc": acc,
            "macro_f1": f1,
            **collapse,
        }
        if extra:
            row.update(extra)
        results.append(row)

    # --- 1. Source classifier (no adaptation) ---
    model = build_model(eval_args, ckpt, device)
    acc, f1, collapse = eval_classifier_query(model, qry_loader, device, eval_args, ckpt_args)
    record("source_classifier", acc, f1, collapse)

    # --- 2. Entropy minimization TTA on cal Block A only ---
    model = build_model(eval_args, ckpt, device)
    tta_args = make_tta_args(eval_args, args, "entropy_min")
    mode_used = entropy_min_adapt(model, cal_loader, device, eval_args, ckpt_args, tta_args)
    acc, f1, collapse = eval_classifier_query(model, qry_loader, device, eval_args, ckpt_args)
    record("entropy_min_tta", acc, f1, collapse, {"tta_mode_used": mode_used})

    # --- 3. Pseudo-label prototype (cal Block A only) ---
    model = build_model(eval_args, ckpt, device)
    train_z, train_y = collect_train_embeddings(model, src_loader, device, eval_args, ckpt_args)
    src_prototypes, src_labels = build_prototypes(train_z, train_y)
    pseudo_args = make_tta_args(eval_args, args, "pseudo_proto")
    adapted_proto, pseudo_stats = pseudo_proto_adapt(
        model, cal_loader, device, pseudo_args, ckpt_args, src_prototypes, src_labels,
    )
    acc, f1, collapse = eval_prototype_query(
        model, qry_loader, device, eval_args, ckpt_args, adapted_proto, src_labels,
    )
    record("pseudo_proto_tta", acc, f1, collapse, pseudo_stats)

    # --- 4. RCPA-T reference from existing summary if provided ---
    if args.rcpa_summary:
        with Path(args.rcpa_summary).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (
                    row["direction"] == args.direction
                    and int(row["seed"]) == args.seed
                    and int(row["split_seed"]) == args.split_seed
                    and row["method"] == "RCPA-T"
                ):
                    for k in [1, 3, 5, 10]:
                        if int(row["shot_k"]) == k:
                            results.append({
                                "method": f"RCPA-T_K{k}",
                                "direction": args.direction,
                                "seed": args.seed,
                                "split_seed": args.split_seed,
                                "file_acc": float(row["file_acc"]),
                                "macro_f1": float(row["macro_f1"]),
                                "top1_file_mass": "",
                                "note": "from RCPA full/summary (labeled support)",
                            })

    fields = [
        "method", "direction", "seed", "split_seed", "file_acc", "macro_f1",
        "top1_file_class", "top1_file_mass", "num_file_classes_predicted",
        "top1_window_class", "top1_window_mass", "num_window_classes_predicted",
    ]
    out_csv = out_dir / "summary_tta_negative.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    meta = {
        "protocol": "Adapt on target calibration Block A only; evaluate query Block C+D.",
        "no_support_labels": True,
        "no_query_adaptation": True,
        "cal_windows_per_device": 64,
    }
    (out_dir / "tta_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
