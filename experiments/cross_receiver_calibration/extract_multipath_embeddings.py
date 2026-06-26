#!/usr/bin/env python3
"""Extract main/oob/fused embeddings for split manifest in one pass."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DIAG_ROOT = Path(__file__).resolve().parent.parent / "cross_receiver_diagnosis"
sys.path.insert(0, str(DIAG_ROOT))

from evaluate import build_model, load_checkpoint
from extract_calibration_embeddings import SplitWindowDataset, build_eval_args
from lib.extraction import extract_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with Path(args.split_csv).open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    eval_args = build_eval_args(ckpt, args.batch_size)
    ckpt_args = ckpt.get("args", {})
    model = build_model(eval_args, ckpt, device)
    model.eval()

    ds = SplitWindowDataset(split_rows, input_norm=eval_args.input_norm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    main_list, oob_list, fused_list, logit_list = [], [], [], []
    labels, roles, devices, win_idx, row_idx = [], [], [], [], []

    for batch in tqdm(loader, desc="extract multipath"):
        iq = batch["iq"].to(device)
        out = extract_batch(model, iq, ckpt_args, eval_args.model_type)
        main_list.append(out["main"].cpu().numpy())
        oob_list.append(out["oob"].cpu().numpy())
        fused_list.append(out["fused"].cpu().numpy())
        logit_list.append(out["logits"].cpu().numpy())
        labels.extend(batch["label"].tolist())
        roles.extend(batch["role"])
        devices.extend(batch["device_id"])
        win_idx.extend(batch["window_index"])
        row_idx.extend(batch["row_idx"])

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        main=np.concatenate(main_list, axis=0),
        oob=np.concatenate(oob_list, axis=0),
        fused=np.concatenate(fused_list, axis=0),
        logits=np.concatenate(logit_list, axis=0),
        labels=np.array(labels, dtype=np.int64),
        roles=np.array(roles, dtype=object),
        device_ids=np.array(devices, dtype=np.int64),
        window_indices=np.array(win_idx, dtype=np.int64),
        row_indices=np.array(row_idx, dtype=np.int64),
    )
    meta = {"checkpoint": args.checkpoint, "split_csv": args.split_csv}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out_path} shape={np.concatenate(fused_list, axis=0).shape}")


if __name__ == "__main__":
    main()
