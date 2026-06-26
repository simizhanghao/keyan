#!/usr/bin/env python3
"""Extract fused embeddings for split manifest rows."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DIAG_ROOT = Path(__file__).resolve().parent.parent / "cross_receiver_diagnosis"
sys.path.insert(0, str(DIAG_ROOT))

from rfhstu.data import DOMAIN_FIELDS, complex_iq_to_channels, normalize_iq
from evaluate import build_model, load_checkpoint, prepare_model_input
from lib.extraction import extract_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--embedding-path", default="fused", choices=["main", "oob", "fused"])
    return p.parse_args()


class SplitWindowDataset(Dataset):
    def __init__(self, rows: list[dict], input_norm: str = "iq_rms", window_size: int = 8192):
        self.rows = rows
        self.input_norm = input_norm
        self.window_size = window_size
        self._memmaps: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, path: str) -> np.memmap:
        if path not in self._memmaps:
            self._memmaps[path] = np.memmap(path, dtype=np.complex64, mode="r")
        return self._memmaps[path]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        mm = self._open(row["file_path"])
        off = int(row["sample_offset"])
        iq = np.asarray(mm[off : off + self.window_size])
        channels = complex_iq_to_channels(iq)
        if self.input_norm == "iq_rms":
            channels = normalize_iq(channels)
        return {
            "iq": torch.from_numpy(channels.copy()),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "device_id": int(row["device_id"]),
            "window_index": int(row["window_index"]),
            "role": row["role"],
            "row_idx": idx,
        }


def build_eval_args(ckpt: dict, batch_size: int) -> argparse.Namespace:
    ckpt_args = ckpt.get("args", {})
    ns = argparse.Namespace(**{k: v for k, v in ckpt_args.items()})
    ns.model_type = ckpt.get("model_type", ckpt_args.get("model_type", "rf_hstu"))
    ns.cnn_input_type = ckpt_args.get("cnn_input_type", "iq")
    ns.dim = ckpt_args.get("dim", 64)
    ns.depth = ckpt_args.get("depth", 2)
    ns.dropout = ckpt_args.get("dropout", 0.1)
    ns.window_size = ckpt_args.get("window_size", 8192)
    ns.patch_size = ckpt_args.get("patch_size", 256)
    ns.sample_rate = ckpt_args.get("sample_rate", 1e6)
    ns.lora_bandwidth = ckpt_args.get("lora_bandwidth", 125e3)
    ns.spreading_factor = ckpt_args.get("spreading_factor", 7)
    ns.patch_embed_type = ckpt_args.get("patch_embed_type", "cnn_stem")
    ns.cnn_stem_dim = ckpt_args.get("cnn_stem_dim", 32)
    ns.cnn_stem_kernels = ckpt_args.get("cnn_stem_kernels", [7, 5, 3])
    ns.oob_fusion_type = ckpt_args.get("oob_fusion_type", "cross_attn_oob")
    ns.use_oob_cross_attention = ckpt_args.get("use_oob_cross_attention", True)
    ns.use_chirp_embedding = ckpt_args.get("use_chirp_embedding", True)
    ns.input_norm = ckpt_args.get("input_norm", "iq_rms")
    ns.fft_norm = ckpt_args.get("fft_norm", "log_zscore")
    ns.oob_norm = ckpt_args.get("oob_norm", "ratio")
    ns.no_oob = ckpt_args.get("no_oob", False)
    ns.cnn_hidden_dim = ckpt_args.get("cnn_hidden_dim", 128)
    ns.cnn_dropout = ckpt_args.get("cnn_dropout", 0.3)
    ns.oob_num_heads = ckpt_args.get("oob_num_heads", 4)
    ns.use_multiscale = ckpt_args.get("use_multiscale", False)
    ns.multiscale_ratios = ckpt_args.get("multiscale_ratios", [1, 2, 4])
    ns.multiscale_fusion_type = ckpt_args.get("multiscale_fusion_type", "concat")
    ns.use_cfo_feature = ckpt_args.get("use_cfo_feature", False)
    ns.batch_size = batch_size
    return ns


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

    emb_list, logit_list, labels, roles, devices, win_idx, row_idx = [], [], [], [], [], [], []

    for batch in tqdm(loader, desc="extract split"):
        iq = batch["iq"].to(device)
        out = extract_batch(model, iq, ckpt_args, eval_args.model_type)
        z = out[args.embedding_path]
        emb_list.append(z.cpu().numpy())
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
        embeddings=np.concatenate(emb_list, axis=0),
        logits=np.concatenate(logit_list, axis=0),
        labels=np.array(labels, dtype=np.int64),
        roles=np.array(roles, dtype=object),
        device_ids=np.array(devices, dtype=np.int64),
        window_indices=np.array(win_idx, dtype=np.int64),
        row_indices=np.array(row_idx, dtype=np.int64),
    )
    meta = {"checkpoint": args.checkpoint, "split_csv": args.split_csv, "embedding_path": args.embedding_path}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out_path} shape={np.concatenate(emb_list, axis=0).shape}")


if __name__ == "__main__":
    main()
