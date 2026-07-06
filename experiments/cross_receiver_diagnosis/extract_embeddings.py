#!/usr/bin/env python3
"""Extract window- and file-level embeddings for cross-receiver diagnosis."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfhstu.data import SigMFIQDataset, load_manifest
from evaluate import build_model, load_checkpoint, prepare_model_input
from lib.extraction import extract_batch, l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest_rx1_to_rx2.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-direction", default="rx1_to_rx2", help="Which direction model was trained for")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--samples-per-file", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--root", default=str(ROOT))
    return p.parse_args()


def build_eval_args(ckpt: dict, args: argparse.Namespace) -> argparse.Namespace:
    ckpt_args = ckpt.get("args", {})
    ns = argparse.Namespace()
    for k, v in ckpt_args.items():
        setattr(ns, k, v)
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
    ns.cfo_feature_type = ckpt_args.get("cfo_feature_type", "peak_offset")
    ns.cfo_feature_norm = ckpt_args.get("cfo_feature_norm", "zscore")
    ns.batch_size = args.batch_size
    return ns


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    eval_args = build_eval_args(ckpt, args)
    ckpt_args = ckpt.get("args", {})
    model_type = ckpt.get("model_type", ckpt_args.get("model_type", "rf_hstu"))

    model = build_model(eval_args, ckpt, device)
    model.eval()

    rows = load_manifest(args.manifest, root=args.root)
    dataset = SigMFIQDataset(
        rows,
        window_size=eval_args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=eval_args.input_norm,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    paths_store: dict[str, list[np.ndarray]] = {"main": [], "oob": [], "fused": []}
    labels, receivers, devices, file_paths = [], [], [], []
    logits_list = []

    for batch in tqdm(loader, desc=f"extract {args.model_name}"):
        iq = batch["iq"].to(device)
        out = extract_batch(model, iq, ckpt_args, model_type)
        for path in paths_store:
            paths_store[path].append(out[path].cpu().numpy())
        logits_list.append(out["logits"].cpu().numpy())
        labels.extend(batch["label"].tolist())
        receivers.extend(batch["domains"][:, 1].tolist())  # receiver field index 1
        devices.extend(batch["device"].tolist())
        file_paths.extend(batch["file_path"])

    embeddings = {k: np.concatenate(v, axis=0) for k, v in paths_store.items()}
    labels = np.array(labels, dtype=np.int64)
    receivers = np.array(receivers, dtype=np.int64)
    devices = np.array(devices, dtype=np.int64)
    logits = np.concatenate(logits_list, axis=0)

    # File-level mean embeddings
    file_groups: dict[str, list[int]] = defaultdict(list)
    for idx, fp in enumerate(file_paths):
        file_groups[fp].append(idx)

    file_emb = {k: [] for k in embeddings}
    file_labels, file_receivers, file_devices, file_names = [], [], [], []
    for fp, indices in sorted(file_groups.items()):
        idx = np.array(indices)
        for k in file_emb:
            file_emb[k].append(embeddings[k][idx].mean(axis=0))
        file_labels.append(int(labels[idx[0]]))
        file_receivers.append(int(receivers[idx[0]]))
        file_devices.append(int(devices[idx[0]]))
        file_names.append(fp)

    file_emb = {k: np.stack(v) for k, v in file_emb.items()}
    file_labels = np.array(file_labels)
    file_receivers = np.array(file_receivers)
    file_devices = np.array(file_devices)

    np.savez_compressed(
        out_dir / "window_embeddings.npz",
        **embeddings,
        labels=labels,
        receivers=receivers,
        devices=devices,
        logits=logits,
    )
    np.savez_compressed(
        out_dir / "file_embeddings.npz",
        **file_emb,
        labels=file_labels,
        receivers=file_receivers,
        devices=file_devices,
        file_names=np.array(file_names, dtype=object),
    )

    meta = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "train_direction": args.train_direction,
        "seed": args.seed,
        "num_windows": int(len(labels)),
        "num_files": int(len(file_labels)),
        "model_type": model_type,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
