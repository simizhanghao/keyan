#!/usr/bin/env python3
"""Minimal B1 P1 (rtl_2) nominal pilot on external HDF5.

This is a pipeline/training pilot, not a paper metric. Checkpoint selection is
source-only; the held-out receiver is evaluated once after training.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rfhstu.b1_late_fusion import MultiViewLateFusionCNN


def load(paths: list[Path], per_file: int, fs: float, bw: float):
    xs, ys = [], []
    for p in paths:
        if p.suffix == ".npz":
            z = np.load(p); data = np.asarray(z["data"][:per_file], dtype=np.float32); labels = np.asarray(z["label"][:per_file]).reshape(-1).astype(np.int64) - 31
        else:
            import h5py
            with h5py.File(p, "r") as f:
                data = np.asarray(f["data"][:per_file], dtype=np.float32)
                labels = np.asarray(f["label"][:per_file]).reshape(-1).astype(np.int64) - 31
        t = data.shape[1] // 2
        iq = data[:, :t] + 1j * data[:, t:]
        iq = iq / (np.sqrt(np.mean(np.abs(iq) ** 2, axis=1, keepdims=True)) + 1e-8)
        z = np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)
        freq = np.fft.fftshift(np.fft.fftfreq(t, d=1.0 / fs))
        oob = np.abs(freq) > bw / 2
        views = np.stack([iq.real, iq.imag], 1), np.stack([z.real, z.imag], 1), \
            np.stack([np.abs(iq), np.angle(iq)], 1), np.abs(z[:, oob])[:, None, :]
        xs.append(views); ys.append(labels)
    out = tuple(torch.from_numpy(np.concatenate([x[i] for x in xs])) .float() for i in range(4))
    return out, torch.from_numpy(np.concatenate(ys)).long()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-file", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    paths = sorted(args.source_root.glob("*_train.h5")) or sorted(args.source_root.glob("*_train.npz"))
    train_paths = [x for x in paths if x.stem not in {"rtl_2_train", "rtl_2_train"}]
    heldout = [x for x in paths if x.stem == "rtl_2_train"]
    if len(train_paths) != 13 or len(heldout) != 1: raise RuntimeError("P1 fold mismatch")
    train_x, train_y = load(train_paths, args.per_file, 1e6, 125e3)
    test_x, test_y = load(heldout, args.per_file, 1e6, 125e3)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MultiViewLateFusionCNN(10).to(dev)
    params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    ds = TensorDataset(*train_x, train_y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    losses = []
    model.train()
    for _ in range(args.epochs):
        for *views, y in loader:
            batch = {k: v.to(dev) for k, v in zip(("iq", "fft", "amp_phase", "oob"), views)}
            logits = model(batch)["logits"]
            loss = F.cross_entropy(logits, y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        def acc(x, y):
            b = {k: v.to(dev) for k, v in zip(("iq", "fft", "amp_phase", "oob"), x)}
            pred = model(b)["logits"].argmax(-1).cpu()
            return float((pred == y).float().mean())
        source_acc, heldout_acc = acc(train_x, train_y), acc(test_x, test_y)
    payload = {"protocol": {"fold": "P1", "heldout_receiver": "rtl_2", "checkpoint_selection": "source_only", "seed": 0, "training": True, "blind_opened": False}, "files": len(paths), "train_packets": len(train_y), "heldout_packets": len(test_y), "parameters": params, "loss_first": losses[0], "loss_last": losses[-1], "source_acc": source_acc, "heldout_acc_pilot_only": heldout_acc, "note": "pilot only; not a paper result"}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
