from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DOMAIN_FIELDS, SigMFIQDataset, load_manifest


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="data/raw/osu_lora/manifest_all.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--setup", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--samples-per-file", type=int, default=128)
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0)
    parser.add_argument("--lora-bandwidth", type=float, default=125_000.0)
    parser.add_argument("--no-oob", action="store_true")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(args: argparse.Namespace) -> tuple[SigMFIQDataset, SigMFIQDataset]:
    train_rows = load_manifest(args.manifest, root=args.root, split="train", setup=args.setup, max_files=args.max_files)
    val_rows = load_manifest(args.manifest, root=args.root, split="val", setup=args.setup, max_files=args.max_files)
    if not val_rows:
        val_rows = load_manifest(args.manifest, root=args.root, split=None, setup=args.setup, max_files=args.max_files)
    train_ds = SigMFIQDataset(
        train_rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=True,
        seed=args.seed,
    )
    val_ds = SigMFIQDataset(
        val_rows,
        window_size=args.window_size,
        samples_per_file=max(1, min(args.samples_per_file, 32)),
        random_windows=False,
        seed=args.seed,
    )
    return train_ds, val_ds


def make_loader(dataset: SigMFIQDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def save_checkpoint(path: str | Path, model: torch.nn.Module, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "domain_fields": DOMAIN_FIELDS,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())

