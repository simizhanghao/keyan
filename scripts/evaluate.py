from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.prototype import build_prototypes, prototype_predict
from rfhstu.train_utils import accuracy, add_common_args, format_metrics, load_checkpoint, make_datasets, make_loader, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF-HSTU classifier or prototype head.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prototype", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace, ckpt: dict, device: torch.device) -> DeviceClassifier:
    ckpt_args = ckpt.get("args", {})
    dim = ckpt_args.get("dim", args.dim)
    depth = ckpt_args.get("depth", args.depth)
    dropout = ckpt_args.get("dropout", args.dropout)
    window_size = ckpt_args.get("window_size", args.window_size)
    patch_size = ckpt_args.get("patch_size", args.patch_size)
    sample_rate = ckpt_args.get("sample_rate", args.sample_rate)
    lora_bandwidth = ckpt_args.get("lora_bandwidth", args.lora_bandwidth)
    use_oob = not ckpt_args.get("no_oob", args.no_oob)
    num_classes = ckpt.get("num_classes")
    if num_classes is None:
        num_classes = ckpt_args.get("num_classes", 25)
    embedder = RFPatchEmbedder(
        window_size=window_size,
        patch_size=patch_size,
        sample_rate=sample_rate,
        lora_bandwidth=lora_bandwidth,
        use_oob=use_oob,
    )
    model = DeviceClassifier(
        embedder,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        dropout=dropout,
        domain_sizes=ckpt.get("domain_sizes"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_embeddings(model: DeviceClassifier, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    for batch in tqdm(loader, leave=False, desc="embed"):
        out = model(batch["iq"].to(device))
        embeddings.append(out["embedding"].cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def eval_classifier(model: DeviceClassifier, loader, device: torch.device) -> dict[str, float]:
    total_acc = 0.0
    count = 0
    for batch in tqdm(loader, leave=False, desc="eval"):
        labels = batch["label"].to(device)
        out = model(batch["iq"].to(device))
        bsz = labels.shape[0]
        total_acc += accuracy(out["logits"], labels) * bsz
        count += bsz
    return {"acc": total_acc / max(1, count)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = build_model(args, ckpt, device)
    train_ds, val_ds = make_datasets(args)
    train_loader = make_loader(train_ds, args, shuffle=False)
    val_loader = make_loader(val_ds, args, shuffle=False)

    if args.prototype:
        train_z, train_y = collect_embeddings(model, train_loader, device)
        val_z, val_y = collect_embeddings(model, val_loader, device)
        prototypes, proto_labels = build_prototypes(train_z, train_y)
        pred = prototype_predict(val_z, prototypes, proto_labels)
        metrics = {"prototype_acc": (pred == val_y).float().mean().item()}
    else:
        metrics = eval_classifier(model, val_loader, device)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()

