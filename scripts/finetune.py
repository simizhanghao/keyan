from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.data import DOMAIN_FIELDS, infer_domain_sizes
from rfhstu.losses import supervised_contrastive_loss
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.train_utils import accuracy, add_common_args, format_metrics, load_checkpoint, make_datasets, make_loader, resolve_device, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-HSTU for device classification.")
    add_common_args(parser)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--out-dir", default="runs/finetune")
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--use-adversarial", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def maybe_load_pretrained(model: DeviceClassifier, path: str | None, device: torch.device) -> None:
    if not path:
        return
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"]
    encoder_state = {key.replace("encoder.", "", 1): value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"loaded_pretrained={path} missing={len(missing)} unexpected={len(unexpected)}")


def run_epoch(model: DeviceClassifier, loader, optimizer, device: torch.device, args: argparse.Namespace, train: bool) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "acc": 0.0}
    count = 0
    field_to_col = {field: idx for idx, field in enumerate(DOMAIN_FIELDS)}
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq = batch["iq"].to(device)
        labels = batch["label"].to(device)
        domains = batch["domains"].to(device)
        with torch.set_grad_enabled(train):
            out = model(iq, adv_lambda=args.adv_lambda)
            logits = out["logits"]
            z = out["embedding"]
            loss = F.cross_entropy(logits, labels)
            if args.use_contrastive:
                loss = loss + args.contrastive_weight * supervised_contrastive_loss(z, labels, args.temperature)
            if args.use_adversarial and model.domain_head is not None:
                domain_loss = model.domain_head.loss(out["domain_logits"], domains, field_to_col)
                loss = loss + args.adversarial_weight * domain_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bsz = iq.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits.detach(), labels) * bsz
        count += bsz
    return {key: value / max(1, count) for key, value in total.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_ds, val_ds = make_datasets(args)
    embedder = RFPatchEmbedder(
        window_size=args.window_size,
        patch_size=args.patch_size,
        sample_rate=args.sample_rate,
        lora_bandwidth=args.lora_bandwidth,
        use_oob=not args.no_oob,
    )
    domain_sizes = infer_domain_sizes([*train_ds.rows, *val_ds.rows]) if args.use_adversarial else None
    num_classes = max(row.label for row in [*train_ds.rows, *val_ds.rows]) + 1
    model = DeviceClassifier(
        embedder,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        dropout=args.dropout,
        domain_sizes=domain_sizes,
    ).to(device)
    maybe_load_pretrained(model, args.pretrained, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        print(f"epoch={epoch} train_{format_metrics(train_metrics)} val_{format_metrics(val_metrics)}")
        extra = {
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "num_classes": num_classes,
            "domain_sizes": domain_sizes,
        }
        save_checkpoint(Path(args.out_dir) / "last.pt", model, args, extra)
        if val_metrics["acc"] > best:
            best = val_metrics["acc"]
            save_checkpoint(Path(args.out_dir) / "best.pt", model, args, extra)


if __name__ == "__main__":
    main()
