#!/usr/bin/env python3
"""EM-aware consistency regularization training (EM-Aug CE / EM-CR)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rfhstu.em_perturbations import (  # noqa: E402
    apply_em_perturbation,
    sample_emaug_training_perturb_config,
    sample_emcr_training_perturb_config,
    sample_weak_cfo_perturb_config,
)
from rfhstu.train_utils import (  # noqa: E402
    add_common_args,
    accuracy,
    format_metrics,
    load_checkpoint,
    make_datasets,
    make_loader,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from evaluate import build_model  # noqa: E402
from finetune import classification_loss, prepare_model_input  # noqa: E402


DEBUG_MODES = ("clean_only", "em_aug_ce", "weak_cfo", "em_cr_stopgrad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EM-CR consistency training")
    add_common_args(p)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--mode",
        choices=list(DEBUG_MODES) + ["em_cr", "em_cr_emb"],
        default="em_cr_stopgrad",
    )
    p.add_argument("--lambda-kl", type=float, default=0.1)
    p.add_argument("--lambda-emb", type=float, default=0.0)
    p.add_argument("--kl-temperature", type=float, default=2.0)
    p.add_argument("--freeze-head-only", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    return p.parse_args()


def freeze_backbone_train_head(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def sample_perturb(mode: str, args: argparse.Namespace, rng: torch.Generator):
    if mode == "clean_only":
        return None
    if mode == "em_aug_ce":
        return sample_emaug_training_perturb_config(args.sample_rate, args.lora_bandwidth, rng)
    if mode == "weak_cfo":
        return sample_weak_cfo_perturb_config(args.sample_rate, args.lora_bandwidth, rng)
    return sample_emcr_training_perturb_config(args.sample_rate, args.lora_bandwidth, rng)


def kl_stopgrad_teacher(logits_clean: torch.Tensor, logits_aug: torch.Tensor, temp: float) -> torch.Tensor:
    p_teacher = F.softmax(logits_clean.detach() / temp, dim=-1)
    log_p_student = F.log_softmax(logits_aug / temp, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temp ** 2)


def run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
    rng: torch.Generator,
) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "acc": 0.0, "ce_clean": 0.0, "ce_aug": 0.0, "kl": 0.0}
    count = 0
    use_aug = args.mode != "clean_only"
    use_kl = args.mode in ("em_cr", "em_cr_emb", "em_cr_stopgrad")

    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq_clean = batch["iq"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            out_clean = model(prepare_model_input(iq_clean, args))
            logits_clean = out_clean["logits"]

            if train and use_aug:
                pert_cfg = sample_perturb(args.mode, args, rng)
                iq_aug = apply_em_perturbation(iq_clean.clone(), pert_cfg) if pert_cfg else iq_clean
            else:
                iq_aug = iq_clean

            out_aug = model(prepare_model_input(iq_aug, args))
            logits_aug = out_aug["logits"]

            loss_ce_clean = classification_loss(logits_clean, labels, args, None)
            loss = loss_ce_clean
            loss_ce_aug = logits_clean.new_zeros(())
            kl_val = logits_clean.new_zeros(())

            if use_aug:
                loss_ce_aug = classification_loss(logits_aug, labels, args, None)
                loss = loss + loss_ce_aug

            if use_kl and use_aug:
                if args.mode == "em_cr_stopgrad":
                    kl_val = kl_stopgrad_teacher(logits_clean, logits_aug, args.kl_temperature)
                else:
                    kl_val = F.kl_div(
                        F.log_softmax(logits_aug, dim=-1),
                        F.softmax(logits_clean, dim=-1),
                        reduction="batchmean",
                    )
                loss = loss + args.lambda_kl * kl_val

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                optimizer.step()

        bsz = labels.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits_clean.detach(), labels) * bsz
        total["ce_clean"] += loss_ce_clean.item() * bsz
        total["ce_aug"] += float(loss_ce_aug.item()) * bsz
        total["kl"] += float(kl_val.item()) * bsz
        count += bsz
    return {k: v / max(1, count) for k, v in total.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.root = str(ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.init_checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    # Keep CLI training hyperparameters; ckpt args are for model/data architecture only.
    cli_training_keys = {
        "epochs", "lr", "weight_decay", "batch_size", "max_files",
        "samples_per_file", "mode", "lambda_kl", "lambda_emb",
        "kl_temperature", "freeze_head_only", "grad_clip", "label_smoothing",
        "seed", "num_workers",
    }
    for key, val in ckpt_args.items():
        if key in cli_training_keys or key in ("manifest", "out_dir"):
            continue
        if hasattr(args, key):
            try:
                setattr(args, key, val)
            except Exception:
                pass

    num_classes = int(ckpt.get("num_classes", ckpt_args.get("num_classes", 24)))
    model = build_model(args, ckpt, device)
    model.train()
    if args.freeze_head_only:
        freeze_backbone_train_head(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)

    train_ds, val_ds = make_datasets(args)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)

    log_rows: list[dict] = []
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, args, train=True, rng=rng)
        val_m = run_epoch(model, val_loader, optimizer, device, args, train=False, rng=rng)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        log_rows.append(row)
        print(f"epoch={epoch} train_{format_metrics(train_m)} val_{format_metrics(val_m)}")

        extra = {
            "epoch": epoch,
            "val_acc": val_m["acc"],
            "num_classes": num_classes,
            "em_cr_mode": args.mode,
            "lambda_kl": args.lambda_kl,
            "freeze_head_only": args.freeze_head_only,
            "init_checkpoint": args.init_checkpoint,
        }
        save_checkpoint(out_dir / "last.pt", model, args, extra)
        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            save_checkpoint(out_dir / "best.pt", model, args, extra)

    with (out_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
