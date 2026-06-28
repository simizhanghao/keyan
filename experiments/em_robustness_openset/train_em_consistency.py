#!/usr/bin/env python3
"""EM-aware consistency regularization training (EM-Aug CE / EM-CR)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rfhstu.em_perturbations import apply_em_perturbation, sample_emcr_training_perturb_config  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EM-CR consistency training")
    add_common_args(p)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["em_aug_ce", "em_cr", "em_cr_emb"], default="em_cr")
    p.add_argument("--lambda-kl", type=float, default=0.5)
    p.add_argument("--lambda-emb", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    return p.parse_args()


def kl_consistency(p_clean: torch.Tensor, p_aug: torch.Tensor) -> torch.Tensor:
    log_p_aug = F.log_softmax(p_aug, dim=-1)
    p_c = F.softmax(p_clean, dim=-1)
    return F.kl_div(log_p_aug, p_c, reduction="batchmean")


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
    total = {"loss": 0.0, "acc": 0.0, "ce_clean": 0.0, "ce_aug": 0.0, "kl": 0.0, "emb": 0.0}
    count = 0
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq_clean = batch["iq"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            out_clean = model(prepare_model_input(iq_clean, args))
            logits_clean = out_clean["logits"]
            emb_clean = out_clean.get("embedding")

            if train and args.mode != "em_aug_ce":
                pert_cfg = sample_emcr_training_perturb_config(
                    sample_rate=args.sample_rate,
                    lora_bw=args.lora_bandwidth,
                    rng=rng,
                )
                iq_aug = apply_em_perturbation(iq_clean.clone(), pert_cfg)
            elif train:
                pert_cfg = sample_emcr_training_perturb_config(
                    sample_rate=args.sample_rate,
                    lora_bw=args.lora_bandwidth,
                    rng=rng,
                )
                iq_aug = apply_em_perturbation(iq_clean.clone(), pert_cfg)
            else:
                iq_aug = iq_clean

            out_aug = model(prepare_model_input(iq_aug, args))
            logits_aug = out_aug["logits"]
            emb_aug = out_aug.get("embedding")

            loss_ce_clean = classification_loss(logits_clean, labels, args, None)
            loss_ce_aug = classification_loss(logits_aug, labels, args, None)
            loss = loss_ce_clean + loss_ce_aug

            kl_val = logits_clean.new_zeros(())
            emb_val = logits_clean.new_zeros(())
            if args.mode in ("em_cr", "em_cr_emb"):
                kl_val = kl_consistency(logits_clean, logits_aug)
                loss = loss + args.lambda_kl * kl_val
            if args.mode == "em_cr_emb" and emb_clean is not None and emb_aug is not None:
                emb_val = F.mse_loss(emb_clean, emb_aug)
                loss = loss + args.lambda_emb * emb_val

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = labels.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits_clean.detach(), labels) * bsz
        total["ce_clean"] += loss_ce_clean.item() * bsz
        total["ce_aug"] += loss_ce_aug.item() * bsz
        total["kl"] += float(kl_val.item()) * bsz
        total["emb"] += float(emb_val.item()) * bsz
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
    for key, val in ckpt_args.items():
        if hasattr(args, key) and key not in ("manifest", "out_dir", "epochs", "max_files"):
            try:
                setattr(args, key, val)
            except Exception:
                pass

    num_classes = int(ckpt.get("num_classes", ckpt_args.get("num_classes", 24)))
    model = build_model(args, ckpt, device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)

    train_ds, val_ds = make_datasets(args)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)

    log_path = out_dir / "train_log.csv"
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
            "lambda_emb": args.lambda_emb,
            "init_checkpoint": args.init_checkpoint,
        }
        save_checkpoint(out_dir / "last.pt", model, args, extra)
        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            save_checkpoint(out_dir / "best.pt", model, args, extra)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
