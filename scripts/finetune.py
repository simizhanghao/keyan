from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.cnn_baseline import OSUCNNBaseline, build_cnn_input
from rfhstu.data import DOMAIN_FIELDS, infer_domain_sizes
from rfhstu.losses import (
    CenterLoss,
    SupConLoss,
    coral_loss,
    focal_loss,
    information_maximization_loss,
    macro_f1_from_logits,
    supervised_contrastive_loss,
)
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.train_utils import (
    accuracy,
    add_common_args,
    apply_receiver_style,
    format_metrics,
    paired_second_view,
    forward_with_batch,
    load_checkpoint,
    make_datasets,
    make_loader,
    make_target_unlabeled_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-HSTU for device classification.")
    add_common_args(parser)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Encoder-only warm start (legacy). Forbidden for 2B-1 F0/F1; drops classifier.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Full-model warm start from a DeviceClassifier best.pt (embedder+encoder+classifier). "
        "Required for Phase 2B-1 F0 identity-first. Mutually exclusive with --pretrained.",
    )
    parser.add_argument("--out-dir", default="runs/finetune")
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--use-adversarial", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-supcon", action="store_true")
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--use-supcon-proj", action="store_true")
    parser.add_argument("--supcon-proj-dim", type=int, default=64)
    parser.add_argument("--augment-rf", action="store_true")
    parser.add_argument("--aug-phase-std", type=float, default=0.0)
    parser.add_argument("--aug-amp-std", type=float, default=0.0)
    parser.add_argument("--aug-noise-std", type=float, default=0.0)
    parser.add_argument("--aug-time-shift", type=int, default=0)
    parser.add_argument("--use-hard-margin", action="store_true")
    parser.add_argument("--hard-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--use-center-loss", action="store_true")
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    return parser.parse_args()


class SupConProjector(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def maybe_load_pretrained(model: torch.nn.Module, path: str | None, device: torch.device) -> None:
    if not path:
        return
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"]
    encoder_state = {key.replace("encoder.", "", 1): value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"loaded_pretrained={path} missing={len(missing)} unexpected={len(unexpected)}")
    print("WARNING: --pretrained is encoder-only; classifier is NOT restored. Use --init-checkpoint for F0.")


def maybe_load_init_checkpoint(model: torch.nn.Module, path: str | None, device: torch.device) -> None:
    """Full DeviceClassifier state (strict). Used by Phase 2B-1 F0 identity-first."""
    if not path:
        return
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    n_cls = sum(1 for k in state if k.startswith("classifier."))
    if n_cls < 1:
        raise RuntimeError(f"--init-checkpoint has no classifier.* keys: {path}")
    print(
        f"loaded_init_checkpoint={path} "
        f"classifier_tensors={n_cls} missing={len(missing)} unexpected={len(unexpected)}"
    )


def compute_class_weights(train_ds, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for row in train_ds.rows:
        counts[row.label] += 1.0
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights.to(device)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if args.loss_type == "focal":
        return focal_loss(
            logits,
            labels,
            gamma=args.focal_gamma,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )


def prepare_model_input(iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.model_type == "osu_cnn":
        return build_cnn_input(
            iq,
            input_type=args.cnn_input_type,
            sample_rate=args.sample_rate,
            lora_bandwidth=args.lora_bandwidth,
        )
    return iq


def augment_iq(iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.augment_rf:
        return iq
    x = iq
    bsz = x.shape[0]
    if args.aug_phase_std > 0:
        phi = torch.randn(bsz, 1, device=x.device, dtype=x.dtype) * args.aug_phase_std
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        i = x[:, 0] * cos_phi - x[:, 1] * sin_phi
        q = x[:, 0] * sin_phi + x[:, 1] * cos_phi
        x = torch.stack([i, q], dim=1)
    if args.aug_amp_std > 0:
        scale = 1.0 + torch.randn(bsz, 1, 1, device=x.device, dtype=x.dtype) * args.aug_amp_std
        x = x * scale.clamp_min(0.05)
    if args.aug_noise_std > 0:
        rms = torch.sqrt(x.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8))
        x = x + torch.randn_like(x) * (args.aug_noise_std * rms)
    if args.aug_time_shift > 0:
        shifts = torch.randint(-args.aug_time_shift, args.aug_time_shift + 1, (bsz,), device=x.device)
        x = torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(x, shifts)], dim=0)
    return x


def augment_receiver_style(iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Train-only receiver-style augmentation. Off unless --augment-receiver-style."""
    if not getattr(args, "augment_receiver_style", False):
        return iq
    return apply_receiver_style(iq, args, lock_inband=False)


def hard_negative_margin_loss(features: torch.Tensor, labels: torch.Tensor, classifier: nn.Module, margin: float) -> torch.Tensor:
    if not hasattr(classifier, "weight"):
        raise RuntimeError("--use-hard-margin requires a linear classifier with a weight matrix.")
    weight = classifier.weight
    f = F.normalize(features, dim=-1)
    w = F.normalize(weight, dim=-1)
    sim = f @ w.T
    pos = sim.gather(1, labels.view(-1, 1)).squeeze(1)
    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask.scatter_(1, labels.view(-1, 1), False)
    neg = sim.masked_fill(~neg_mask, -torch.inf).max(dim=1).values
    return F.relu(margin + neg - pos).mean()


def run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
    supcon_projector: torch.nn.Module | None = None,
    center_loss: CenterLoss | None = None,
    class_weights: torch.Tensor | None = None,
    num_classes: int = 0,
) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}
    count = 0
    field_to_col = {field: idx for idx, field in enumerate(DOMAIN_FIELDS)}
    supcon = SupConLoss(args.supcon_temperature)
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq = batch["iq"].to(device)
        if train:
            iq = augment_iq(iq, args)
            iq = augment_receiver_style(iq, args)
        model_input = prepare_model_input(iq, args)
        labels = batch["label"].to(device)
        domains = batch["domains"].to(device)
        paired = bool(train and getattr(args, "paired_view", "off") != "off")
        with torch.set_grad_enabled(train):
            out = forward_with_batch(
                model, model_input, batch, adv_lambda=args.adv_lambda, return_features=args.use_supcon
            )
            logits = out["logits"]
            z = out.get("features", out["embedding"])
            loss = classification_loss(logits, labels, args, class_weights)
            if paired:
                iq_b = paired_second_view(iq, args)
                out_b = forward_with_batch(
                    model,
                    prepare_model_input(iq_b, args),
                    batch,
                    adv_lambda=args.adv_lambda,
                    return_features=False,
                )
                loss = 0.5 * loss + 0.5 * classification_loss(out_b["logits"], labels, args, class_weights)
            elif args.use_hard_margin:
                if args.use_supcon:
                    raise RuntimeError("--use-hard-margin and --use-supcon should not be enabled together in this first version.")
                loss = loss + args.hard_margin_weight * hard_negative_margin_loss(z, labels, model.classifier, args.hard_margin)
            if not paired:
                if args.use_center_loss:
                    if center_loss is None:
                        raise RuntimeError("--use-center-loss requires a CenterLoss module.")
                    loss = loss + args.center_loss_weight * center_loss(z, labels)
                if args.use_supcon:
                    supcon_z = supcon_projector(z) if supcon_projector is not None else z
                    loss = loss + args.supcon_weight * supcon(supcon_z, labels)
                if args.use_contrastive:
                    loss = loss + args.contrastive_weight * supervised_contrastive_loss(z, labels, args.temperature)
                if args.use_adversarial and model.domain_head is not None:
                    domain_loss = model.domain_head.loss(out["domain_logits"], domains, field_to_col)
                    loss = loss + args.adversarial_weight * domain_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bsz = model_input.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits.detach(), labels) * bsz
        if not train and num_classes > 0:
            total["macro_f1"] += macro_f1_from_logits(logits.detach(), labels, num_classes) * bsz
        count += bsz
    metrics = {key: value / max(1, count) for key, value in total.items()}
    if train:
        metrics.pop("macro_f1", None)
    return metrics


def run_epoch_aligned(
    model: torch.nn.Module,
    source_loader,
    target_loader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
    supcon_projector: torch.nn.Module | None = None,
    center_loss: CenterLoss | None = None,
    class_weights: torch.Tensor | None = None,
    num_classes: int = 0,
) -> dict[str, float]:
    """Source labeled CE + optional target-unlabeled CORAL / IM (no target labels)."""
    model.train(train)
    total = {"loss": 0.0, "acc": 0.0, "coral": 0.0, "im": 0.0, "macro_f1": 0.0}
    count = 0
    field_to_col = {field: idx for idx, field in enumerate(DOMAIN_FIELDS)}
    supcon = SupConLoss(args.supcon_temperature)
    target_iter = itertools.cycle(target_loader)
    align = args.domain_align_loss
    for batch in tqdm(source_loader, leave=False, desc="train_align" if train else "val_align"):
        iq_s = batch["iq"].to(device)
        if train:
            iq_s = augment_iq(iq_s, args)
            iq_s = augment_receiver_style(iq_s, args)
        labels = batch["label"].to(device)
        domains = batch["domains"].to(device)
        with torch.set_grad_enabled(train):
            out_s = forward_with_batch(
                model,
                prepare_model_input(iq_s, args),
                batch,
                adv_lambda=args.adv_lambda,
                return_features=args.use_supcon,
            )
            logits = out_s["logits"]
            z_s = out_s.get("features", out_s["embedding"])
            loss = classification_loss(logits, labels, args, class_weights)
            coral_val = logits.new_zeros(())
            im_val = logits.new_zeros(())
            if train and align in {"coral", "coral_im"}:
                z_t_list = []
                logits_t_list = []
                for _ in range(max(1, args.target_loader_ratio)):
                    t_batch = next(target_iter)
                    iq_t = t_batch["iq"].to(device)
                    out_t = forward_with_batch(
                        model,
                        prepare_model_input(iq_t, args),
                        t_batch,
                        adv_lambda=args.adv_lambda,
                        return_features=args.use_supcon,
                    )
                    z_t_list.append(out_t.get("features", out_t["embedding"]))
                    logits_t_list.append(out_t["logits"])
                z_t = torch.cat(z_t_list, dim=0)
                coral_val = coral_loss(z_s, z_t)
                loss = loss + args.domain_align_weight * coral_val
                if align == "coral_im":
                    logits_t = torch.cat(logits_t_list, dim=0)
                    im_val = information_maximization_loss(logits_t)
                    loss = loss + args.im_weight * im_val
            if args.use_hard_margin:
                if args.use_supcon:
                    raise RuntimeError("--use-hard-margin and --use-supcon should not be enabled together in this first version.")
                z = out_s.get("features", out_s["embedding"])
                loss = loss + args.hard_margin_weight * hard_negative_margin_loss(z, labels, model.classifier, args.hard_margin)
            if args.use_center_loss:
                if center_loss is None:
                    raise RuntimeError("--use-center-loss requires a CenterLoss module.")
                z = out_s.get("features", out_s["embedding"])
                loss = loss + args.center_loss_weight * center_loss(z, labels)
            if args.use_supcon:
                z = out_s.get("features", out_s["embedding"])
                supcon_z = supcon_projector(z) if supcon_projector is not None else z
                loss = loss + args.supcon_weight * supcon(supcon_z, labels)
            if args.use_contrastive:
                z = out_s.get("features", out_s["embedding"])
                loss = loss + args.contrastive_weight * supervised_contrastive_loss(z, labels, args.temperature)
            if args.use_adversarial and model.domain_head is not None:
                domain_loss = model.domain_head.loss(out_s["domain_logits"], domains, field_to_col)
                loss = loss + args.adversarial_weight * domain_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bsz = iq_s.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits.detach(), labels) * bsz
        total["coral"] += float(coral_val.item()) * bsz
        total["im"] += float(im_val.item()) * bsz
        if not train and num_classes > 0:
            total["macro_f1"] += macro_f1_from_logits(logits.detach(), labels, num_classes) * bsz
        count += bsz
    metrics = {key: value / max(1, count) for key, value in total.items()}
    if train:
        metrics.pop("macro_f1", None)
    return metrics


def checkpoint_score(val_metrics: dict[str, float], args: argparse.Namespace) -> float:
    if args.checkpoint_metric == "macro_f1":
        return val_metrics.get("macro_f1", -1.0)
    return val_metrics.get("acc", -1.0)


def update_bn_from_loader(model: torch.nn.Module, loader, device: torch.device, args: argparse.Namespace) -> None:
    """Refresh BatchNorm running stats after SWA (dict-batch compatible)."""
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            forward_with_batch(model, prepare_model_input(iq, args), batch)
    model.eval()


def main() -> None:
    args = parse_args()
    enabled_metric_losses = sum(bool(flag) for flag in [args.use_hard_margin, args.use_supcon, args.use_center_loss])
    if enabled_metric_losses > 1:
        raise ValueError("Enable only one of --use-hard-margin, --use-supcon, or --use-center-loss in this first version.")
    if args.pretrained and args.init_checkpoint:
        raise ValueError("Use only one of --pretrained (encoder-only) or --init-checkpoint (full model).")
    if args.oob_identity_shuffle:
        if args.model_type == "osu_cnn" or args.no_oob or args.oob_fusion_type == "no_oob":
            raise ValueError("OOB identity shuffle requires a Full model with an OOB branch.")
    paired_mode = getattr(args, "paired_view", "off")
    if paired_mode != "off":
        extras = [
            args.augment_receiver_style,
            args.augment_rf,
            args.use_contrastive,
            args.use_adversarial,
            args.use_supcon,
            args.use_hard_margin,
            args.use_center_loss,
            args.use_target_unlabeled,
        ]
        if any(extras):
            raise ValueError("--paired-view is CE-only and cannot combine with receiver-style/RF aug or extra losses.")
        print(f"paired_view={paired_mode} (train-only; val stays clean)")
    if args.init_checkpoint:
        print(f"init_checkpoint={args.init_checkpoint} (full state; fresh optimizer)")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_ds, val_ds = make_datasets(args)
    num_classes = max(row.label for row in [*train_ds.rows, *val_ds.rows]) + 1
    class_weights = compute_class_weights(train_ds, num_classes, device) if args.class_balanced_ce else None
    if args.model_type == "osu_cnn":
        domain_sizes = None
        model = OSUCNNBaseline(
            num_classes=num_classes,
            in_channels=2,
            hidden_dim=args.cnn_hidden_dim,
            dropout=args.cnn_dropout,
        ).to(device)
    else:
        embedder = RFPatchEmbedder(
            window_size=args.window_size,
            patch_size=args.patch_size,
            sample_rate=args.sample_rate,
            lora_bandwidth=args.lora_bandwidth,
            spreading_factor=args.spreading_factor,
            use_oob=not args.no_oob and args.oob_fusion_type != "no_oob",
            oob_fusion_type=args.oob_fusion_type,
            use_oob_cross_attention=args.use_oob_cross_attention,
            patch_embed_type=args.patch_embed_type,
            dim=args.dim,
            cnn_stem_dim=args.cnn_stem_dim,
            cnn_stem_kernels=args.cnn_stem_kernels,
            fft_norm=args.fft_norm,
            oob_norm=args.oob_norm,
            fft_source=getattr(args, "fft_source", "full"),
        )
        domain_sizes = infer_domain_sizes([*train_ds.rows, *val_ds.rows]) if args.use_adversarial else None
        model = DeviceClassifier(
            embedder,
            num_classes=num_classes,
            dim=args.dim,
            depth=args.depth,
            dropout=args.dropout,
            domain_sizes=domain_sizes,
            use_chirp_embedding=args.use_chirp_embedding,
            oob_num_heads=args.oob_num_heads,
            use_multiscale=args.use_multiscale,
            multiscale_ratios=args.multiscale_ratios,
            multiscale_fusion_type=args.multiscale_fusion_type,
            use_cfo_feature=args.use_cfo_feature,
            cfo_feature_type=args.cfo_feature_type,
            cfo_feature_norm=args.cfo_feature_norm,
            oob_dropout=args.oob_dropout,
            mixstyle=args.mixstyle,
            mixstyle_alpha=args.mixstyle_alpha,
        ).to(device)
    maybe_load_pretrained(model, args.pretrained, device)
    maybe_load_init_checkpoint(model, args.init_checkpoint, device)
    supcon_projector = None
    if args.use_supcon and args.use_supcon_proj:
        supcon_projector = SupConProjector(args.dim if args.model_type == "rf_hstu" else args.cnn_hidden_dim, args.supcon_proj_dim).to(device)
    center_loss = None
    if args.use_center_loss:
        if args.model_type == "osu_cnn":
            feat_dim = args.cnn_hidden_dim
        else:
            feat_dim = getattr(model, "embedding_dim", args.dim)
        center_loss = CenterLoss(num_classes, feat_dim).to(device)
    optim_params = list(model.parameters())
    if supcon_projector is not None:
        optim_params.extend(supcon_projector.parameters())
    if center_loss is not None:
        optim_params.extend(center_loss.parameters())
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    swa_model = AveragedModel(model) if args.use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=max(args.lr * 0.1, 1e-5)) if args.use_swa else None
    swa_start = max(1, int(math.ceil(args.epochs * 0.8))) if args.use_swa else args.epochs + 1
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    use_alignment = (
        args.use_target_unlabeled
        and args.domain_align_loss != "none"
        and args.model_type == "rf_hstu"
    )
    target_loader = None
    if use_alignment:
        target_ds = make_target_unlabeled_dataset(args)
        target_loader = make_loader(target_ds, args, shuffle=True)
        print(
            f"target_unlabeled rows={len(target_ds.rows)} "
            f"align={args.domain_align_loss} "
            f"coral_w={args.domain_align_weight} im_w={args.im_weight}"
        )
    elif args.use_target_unlabeled and args.domain_align_loss != "none":
        print("warning: domain alignment requires --model-type rf_hstu; falling back to source-only training.")

    best = -1.0
    epoch_fn = run_epoch_aligned if use_alignment and target_loader is not None else run_epoch
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if target_loader is not None and hasattr(target_loader.dataset, "set_epoch"):
            target_loader.dataset.set_epoch(epoch)
        if epoch_fn is run_epoch_aligned:
            train_metrics = epoch_fn(
                model, train_loader, target_loader, optimizer, device, args,
                train=True, supcon_projector=supcon_projector, center_loss=center_loss,
                class_weights=class_weights, num_classes=num_classes,
            )
            val_metrics = run_epoch(
                model, val_loader, optimizer, device, args, train=False,
                supcon_projector=supcon_projector, center_loss=center_loss,
                class_weights=class_weights, num_classes=num_classes,
            )
        else:
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, args, train=True,
                supcon_projector=supcon_projector, center_loss=center_loss,
                class_weights=class_weights, num_classes=num_classes,
            )
            val_metrics = run_epoch(
                model, val_loader, optimizer, device, args, train=False,
                supcon_projector=supcon_projector, center_loss=center_loss,
                class_weights=class_weights, num_classes=num_classes,
            )
        if args.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        print(f"epoch={epoch} train_{format_metrics(train_metrics)} val_{format_metrics(val_metrics)}")
        extra = {
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics.get("macro_f1", 0.0),
            "checkpoint_metric": args.checkpoint_metric,
            "num_classes": num_classes,
            "domain_sizes": domain_sizes,
            "model_type": args.model_type,
            "cnn_input_type": args.cnn_input_type,
        }
        if supcon_projector is not None:
            extra["supcon_projector"] = supcon_projector.state_dict()
        if center_loss is not None:
            extra["center_loss"] = center_loss.state_dict()
        save_checkpoint(Path(args.out_dir) / "last.pt", model, args, extra)
        score = checkpoint_score(val_metrics, args)
        if score > best:
            best = score
            save_checkpoint(Path(args.out_dir) / "best.pt", model, args, extra)

    if swa_model is not None:
        print(f"finalizing SWA from epoch>={swa_start}")
        update_bn_from_loader(swa_model.module, train_loader, device, args)
        swa_extra = {
            "epoch": args.epochs,
            "val_acc": best if args.checkpoint_metric == "acc" else "",
            "val_macro_f1": best if args.checkpoint_metric == "macro_f1" else "",
            "checkpoint_metric": args.checkpoint_metric,
            "num_classes": num_classes,
            "domain_sizes": domain_sizes,
            "model_type": args.model_type,
            "cnn_input_type": args.cnn_input_type,
            "swa": True,
        }
        save_checkpoint(Path(args.out_dir) / "swa.pt", swa_model.module, args, swa_extra)


if __name__ == "__main__":
    main()
