"""Shared utilities for EM robustness and open-set experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rfhstu.data import SigMFIQDataset, load_manifest  # noqa: E402
from rfhstu.em_perturbations import EmPerturbConfig, apply_em_perturbation  # noqa: E402
from evaluate import build_model, load_checkpoint, prepare_model_input  # noqa: E402

DEFAULT_MANIFEST = "data/paper/cross_day_day1to5_source_only.csv"
DEFAULT_CHECKPOINT = (
    "outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
)
ALL_DEVICE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def populate_args_from_ckpt(args: argparse.Namespace, ckpt: dict) -> dict[str, Any]:
    ckpt_args = ckpt.get("args", {})
    args.model_type = ckpt_args.get("model_type", getattr(args, "model_type", "rf_hstu"))
    args.cnn_input_type = ckpt_args.get("cnn_input_type", "iq")
    args.dim = ckpt_args.get("dim", 64)
    args.depth = ckpt_args.get("depth", 2)
    args.dropout = ckpt_args.get("dropout", 0.1)
    args.window_size = ckpt_args.get("window_size", getattr(args, "window_size", 8192))
    args.patch_size = ckpt_args.get("patch_size", 256)
    args.spreading_factor = ckpt_args.get("spreading_factor", 7)
    args.patch_embed_type = ckpt_args.get("patch_embed_type", "cnn_stem")
    args.cnn_stem_dim = ckpt_args.get("cnn_stem_dim", 32)
    args.cnn_stem_kernels = ckpt_args.get("cnn_stem_kernels", [7, 5, 3])
    args.oob_fusion_type = ckpt_args.get("oob_fusion_type", "cross_attn_oob")
    args.use_oob_cross_attention = ckpt_args.get("use_oob_cross_attention", True)
    args.use_chirp_embedding = ckpt_args.get("use_chirp_embedding", True)
    args.input_norm = ckpt_args.get("input_norm", "iq_rms")
    args.fft_norm = ckpt_args.get("fft_norm", "log_zscore")
    args.oob_norm = ckpt_args.get("oob_norm", "zscore")
    args.no_oob = ckpt_args.get("no_oob", False)
    args.cnn_hidden_dim = ckpt_args.get("cnn_hidden_dim", 128)
    args.cnn_dropout = ckpt_args.get("cnn_dropout", 0.3)
    args.oob_num_heads = ckpt_args.get("oob_num_heads", 4)
    args.use_multiscale = ckpt_args.get("use_multiscale", False)
    args.multiscale_ratios = ckpt_args.get("multiscale_ratios", [1, 2, 4])
    args.multiscale_fusion_type = ckpt_args.get("multiscale_fusion_type", "concat")
    args.use_cfo_feature = ckpt_args.get("use_cfo_feature", False)
    args.cfo_feature_type = ckpt_args.get("cfo_feature_type", "peak_offset")
    args.cfo_feature_norm = ckpt_args.get("cfo_feature_norm", "zscore")
    args.sample_rate = ckpt_args.get("sample_rate", 1_000_000.0)
    args.lora_bandwidth = ckpt_args.get("lora_bandwidth", 125_000.0)
    args.spreading_factor = ckpt_args.get("spreading_factor", 7)
    return ckpt_args


def make_eval_loader(args: argparse.Namespace, split: str) -> DataLoader:
    rows = load_manifest(args.manifest, root=args.root, split=split)
    ds = SigMFIQDataset(
        rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=args.input_norm,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


@torch.no_grad()
def collect_file_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_args: dict,
    perturb_cfg: EmPerturbConfig | None = None,
) -> list[dict]:
    """Aggregate window logits/embeddings to file-level features."""
    file_windows: dict[str, list[dict]] = {}
    for batch in tqdm(loader, leave=False):
        iq = batch["iq"].to(device)
        if perturb_cfg is not None:
            iq = apply_em_perturbation(iq, perturb_cfg)
        model_input = prepare_model_input(iq, args, ckpt_args)
        out = model(model_input)
        logits = out["logits"]
        emb = F.normalize(out["embedding"], dim=-1)
        probs = F.softmax(logits, dim=-1)
        for idx in range(logits.shape[0]):
            fp = batch["file_path"][idx]
            file_windows.setdefault(fp, []).append(
                {
                    "label": int(batch["label"][idx].item()),
                    "device": int(batch["device"][idx].item()),
                    "logits": logits[idx].cpu(),
                    "embedding": emb[idx].cpu(),
                    "msp": float(probs[idx].max().item()),
                    "energy": float(-torch.logsumexp(logits[idx], dim=0).item()),
                    "split": batch["split"][idx],
                }
            )

    rows: list[dict] = []
    for fp, wins in file_windows.items():
        logits = torch.stack([w["logits"] for w in wins]).mean(dim=0)
        embedding = F.normalize(torch.stack([w["embedding"] for w in wins]).mean(dim=0), dim=0)
        probs = F.softmax(logits, dim=-1)
        label = wins[0]["label"]
        pred = int(probs.argmax().item())
        rows.append(
            {
                "file_path": fp,
                "label": label,
                "pred": pred,
                "correct": int(pred == label),
                "msp": float(probs.max().item()),
                "energy": float(-torch.logsumexp(logits, dim=0).item()),
                "embedding": embedding.numpy(),
                "split": wins[0]["split"],
            }
        )
    return rows


def pick_unknown_devices(seed: int, n_unknown: int = 4) -> list[int]:
    rng = np.random.default_rng(seed)
    devices = list(ALL_DEVICE_IDS)
    rng.shuffle(devices)
    return sorted(devices[:n_unknown])


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC with y_true=1 known, 0 unknown; higher score => more likely known."""
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0
    auc = 0.0
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            auc += tp
    return float(auc / (pos * neg))


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Equal error rate (fraction)."""
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        return float("nan")
    best = 1.0
    for t in thresholds:
        pred_known = scores >= t
        far = np.mean(pred_known[y_true == 0]) if np.any(y_true == 0) else 0.0
        frr = np.mean(~pred_known[y_true == 1]) if np.any(y_true == 1) else 0.0
        best = min(best, abs(far - frr))
    return float(best)


def compute_fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    thresholds = np.unique(scores)[::-1]
    for t in thresholds:
        pred_known = scores >= t
        tpr = np.mean(pred_known[y_true == 1]) if np.any(y_true == 1) else 0.0
        if tpr >= tpr_target:
            return float(np.mean(pred_known[y_true == 0]) if np.any(y_true == 0) else 0.0)
    return 1.0


def select_threshold_on_val(val_known_scores: np.ndarray, val_unknown_scores: np.ndarray) -> float:
    """Pick threshold on validation known vs unknown score distributions."""
    y = np.concatenate([np.ones(len(val_known_scores)), np.zeros(len(val_unknown_scores))])
    s = np.concatenate([val_known_scores, val_unknown_scores])
    return float(np.median(s))


def robustness_summary(file_accs: list[float], clean_acc: float) -> dict[str, float]:
    accs = np.array(file_accs, dtype=np.float64)
    x = np.linspace(0, 1, len(accs))
    auc = float(np.trapz(accs, x))
    return {
        "clean_file_acc": clean_acc,
        "avg_robust_acc": float(accs.mean()),
        "robustness_auc": auc,
        "max_drop": float(clean_acc - accs.min()),
        "mean_drop": float(clean_acc - accs.mean()),
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
