#!/usr/bin/env python3
"""Evaluate a trained checkpoint under EM perturbation sweeps.

Usage:
  python experiments/em_robustness_openset/eval_robustness_curves.py \\
    --manifest data/paper/cross_day_day1to5_source_only.csv \\
    --checkpoint outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt \\
    --perturb-type awgn_snr_db \\
    --out-csv experiments/em_robustness_openset/results/awgn_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rfhstu.data import SigMFIQDataset, load_manifest, DOMAIN_FIELDS  # noqa: E402
from rfhstu.em_perturbations import ROBUSTNESS_SWEEPS, apply_em_perturbation, make_sweep_config  # noqa: E402
from evaluate import build_model, file_level_predictions, load_checkpoint, macro_f1, prepare_model_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EM perturbation robustness curve evaluation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--perturb-type", choices=list(ROBUSTNESS_SWEEPS.keys()) + ["iq_imbalance", "iq_phase_deg"], required=True)
    p.add_argument("--strengths", nargs="*", type=float, default=None)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--samples-per-file", type=int, default=256)
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-rate", type=float, default=1e6)
    p.add_argument("--lora-bandwidth", type=float, default=125e3)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--file-vote-mode", default="mean_logits")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def eval_at_strength(model, loader, device, args, ckpt_args, perturb_cfg) -> dict:
    window_rows = []
    for batch in tqdm(loader, leave=False):
        labels = batch["label"]
        iq = batch["iq"].to(device)
        iq = apply_em_perturbation(iq, perturb_cfg)
        model_input = prepare_model_input(iq, args, ckpt_args)
        out = model(model_input)
        scores = out["logits"].detach().cpu()
        probs = F.softmax(scores, dim=-1)
        preds = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1).values
        domains = batch["domains"]
        for idx in range(labels.shape[0]):
            row = {
                "file_path": batch["file_path"][idx],
                "window_index": int(batch["window_index"][idx].item()),
                "sample_offset": int(batch["sample_offset"][idx].item()),
                "label": int(labels[idx].item()),
                "pred": int(preds[idx].item()),
                "correct": int(preds[idx].item() == labels[idx].item()),
                "split": batch["split"][idx],
                "setup": batch["setup"][idx],
                "confidence": float(confidences[idx].item()),
                "_scores": scores[idx],
            }
            for col, field in enumerate(DOMAIN_FIELDS):
                row[field] = int(domains[idx, col].item())
            window_rows.append(row)

    file_rows = file_level_predictions(window_rows, "classifier", vote_mode=args.file_vote_mode)
    num_classes = int(ckpt_args.get("num_classes", 24))
    return {
        "window_acc": sum(r["correct"] for r in window_rows) / max(1, len(window_rows)),
        "file_acc": sum(r["correct"] for r in file_rows) / max(1, len(file_rows)),
        "window_macro_f1": macro_f1([r["label"] for r in window_rows], [r["pred"] for r in window_rows], num_classes),
        "file_macro_f1": macro_f1([r["label"] for r in file_rows], [r["pred"] for r in file_rows], num_classes),
        "num_windows": len(window_rows),
        "num_files": len(file_rows),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    args.model_type = ckpt_args.get("model_type", "rf_hstu")
    args.cnn_input_type = ckpt_args.get("cnn_input_type", "iq")
    args.dim = ckpt_args.get("dim", 64)
    args.depth = ckpt_args.get("depth", 2)
    args.dropout = ckpt_args.get("dropout", 0.1)
    args.window_size = ckpt_args.get("window_size", args.window_size)
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

    model = build_model(args, ckpt, device)

    eval_rows = load_manifest(args.manifest, root=args.root, split=args.eval_split)
    dataset = SigMFIQDataset(
        eval_rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=args.input_norm,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
    )

    key = "iq_imbalance_alpha" if args.perturb_type == "iq_imbalance" else args.perturb_type
    strengths = args.strengths if args.strengths else ROBUSTNESS_SWEEPS.get(key, [0.0])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    clean_acc = None
    for s in strengths:
        cfg = make_sweep_config(args.perturb_type, s, args.sample_rate, args.lora_bandwidth)
        metrics = eval_at_strength(model, loader, device, args, ckpt_args, cfg)
        row = {"perturb_type": args.perturb_type, "strength": s, **metrics}
        if s == strengths[0] or (args.perturb_type == "awgn_snr_db" and s >= 25):
            clean_acc = metrics["file_acc"]
        results.append(row)
        print(f"  {args.perturb_type}={s}: file_acc={metrics['file_acc']:.4f}")

    if clean_acc is None:
        clean_acc = results[0]["file_acc"]
    perturbed = [r["file_acc"] for r in results]
    summary = {
        "clean_file_acc": clean_acc,
        "avg_robust_acc": float(sum(perturbed) / len(perturbed)),
        "robustness_auc": float(sum(perturbed) / len(perturbed)),
        "max_drop": float(clean_acc - min(perturbed)),
    }

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    meta = {
        "manifest": args.manifest,
        "checkpoint": str(ckpt_path),
        "perturb_type": args.perturb_type,
        "strengths": strengths,
        "robustness_summary": summary,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
