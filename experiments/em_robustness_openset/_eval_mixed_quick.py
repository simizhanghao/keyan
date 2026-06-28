#!/usr/bin/env python3
"""Quick eval of MIXED_STRESS_PRESETS for summary/plot."""
from __future__ import annotations

import argparse
import csv
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
from rfhstu.em_perturbations import MIXED_STRESS_PRESETS, apply_em_perturbation  # noqa: E402
from evaluate import build_model, file_level_predictions, load_checkpoint, prepare_model_input  # noqa: E402

CKPT = "outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", required=True)
    p.add_argument("--manifest", default="data/paper/cross_day_day1to5_source_only.csv")
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--samples-per-file", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def eval_preset(model, loader, device, args, ckpt_args, preset_name, cfg) -> dict:
    window_rows = []
    for batch in tqdm(loader, leave=False):
        labels = batch["label"]
        iq = batch["iq"].to(device)
        iq = apply_em_perturbation(iq, cfg)
        model_input = prepare_model_input(iq, args, ckpt_args)
        out = model(model_input)
        scores = out["logits"].detach().cpu()
        preds = F.softmax(scores, dim=-1).argmax(dim=-1)
        domains = batch["domains"]
        for idx in range(labels.shape[0]):
            row = {
                "file_path": batch["file_path"][idx],
                "label": int(labels[idx].item()),
                "pred": int(preds[idx].item()),
                "correct": int(preds[idx].item() == labels[idx].item()),
                "split": batch["split"][idx],
                "setup": batch["setup"][idx],
                "_scores": scores[idx],
            }
            for col, field in enumerate(DOMAIN_FIELDS):
                row[field] = int(domains[idx, col].item())
            window_rows.append(row)
    file_rows = file_level_predictions(window_rows, "classifier", vote_mode="mean_logits")
    file_acc = sum(r["correct"] for r in file_rows) / max(1, len(file_rows))
    window_acc = sum(r["correct"] for r in window_rows) / max(1, len(window_rows))
    return {
        "perturb_type": "mixed_stress",
        "preset": preset_name,
        "strength": preset_name,
        "window_acc": window_acc,
        "file_acc": file_acc,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    args.root = str(ROOT)
    args.model_type = ckpt_args.get("model_type", "rf_hstu")
    args.cnn_input_type = ckpt_args.get("cnn_input_type", "iq")
    args.dim = ckpt_args.get("dim", 64)
    args.depth = ckpt_args.get("depth", 2)
    args.dropout = ckpt_args.get("dropout", 0.1)
    args.window_size = int(ckpt_args.get("window_size", 8192))
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
    args.sample_rate = float(ckpt_args.get("sample_rate", 1e6))
    args.lora_bandwidth = float(ckpt_args.get("lora_bandwidth", 125e3))

    model = build_model(args, ckpt, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = load_manifest(args.manifest, root=str(ROOT), split="test")
    ds = SigMFIQDataset(
        rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=0,
        input_norm=ckpt_args.get("input_norm", "iq_rms"),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    for name, cfg in MIXED_STRESS_PRESETS.items():
        print(f"mixed preset: {name}")
        results.append(eval_preset(model, loader, device, args, ckpt_args, name, cfg))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["perturb_type", "preset", "strength", "window_acc", "file_acc"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
