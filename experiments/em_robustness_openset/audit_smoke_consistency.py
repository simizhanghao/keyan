#!/usr/bin/env python3
"""Smoke audit: verify clean-equivalent conditions share the same baseline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_CHECKPOINT, DEFAULT_MANIFEST, populate_args_from_ckpt, resolve_device  # noqa: E402
from evaluate import build_model, load_checkpoint  # noqa: E402
from eval_robustness_curves import eval_at_strength  # noqa: E402
from rfhstu.data import SigMFIQDataset, load_manifest  # noqa: E402
from rfhstu.em_perturbations import (  # noqa: E402
    EmPerturbConfig,
    apply_awgn,
    apply_em_perturbation,
    clean_config,
    make_sweep_config,
)
from torch.utils.data import DataLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--samples-per-file", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--file-vote-mode", default="mean_logits")
    return p.parse_args()


def build_clean_conditions(sr: float, bw: float) -> list[tuple[str, EmPerturbConfig, str]]:
    return [
        ("no_perturb", clean_config(sr, bw), "skip apply_em_perturbation"),
        ("awgn_snr_inf", make_sweep_config("awgn_snr_db", 100.0, sr, bw), "SNR>=100 dB, no noise"),
        ("cfo_norm_0", make_sweep_config("cfo_norm", 0.0, sr, bw), "CFO norm=0 skipped"),
        ("phase_noise_0", make_sweep_config("phase_noise_std", 0.0, sr, bw), "sigma=0 skipped"),
        ("iq_amp0_phase0", make_sweep_config("iq_amp_db", 0.0, sr, bw), "IQ imbalance 0 skipped"),
        ("filter_tilt_0", make_sweep_config("filter_tilt_norm", 0.0, sr, bw), "tilt=0 skipped"),
        ("awgn_30db_ref", make_sweep_config("awgn_snr_db", 30.0, sr, bw), "NOT clean — reference only"),
    ]


def window_sampling_audit(args: argparse.Namespace) -> list[dict]:
    rows = load_manifest(args.manifest, root=args.root, split=args.eval_split)
    ds = SigMFIQDataset(
        rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=args.input_norm,
    )
    out = []
    for idx in range(min(len(ds), args.samples_per_file * 3)):
        sample = ds[idx]
        out.append(
            {
                "dataset_index": idx,
                "file_path": sample["file_path"],
                "window_index": int(sample["window_index"].item()),
                "sample_offset": int(sample["sample_offset"].item()),
                "label": int(sample["label"].item()),
            }
        )
    return out


def awgn_power_audit(device: torch.device) -> dict:
    """Sanity-check AWGN power on synthetic normalized IQ."""
    iq = torch.randn(4, 2, 8192, device=device)
    iq = iq / iq.square().mean(dim=(1, 2), keepdim=True).sqrt()
    snr_db = 30.0
    noisy = apply_awgn(iq, snr_db, 1e6, 125e3)
    sig_p = iq.square().mean(dim=(1, 2))
    noise = noisy - iq
    noise_p = noise.square().mean(dim=(1, 2))
    snr_meas = 10.0 * torch.log10(sig_p / noise_p.clamp_min(1e-12))
    return {
        "snr_target_db": snr_db,
        "snr_measured_mean_db": float(snr_meas.mean().item()),
        "snr_measured_std_db": float(snr_meas.std().item()),
        "note": "Measured on per-window iq_rms-like normalized tensor",
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir or f"experiments/em_robustness_openset/results/smoke_audit_{datetime.now():%Y%m%d_%H%M}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt_args = populate_args_from_ckpt(args, ckpt)
    args.file_vote_mode = getattr(args, "file_vote_mode", "mean_logits")

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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sr = float(args.sample_rate)
    bw = float(args.lora_bandwidth)
    conditions = build_clean_conditions(sr, bw)

    equiv_rows = []
    file_preds: dict[str, dict[str, int]] = {}

    for name, cfg, note in conditions:
        metrics = eval_at_strength(model, loader, device, args, ckpt_args, cfg)
        equiv_rows.append(
            {
                "condition": name,
                "note": note,
                "file_acc": metrics["file_acc"],
                "window_acc": metrics["window_acc"],
                "num_windows": metrics["num_windows"],
                "num_files": metrics["num_files"],
            }
        )
        print(f"{name}: file_acc={metrics['file_acc']:.4f} ({note})")

    clean_accs = [r["file_acc"] for r in equiv_rows if r["condition"] != "awgn_30db_ref"]
    clean_min, clean_max = min(clean_accs), max(clean_accs)
    spread = clean_max - clean_min
    passed = spread <= 1.0 / max(1, equiv_rows[0]["num_files"])

    # Per-file prediction agreement (no_perturb vs cfo_norm_0)
    def collect_file_preds(cfg: EmPerturbConfig) -> dict[str, int]:
        from evaluate import file_level_predictions
        import torch.nn.functional as F
        from evaluate import prepare_model_input

        from rfhstu.data import DOMAIN_FIELDS

        window_rows = []
        for batch in loader:
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
        file_rows = file_level_predictions(window_rows, "classifier", vote_mode=args.file_vote_mode)
        return {r["file_path"]: int(r["pred"]) for r in file_rows}

    with torch.no_grad():
        preds_no = collect_file_preds(clean_config(sr, bw))
        preds_cfo0 = collect_file_preds(make_sweep_config("cfo_norm", 0.0, sr, bw))
    disagree = sum(1 for k in preds_no if preds_no[k] != preds_cfo0.get(k, -1))

    win_rows = window_sampling_audit(args)
    with (out_dir / "window_sampling_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        w.writerows(win_rows)

    with (out_dir / "clean_equivalence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(equiv_rows[0].keys()))
        w.writeheader()
        w.writerows(equiv_rows)

    ckpt_audit = {
        "checkpoint": str(ckpt_path.resolve()),
        "manifest": args.manifest,
        "eval_split": args.eval_split,
        "samples_per_file": args.samples_per_file,
        "seed": args.seed,
        "file_vote_mode": args.file_vote_mode,
        "input_norm": args.input_norm,
        "num_classes": int(ckpt.get("num_classes", ckpt_args.get("num_classes", 24))),
        "num_test_files": len(eval_rows),
        "awgn_power_audit": awgn_power_audit(device),
    }
    (out_dir / "checkpoint_audit.json").write_text(json.dumps(ckpt_audit, indent=2), encoding="utf-8")

    awgn30 = next(r for r in equiv_rows if r["condition"] == "awgn_30db_ref")
    clean_mean = sum(clean_accs) / len(clean_accs)

    report_lines = [
        "# Smoke Audit Report",
        "",
        f"**Date:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Output:** `{out_dir}`",
        "",
        "## Verdict",
        "",
        f"- Clean-equivalent spread (file-acc max-min): **{spread:.4f}** ({spread*100:.1f} pp)",
        f"- Per-file pred disagree (no_perturb vs cfo_norm_0): **{disagree}** files",
        f"- Audit passed: **{'YES' if passed and disagree <= 1 else 'NO — fix before full run'}**",
        "",
        "## Answers",
        "",
        "1. **Same checkpoint?** Yes — single load: `" + str(ckpt_path) + "`",
        f"2. **Same windows/file?** Yes — `{args.samples_per_file}` windows, seed `{args.seed}`, deterministic offsets (see window_sampling_audit.csv)",
        f"3. **Same Day5 test manifest?** Yes — `{args.manifest}`, split=`{args.eval_split}`, `{len(eval_rows)}` files",
        f"4. **Same file-level voting?** Yes — `{args.file_vote_mode}`",
        "5. **Perturbation disabled = clean?** Zero-strength CFO/phase/IQ/filter now skip apply; AWGN uses SNR>=100 as clean",
        f"6. **AWGN power?** Target 30 dB, measured mean {ckpt_audit['awgn_power_audit']['snr_measured_mean_db']:.2f} dB (I/Q split fixed)",
        "7. **Normalization?** `input_norm=" + str(args.input_norm) + "` in dataset; perturb after norm",
        "",
        "## Key clarification",
        "",
        f"- **Clean baseline (mean over clean-equivalent):** {clean_mean*100:.1f}%",
        f"- **AWGN 30 dB (perturbed, NOT clean):** {awgn30['file_acc']*100:.1f}%",
        "",
        "Prior smoke compared AWGN **30 dB** to CFO **clean (0.0)** — different conditions.",
        "After audit, compare clean-equivalent rows only for baseline consistency.",
        "",
        "## Clean-equivalent results",
        "",
        "| Condition | File-Acc | Note |",
        "|-----------|----------|------|",
    ]
    for r in equiv_rows:
        report_lines.append(f"| {r['condition']} | {r['file_acc']*100:.1f}% | {r['note']} |")

    (out_dir / "SMOKE_AUDIT_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Audit passed={passed}, clean spread={spread:.4f}, disagree={disagree}")
    print(f"Wrote {out_dir / 'SMOKE_AUDIT_REPORT.md'}")


if __name__ == "__main__":
    main()
