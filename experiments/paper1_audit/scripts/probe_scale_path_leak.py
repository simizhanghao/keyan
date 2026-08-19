#!/usr/bin/env python3
"""2A-3: which feature path still moves under oob_scale after C1.

No training. Day4 val only. Day5 unused. Real RX2 unused.
Does not change oob_norm / fft_norm / input_norm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rfhstu.data import SigMFIQDataset, load_manifest
from rfhstu.features import torch_rf_views
from rfhstu.models import RFPatchEmbedder
from rfhstu.train_utils import apply_receiver_style

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
OUT_DIR = KEYAN / "experiments/paper1_audit/results/scale_path_leak"
CKPT = KEYAN / "experiments/paper1_audit/results/matched_seed0/runs/C_full_ratio_rms/seed_0/best.pt"
LIVE = 0.05
STABLE = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose remaining scale paths after C1.")
    p.add_argument("--manifest", default=str(KEYAN / "data/paper/cross_day_day1to5_source_only.csv"))
    p.add_argument("--root", default="/data1/hcc/llm4RF")
    p.add_argument("--checkpoint", default=str(CKPT))
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--samples-per-file", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def rx_args() -> argparse.Namespace:
    return argparse.Namespace(
        sample_rate=1_000_000.0,
        lora_bandwidth=125_000.0,
        rx_factor="oob_scale",
        rx_spectral_tilt_db_min=-3.0,
        rx_spectral_tilt_db_max=3.0,
        rx_oob_scale_min=0.5,
        rx_oob_scale_max=2.0,
        rx_gain_db_min=-6.0,
        rx_gain_db_max=6.0,
        rx_noise_std_min=0.0,
        rx_noise_std_max=0.01,
        rx_inband_scale_min=0.7,
        rx_inband_scale_max=1.5,
    )


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.vector_norm(a - b, dim=-1)
    den = torch.linalg.vector_norm(a, dim=-1).clamp_min(1e-8)
    return num / den


def read_path(mean: float) -> str:
    if mean < STABLE:
        return "STABLE"
    if mean >= LIVE:
        return "LIVE"
    return "WEAK"


def band_masks(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    freq = torch.fft.fftshift(torch.fft.fftfreq(length, d=1.0 / 1_000_000.0)).to(device)
    in_band = freq.abs() <= (125_000.0 / 2.0)
    return in_band, ~in_band


def linear_mag(iq: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.fftshift(torch.fft.fft(torch.complex(iq[:, 0], iq[:, 1]), dim=-1), dim=-1)
    return spec.abs()


def load_embedder(ckpt_path: Path, device: torch.device) -> RFPatchEmbedder:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})
    if saved.get("oob_norm") != "ratio_rms":
        raise SystemExit(f"checkpoint oob_norm must be ratio_rms, got {saved.get('oob_norm')}")
    embedder = RFPatchEmbedder(
        window_size=int(saved.get("window_size", 8192)),
        patch_size=int(saved.get("patch_size", 256)),
        sample_rate=float(saved.get("sample_rate", 1_000_000.0)),
        lora_bandwidth=float(saved.get("lora_bandwidth", 125_000.0)),
        spreading_factor=int(saved.get("spreading_factor", 7)),
        use_oob=True,
        oob_fusion_type=saved.get("oob_fusion_type", "cross_attn_oob"),
        use_oob_cross_attention=bool(saved.get("use_oob_cross_attention", True)),
        patch_embed_type=saved.get("patch_embed_type", "cnn_stem"),
        dim=int(saved.get("dim", 64)),
        cnn_stem_dim=int(saved.get("cnn_stem_dim", 32)),
        cnn_stem_kernels=saved.get("cnn_stem_kernels", "7,5"),
        fft_norm=saved.get("fft_norm", "log_zscore"),
        oob_norm="ratio_rms",
    )
    state = {k[len("embedder.") :]: v for k, v in ckpt["model"].items() if k.startswith("embedder.")}
    embedder.load_state_dict(state, strict=True)
    embedder.to(device).eval()
    return embedder


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.smoke:
        args.samples_per_file = 2
        args.batch_size = 4

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = [r for r in load_manifest(args.manifest, root=args.root, split="val") if int(r.domains["day"]) == 4]
    if any(int(r.domains["day"]) == 5 for r in rows):
        raise SystemExit("Day5 leaked")
    if args.smoke:
        by_dev: dict[int, object] = {}
        for row in rows:
            by_dev.setdefault(int(row.label), row)
        rows = list(by_dev.values())[:2]
        if len(rows) < 2:
            raise SystemExit("smoke needs 2 Day4 devices")

    ds = SigMFIQDataset(
        rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm="iq_rms",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rx = rx_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing C1 seed 0 checkpoint: {ckpt_path}")
    embedder = load_embedder(ckpt_path, device)

    keys = (
        "oob_c1",
        "fft_inband_linear",
        "fft_log_zscore",
        "fft_inband_of_log_zscore",
        "iq_time",
        "amp_phase",
        "cnn_stem",
    )
    sums = {k: 0.0 for k in keys}
    n = 0
    rms_ratio_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            iq_a = apply_receiver_style(iq, rx, lock_inband=True)
            in_band, _ = band_masks(iq.shape[-1], device)
            in_band_f = in_band.to(iq.dtype).view(1, -1)

            iq_v, fft_v, oob_v, ap_v = torch_rf_views(iq, oob_norm="ratio_rms", fft_norm="log_zscore")
            iq_va, fft_va, oob_va, ap_va = torch_rf_views(iq_a, oob_norm="ratio_rms", fft_norm="log_zscore")
            mag = linear_mag(iq)
            mag_a = linear_mag(iq_a)

            pairs = {
                "oob_c1": (oob_v.flatten(1), oob_va.flatten(1)),
                "fft_inband_linear": (mag * in_band_f, mag_a * in_band_f),
                "fft_log_zscore": (fft_v.flatten(1), fft_va.flatten(1)),
                "fft_inband_of_log_zscore": (
                    (fft_v.squeeze(1) * in_band_f).flatten(1),
                    (fft_va.squeeze(1) * in_band_f).flatten(1),
                ),
                "iq_time": (iq.flatten(1), iq_a.flatten(1)),
                "amp_phase": (ap_v.flatten(1), ap_va.flatten(1)),
            }
            stem, _ = embedder(iq)
            stem_a, _ = embedder(iq_a)
            pairs["cnn_stem"] = (stem.flatten(1), stem_a.flatten(1))
            for key, (clean, corrupt) in pairs.items():
                sums[key] += float(relative_l2(clean, corrupt).sum().item())
            rms = torch.sqrt((iq[:, 0] ** 2 + iq[:, 1] ** 2).mean(dim=-1) + 1e-6)
            rms_a = torch.sqrt((iq_a[:, 0] ** 2 + iq_a[:, 1] ** 2).mean(dim=-1) + 1e-6)
            rms_ratio_sum += float((rms_a / rms.clamp_min(1e-6)).sum().item())
            n += iq.shape[0]

    means = {k: round(sums[k] / max(1, n), 6) for k in keys}
    readings = {k: read_path(means[k]) for k in keys}
    if readings["oob_c1"] != "STABLE":
        raise SystemExit(f"oob_c1 rel-L2 {means['oob_c1']} not STABLE; 2A-0 replication failed")
    if readings["fft_inband_linear"] != "STABLE":
        raise SystemExit(f"fft_inband_linear rel-L2 {means['fft_inband_linear']} not STABLE; lock_inband failed")

    leak_keys = ("fft_log_zscore", "fft_inband_of_log_zscore", "iq_time", "amp_phase", "cnn_stem")
    live = [k for k in leak_keys if readings[k] == "LIVE"]
    verdict = "LEAK_CONFIRMED" if live else "NO_LIVE_LEAK"

    payload = {
        "training": False,
        "day5_used": False,
        "real_rx2_used": False,
        "eval_split": "val",
        "days": [4],
        "n_files": len(rows),
        "n_windows": n,
        "samples_per_file": args.samples_per_file,
        "smoke": bool(args.smoke),
        "checkpoint": str(ckpt_path),
        "oob_norm": "ratio_rms",
        "input_norm": "iq_rms",
        "fft_norm": "log_zscore",
        "rx_factor": "oob_scale",
        "lock_inband": True,
        "second_iq_rms_after_rx": False,
        "relative_l2": means,
        "reading": readings,
        "rms_after_rx_over_before": round(rms_ratio_sum / max(1, n), 6),
        "live_paths": live,
        "verdict": verdict,
        "thresholds": {"stable_lt": STABLE, "live_ge": LIVE},
        "note": (
            "Eval path is dataset iq_rms once, then RX, no second RMS. "
            "fft_log_zscore uses the full spectrum (model default). "
            "This file does not change norms or start a new train."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "day4"
    out_json = OUT_DIR / f"scale_path_leak_{suffix}.json"
    out_md = OUT_DIR / f"scale_path_leak_{suffix}.md"
    lines = [
        "# 2A-3 scale path leak (Day4, no training)",
        "",
        f"files={payload['n_files']}  windows={n}  smoke={args.smoke}",
        "Day5 unused. Real RX2 unused. C1 seed 0 embedder only.",
        "",
        "| Path | rel-L2 | Reading |",
        "| --- | -----: | --- |",
    ]
    for key in keys:
        lines.append(f"| {key} | {means[key]:.4f} | {readings[key]} |")
    lines.extend(
        [
            "",
            f"rms(after RX) / rms(before) = {payload['rms_after_rx_over_before']:.4f}",
            f"verdict = {verdict}",
            "live paths: " + (", ".join(live) if live else "(none)"),
            "",
            "STABLE < 0.01; LIVE ≥ 0.05. Do not retune norms from this file.",
            "",
        ]
    )
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    if args.smoke and verdict == "NO_LIVE_LEAK":
        print("smoke: no LIVE path; full run may still move more windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
