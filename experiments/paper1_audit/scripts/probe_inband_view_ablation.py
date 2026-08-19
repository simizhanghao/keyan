#!/usr/bin/env python3
"""2A-4: which main-path view swap kills cnn_stem leak under oob_scale.

Frozen C1 seed 0 stem. No training. Day4 val only.
Does not change features.py / models.py / evaluate.py defaults.
E_all_inband STABLE is a lock_inband control, not a paper finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfhstu.data import SigMFIQDataset, load_manifest
from rfhstu.features import torch_rf_views
from rfhstu.train_utils import apply_receiver_style

from probe_scale_path_leak import (  # type: ignore
    LIVE,
    STABLE,
    band_masks,
    load_embedder,
    read_path,
    relative_l2,
    rx_args,
)

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
OUT_DIR = KEYAN / "experiments/paper1_audit/results/inband_view_ablation"
CKPT = KEYAN / "experiments/paper1_audit/results/matched_seed0/runs/C_full_ratio_rms/seed_0/best.pt"

# (iq, fft, amp_phase) source: full IQ views vs in-band reconstructed views
ARMS = {
    "R0_full": ("full", "full", "full"),
    "A_amp": ("full", "full", "inband"),
    "B_iq": ("inband", "full", "full"),
    "C_fft": ("full", "inband", "full"),
    "D_iq_amp": ("inband", "full", "inband"),
    "E_all_inband": ("inband", "inband", "inband"),
}
PREFERENCE = ("A_amp", "B_iq", "C_fft", "D_iq_amp", "E_all_inband")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate which main view carries the C1 stem leak.")
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


def reconstruct_inband_iq(iq: torch.Tensor, in_band: torch.Tensor) -> torch.Tensor:
    z = torch.complex(iq[:, 0], iq[:, 1])
    spec = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    spec = spec * in_band.to(dtype=spec.dtype).view(1, -1)
    z2 = torch.fft.ifft(torch.fft.ifftshift(spec, dim=-1), dim=-1)
    return torch.stack([z2.real, z2.imag], dim=1).to(dtype=iq.dtype)


def main_views(iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    iq_v, fft_v, _, ap_v = torch_rf_views(iq, oob_norm="ratio_rms", fft_norm="log_zscore")
    return iq_v, fft_v, ap_v


def mix(iq_v: torch.Tensor, fft_v: torch.Tensor, ap_v: torch.Tensor) -> torch.Tensor:
    return torch.cat([iq_v, fft_v, ap_v], dim=1)


def pick_source(name: str, full: torch.Tensor, inband: torch.Tensor) -> torch.Tensor:
    return full if name == "full" else inband


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
    if embedder.cnn_stem is None:
        raise SystemExit("C1 embedder has no cnn_stem")

    sums = {name: 0.0 for name in ARMS}
    n = 0
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            iq_a = apply_receiver_style(iq, rx, lock_inband=True)
            in_band, _ = band_masks(iq.shape[-1], device)
            iq_ib = reconstruct_inband_iq(iq, in_band)
            iq_ib_a = reconstruct_inband_iq(iq_a, in_band)
            full = main_views(iq)
            full_a = main_views(iq_a)
            ib = main_views(iq_ib)
            ib_a = main_views(iq_ib_a)
            for name, (iq_s, fft_s, ap_s) in ARMS.items():
                x = mix(pick_source(iq_s, full[0], ib[0]), pick_source(fft_s, full[1], ib[1]), pick_source(ap_s, full[2], ib[2]))
                x_a = mix(pick_source(iq_s, full_a[0], ib_a[0]), pick_source(fft_s, full_a[1], ib_a[1]), pick_source(ap_s, full_a[2], ib_a[2]))
                stem = embedder.cnn_stem(x).flatten(1)
                stem_a = embedder.cnn_stem(x_a).flatten(1)
                sums[name] += float(relative_l2(stem, stem_a).sum().item())
            n += iq.shape[0]

    means = {k: round(sums[k] / max(1, n), 6) for k in ARMS}
    readings = {k: read_path(means[k]) for k in ARMS}
    if readings["R0_full"] != "LIVE":
        raise SystemExit(f"R0 cnn_stem rel-L2 {means['R0_full']} is not LIVE; 2A-3 not replicated")
    if readings["E_all_inband"] != "STABLE":
        raise SystemExit(
            f"E_all_inband rel-L2 {means['E_all_inband']} not STABLE; in-band recon / lock_inband bug"
        )

    chosen = next((name for name in PREFERENCE if readings[name] == "STABLE"), None)
    if chosen is None:
        verdict = "NO_KILL"
        operator = ""
    else:
        verdict = "SMALLEST_KILL"
        operator = chosen

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
        "cnn_stem_relative_l2": means,
        "reading": readings,
        "smallest_kill": chosen,
        "verdict": verdict,
        "thresholds": {"stable_lt": STABLE, "live_ge": LIVE},
        "note": (
            "E_all_inband STABLE is a lock_inband control, not a scientific finding. "
            "This file does not change model defaults or start training."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "day4"
    out_json = OUT_DIR / f"inband_view_ablation_{suffix}.json"
    out_md = OUT_DIR / f"inband_view_ablation_{suffix}.md"
    lines = [
        "# 2A-4 in-band view ablation (Day4, frozen C1 stem)",
        "",
        f"files={payload['n_files']}  windows={n}  smoke={args.smoke}",
        "Day5 unused. No training. E_all_inband is a control.",
        "",
        "| Arm | iq | fft | amp_phase | stem rel-L2 | Reading |",
        "| --- | --- | --- | --- | -----: | --- |",
    ]
    for name, src in ARMS.items():
        lines.append(f"| {name} | {src[0]} | {src[1]} | {src[2]} | {means[name]:.4f} | {readings[name]} |")
    lines.extend(
        [
            "",
            f"smallest STABLE kill = {chosen or '(none)'}",
            f"verdict = {verdict}",
            "STABLE < 0.01; LIVE ≥ 0.05. Do not train from this file.",
            "",
        ]
    )
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
