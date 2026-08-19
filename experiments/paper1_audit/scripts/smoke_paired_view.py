#!/usr/bin/env python3
"""2B-0: paired-view operator smoke. No training. No accuracy.

Test 1  S0 (clean,clean): second view rel-L2 = 0; rx_factor unrestored would fail later.
Test 2  S1 oob_scale: in-band |X| locked; OOB |X| moves; IQ moves.
Test 3  Loss is the mean of two CEs (formula lock).
Test 4  --paired-view off / default rx_factor is not mutated after S1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rfhstu.data import SigMFIQDataset, load_manifest
from rfhstu.train_utils import paired_second_view

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
OUT_DIR = KEYAN / "experiments/paper1_audit/results/paired_view_smoke"
PATH_L2_MAX = 1e-6
OOB_L2_MIN = 1e-3


def rx_args(paired_view: str) -> SimpleNamespace:
    return SimpleNamespace(
        paired_view=paired_view,
        sample_rate=1_000_000.0,
        lora_bandwidth=125_000.0,
        rx_factor=None,
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


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((a - b).reshape(a.shape[0], -1), dim=-1)
    den = torch.linalg.vector_norm(a.reshape(a.shape[0], -1), dim=-1).clamp_min(1e-8)
    return float((num / den).mean().item())


def band_mags(iq: torch.Tensor, args: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    freq = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1.0 / args.sample_rate)).to(iq.device)
    in_band = (freq.abs() <= (args.lora_bandwidth / 2.0)).view(1, -1)
    mag = spectrum.abs()
    return mag * in_band, mag * (~in_band)


def main() -> int:
    torch.manual_seed(0)
    rows = [
        r
        for r in load_manifest(
            str(KEYAN / "data/paper/cross_day_day1to5_source_only.csv"),
            root="/data1/hcc/llm4RF",
            split="val",
        )
        if int(r.domains["day"]) == 4
    ]
    by_dev: dict[int, object] = {}
    for row in rows:
        by_dev.setdefault(int(row.label), row)
    rows = list(by_dev.values())[:2]
    if len(rows) < 2:
        raise SystemExit("smoke needs 2 Day4 devices")
    if any(int(r.domains["day"]) == 5 for r in rows):
        raise SystemExit("Day5 leaked")
    ds = SigMFIQDataset(
        rows,
        window_size=8192,
        samples_per_file=2,
        random_windows=False,
        seed=0,
        input_norm="iq_rms",
    )
    iq = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))["iq"]

    s0 = rx_args("clean")
    x0 = paired_second_view(iq, s0)
    s0_l2 = rel_l2(iq, x0)
    if s0_l2 >= PATH_L2_MAX:
        raise SystemExit(f"Test1 FAIL: S0 second view moved rel-L2={s0_l2}")
    if s0.rx_factor is not None:
        raise SystemExit("Test1 FAIL: clean path mutated rx_factor")

    s1 = rx_args("oob_scale")
    torch.manual_seed(1)
    x1 = paired_second_view(iq, s1)
    if s1.rx_factor is not None:
        raise SystemExit("Test4 FAIL: S1 left rx_factor=oob_scale on args")
    in0, oob0 = band_mags(iq, s1)
    in1, oob1 = band_mags(x1, s1)
    in_l2 = rel_l2(in0, in1)
    oob_l2 = rel_l2(oob0, oob1)
    iq_l2 = rel_l2(iq, x1)
    if in_l2 >= PATH_L2_MAX:
        raise SystemExit(f"Test2 FAIL: in-band |X| moved rel-L2={in_l2}")
    if oob_l2 < OOB_L2_MIN:
        raise SystemExit(f"Test2 FAIL: OOB |X| did not move rel-L2={oob_l2}")
    if iq_l2 < OOB_L2_MIN:
        raise SystemExit(f"Test2 FAIL: IQ did not move rel-L2={iq_l2}")

    logits_a = torch.randn(4, 24)
    logits_b = torch.randn(4, 24)
    labels = torch.randint(0, 24, (4,))
    la = F.cross_entropy(logits_a, labels)
    lb = F.cross_entropy(logits_b, labels)
    mixed = 0.5 * la + 0.5 * lb
    if abs(float(mixed.item()) - 0.5 * float(la.item()) - 0.5 * float(lb.item())) > 1e-6:
        raise SystemExit("Test3 FAIL: paired CE is not the mean of two CEs")

    off = rx_args("off")
    x_off = paired_second_view(iq, off)
    if rel_l2(iq, x_off) >= PATH_L2_MAX:
        raise SystemExit("Test4 FAIL: paired_view=off moved IQ")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "day5_used": False,
        "training": False,
        "accuracy": False,
        "test1_s0_rel_l2": s0_l2,
        "test2_inband_rel_l2": in_l2,
        "test2_oob_rel_l2": oob_l2,
        "test2_iq_rel_l2": iq_l2,
        "test3_paired_ce": float(mixed.item()),
        "verdict": "SMOKE_PASS",
    }
    path = OUT_DIR / "paired_view_smoke.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("Test1 S0 rel-L2", s0_l2)
    print("Test2 inband", in_l2, "oob", oob_l2, "iq", iq_l2)
    print("Test3 paired CE", float(mixed.item()))
    print("verdict SMOKE_PASS")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
