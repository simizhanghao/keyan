#!/usr/bin/env python3
"""2A-5 beat 1: C_fft operator unit smoke. No training. No accuracy.

Test 1  C_fft FFT view is deterministic on the same IQ.
Test 2  OOB-scale + lock_inband: C_fft FFT rel-L2 < 1e-3; full FFT is not that small.
Test 3  IQ / amp_phase / ratio-OOB match the legacy full path; only FFT may change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rfhstu.data import SigMFIQDataset, load_manifest
from rfhstu.features import torch_rf_views
from rfhstu.train_utils import apply_receiver_style

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
OUT_DIR = KEYAN / "experiments/paper1_audit/results/cfft_operator_smoke"
FFT_L2_MAX = 1e-3
PATH_L2_MAX = 1e-6


def rx_args() -> object:
    return type(
        "RX",
        (),
        {
            "sample_rate": 1_000_000.0,
            "lora_bandwidth": 125_000.0,
            "rx_factor": "oob_scale",
            "rx_spectral_tilt_db_min": -3.0,
            "rx_spectral_tilt_db_max": 3.0,
            "rx_oob_scale_min": 0.5,
            "rx_oob_scale_max": 2.0,
            "rx_gain_db_min": -6.0,
            "rx_gain_db_max": 6.0,
            "rx_noise_std_min": 0.0,
            "rx_noise_std_max": 0.01,
            "rx_inband_scale_min": 0.7,
            "rx_inband_scale_max": 1.5,
        },
    )()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(a - b, dim=-1)
    den = torch.linalg.vector_norm(a, dim=-1).clamp_min(1e-8)
    return float((num / den).mean().item())


def views(iq: torch.Tensor, source: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch_rf_views(iq, oob_norm="ratio", fft_norm="log_zscore", fft_source=source)


def main() -> int:
    torch.manual_seed(0)
    rows = [r for r in load_manifest(str(KEYAN / "data/paper/cross_day_day1to5_source_only.csv"), root="/data1/hcc/llm4RF", split="val") if int(r.domains["day"]) == 4]
    by_dev: dict[int, object] = {}
    for row in rows:
        by_dev.setdefault(int(row.label), row)
    rows = list(by_dev.values())[:2]
    if len(rows) < 2:
        raise SystemExit("smoke needs 2 Day4 devices")
    ds = SigMFIQDataset(rows, window_size=8192, samples_per_file=2, random_windows=False, seed=0, input_norm="iq_rms")
    iq = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))["iq"]
    if any(int(r.domains["day"]) == 5 for r in rows):
        raise SystemExit("Day5 leaked")

    iq_v, fft_a, oob_v, ap_v = views(iq, "inband")
    _, fft_b, _, _ = views(iq, "inband")
    det_l2 = rel_l2(fft_a.flatten(1), fft_b.flatten(1))
    if det_l2 >= PATH_L2_MAX:
        raise SystemExit(f"Test1 FAIL: C_fft not deterministic rel-L2={det_l2}")

    iq_rx = apply_receiver_style(iq, rx_args(), lock_inband=True)
    _, fft_cfft_rx, _, _ = views(iq_rx, "inband")
    _, fft_full, _, _ = views(iq, "full")
    _, fft_full_rx, _, _ = views(iq_rx, "full")
    cfft_scale_l2 = rel_l2(fft_a.flatten(1), fft_cfft_rx.flatten(1))
    full_scale_l2 = rel_l2(fft_full.flatten(1), fft_full_rx.flatten(1))
    if cfft_scale_l2 >= FFT_L2_MAX:
        raise SystemExit(f"Test2 FAIL: C_fft FFT moved under oob_scale rel-L2={cfft_scale_l2}")
    if full_scale_l2 < 0.01:
        raise SystemExit(f"Test2 FAIL: full FFT too stable ({full_scale_l2}); perturbation not applied")

    iq_full, _, oob_full, ap_full = views(iq, "full")
    iq_l2 = rel_l2(iq_full.flatten(1), iq_v.flatten(1))
    ap_l2 = rel_l2(ap_full.flatten(1), ap_v.flatten(1))
    oob_l2 = rel_l2(oob_full.flatten(1), oob_v.flatten(1))
    if max(iq_l2, ap_l2, oob_l2) >= PATH_L2_MAX:
        raise SystemExit(f"Test3 FAIL: a non-FFT path changed iq={iq_l2} amp={ap_l2} oob={oob_l2}")

    payload = {
        "training": False,
        "day5_used": False,
        "accuracy_used": False,
        "n_files": len(rows),
        "n_windows": int(iq.shape[0]),
        "fft_source": "inband",
        "oob_norm": "ratio",
        "test1_deterministic_l2": det_l2,
        "test2_cfft_oob_scale_l2": cfft_scale_l2,
        "test2_full_oob_scale_l2": full_scale_l2,
        "test3_iq_l2": iq_l2,
        "test3_amp_phase_l2": ap_l2,
        "test3_oob_ratio_l2": oob_l2,
        "verdict": "SMOKE_PASS",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cfft_operator_smoke.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("Test1 deterministic", det_l2)
    print("Test2 C_fft oob_scale", cfft_scale_l2, "full", full_scale_l2)
    print("Test3 iq", iq_l2, "amp_phase", ap_l2, "oob_ratio", oob_l2)
    print("verdict SMOKE_PASS")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
