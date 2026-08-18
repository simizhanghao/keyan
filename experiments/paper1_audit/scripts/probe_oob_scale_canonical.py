#!/usr/bin/env python3
"""Phase 2A-0: non-learned C0/C1/C2 on source Day1–4 only. No training. No Day5. No RX2."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rfhstu.data import SigMFIQDataset, load_manifest
from rfhstu.features import torch_rf_views
from rfhstu.train_utils import apply_receiver_style

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
OUT_DIR = KEYAN / "experiments/paper1_audit/results/scale_canonical_probe"
NORMS = ("ratio", "ratio_rms", "ratio_logdc")
NORM_NAME = {"ratio": "C0", "ratio_rms": "C1", "ratio_logdc": "C2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Source-only OOB scale canonical probe.")
    p.add_argument("--manifest", default=str(KEYAN / "data/paper/cross_day_day1to5_source_only.csv"))
    p.add_argument("--root", default="/data1/hcc/llm4RF")
    p.add_argument("--window-size", type=int, default=8192)
    p.add_argument("--samples-per-file", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
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


def oob_vec(iq: torch.Tensor, oob_norm: str) -> torch.Tensor:
    _, _, oob, _ = torch_rf_views(iq, oob_norm=oob_norm, fft_norm="log_zscore")
    return oob.flatten(1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    b = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
    return 1.0 - (a * b).sum(dim=-1)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a-b|| / ||a||. Cosine is scale-blind and cannot test C0."""
    num = torch.linalg.vector_norm(a - b, dim=-1)
    den = torch.linalg.vector_norm(a, dim=-1).clamp_min(1e-8)
    return num / den


def pick_smoke_rows(rows: list, n_devices: int = 2) -> list:
    """Same devices across Day1–4 so d_same and Day4 probe are defined."""
    by_dev_day: dict[int, dict[int, object]] = defaultdict(dict)
    for row in rows:
        by_dev_day[int(row.label)][int(row.domains["day"])] = row
    complete = [lab for lab, days in by_dev_day.items() if {1, 2, 3, 4} <= set(days)]
    if len(complete) < n_devices:
        raise SystemExit(f"smoke needs {n_devices} devices on Day1–4, got {len(complete)}")
    chosen = []
    for lab in complete[:n_devices]:
        for day in (1, 2, 3, 4):
            chosen.append(by_dev_day[lab][day])
    return chosen


def mean_pair_dist(vectors: list[torch.Tensor]) -> float | None:
    if len(vectors) < 2:
        return None
    stacked = torch.stack(vectors)
    stacked = torch.nn.functional.normalize(stacked, dim=-1, eps=1e-8)
    sim = stacked @ stacked.T
    n = stacked.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float((1.0 - sim[iu[0], iu[1]]).mean().item())


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.smoke:
        args.samples_per_file = 2

    train_rows = load_manifest(args.manifest, root=args.root, split="train")
    val_rows = load_manifest(args.manifest, root=args.root, split="val")
    rows = [r for r in train_rows + val_rows if int(r.domains["day"]) != 5]
    if any(int(r.domains["day"]) == 5 for r in rows):
        raise SystemExit("Day5 leaked into the probe")
    if args.smoke:
        rows = pick_smoke_rows(rows, n_devices=2)
    elif args.max_files:
        rows = rows[: args.max_files]
    days = sorted({int(r.domains["day"]) for r in rows})
    if 5 in days:
        raise SystemExit("Day5 present")

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

    file_sums: dict[str, dict[str, torch.Tensor]] = defaultdict(lambda: {n: None for n in NORMS})
    file_counts: dict[str, int] = defaultdict(int)
    file_meta: dict[str, tuple[int, int]] = {}
    inv_l2_sums = {n: 0.0 for n in NORMS}
    inv_cos_sums = {n: 0.0 for n in NORMS}
    inv_n = 0

    for batch in loader:
        iq = batch["iq"]
        iq_a = apply_receiver_style(iq, rx, lock_inband=True)
        for norm in NORMS:
            clean = oob_vec(iq, norm)
            corrupt = oob_vec(iq_a, norm)
            inv_l2_sums[norm] += float(relative_l2(clean, corrupt).sum().item())
            inv_cos_sums[norm] += float(cosine_distance(clean, corrupt).sum().item())
            for i, path in enumerate(batch["file_path"]):
                vec = clean[i].detach().cpu()
                acc = file_sums[path][norm]
                file_sums[path][norm] = vec if acc is None else acc + vec
        inv_n += iq.shape[0]
        for i, path in enumerate(batch["file_path"]):
            file_counts[path] += 1
            if path not in file_meta:
                file_meta[path] = (int(batch["label"][i]), int(batch["domains"][i, 0]))

    invariance_l2 = {NORM_NAME[n]: round(inv_l2_sums[n] / max(1, inv_n), 6) for n in NORMS}
    invariance_cos = {NORM_NAME[n]: round(inv_cos_sums[n] / max(1, inv_n), 6) for n in NORMS}
    file_emb: dict[str, dict[str, torch.Tensor]] = {}
    for path, count in file_counts.items():
        file_emb[path] = {n: file_sums[path][n] / count for n in NORMS}

    rho = {}
    probe = {}
    for norm in NORMS:
        by_dev_day: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for path, (label, day) in file_meta.items():
            by_dev_day[(label, day)].append(file_emb[path][norm])
        means = {k: torch.stack(v).mean(0) for k, v in by_dev_day.items()}
        same: list[float] = []
        devices = sorted({lab for lab, _ in means})
        days_u = sorted({d for _, d in means})
        for lab in devices:
            vecs = [means[(lab, d)] for d in days_u if (lab, d) in means]
            dist = mean_pair_dist(vecs)
            if dist is not None:
                same.append(dist)
        diff: list[float] = []
        keys = list(means)
        for i, k0 in enumerate(keys):
            for k1 in keys[i + 1 :]:
                if k0[0] == k1[0]:
                    continue
                diff.append(float(cosine_distance(means[k0].unsqueeze(0), means[k1].unsqueeze(0)).item()))
        d_same = sum(same) / len(same) if same else math.nan
        d_diff = sum(diff) / len(diff) if diff else math.nan
        rho[NORM_NAME[norm]] = {
            "d_same": round(d_same, 6),
            "d_diff": round(d_diff, 6),
            "rho": None if d_diff == 0 else round(d_same / d_diff, 6),
        }

        centroids: dict[int, list[torch.Tensor]] = defaultdict(list)
        queries: list[tuple[int, torch.Tensor]] = []
        for (lab, day), vec in means.items():
            if day == 4:
                queries.append((lab, vec))
            elif day in {1, 2, 3}:
                centroids[lab].append(vec)
        cents = {lab: torch.stack(vs).mean(0) for lab, vs in centroids.items() if vs}
        correct = 0
        for lab, vec in queries:
            if not cents:
                break
            pred = min(cents, key=lambda c: float(cosine_distance(vec.unsqueeze(0), cents[c].unsqueeze(0))))
            correct += int(pred == lab)
        probe[NORM_NAME[norm]] = {
            "n_day4_files": len(queries),
            "acc": None if not queries else round(correct / len(queries), 4),
        }

    payload = {
        "training": False,
        "day5_used": False,
        "real_rx2_used": False,
        "n_files": len(file_meta),
        "n_windows": inv_n,
        "days": days,
        "samples_per_file": args.samples_per_file,
        "smoke": bool(args.smoke),
        "invariance_relative_l2": invariance_l2,
        "invariance_cosine_distance": invariance_cos,
        "separability": rho,
        "day4_nearest_day123_centroid": probe,
        "note": (
            "Invariance is a sanity check. Method selection uses rho / Day4 centroid only. "
            "Real target-RX is forbidden here. ratio algebra for C0 is unchanged."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "source_day1to4"
    out_json = OUT_DIR / f"scale_canonical_probe_{suffix}.json"
    out_md = OUT_DIR / f"scale_canonical_probe_{suffix}.md"
    lines = [
        "# Phase 2A-0 scale–shape probe",
        "",
        f"files={payload['n_files']}  windows={inv_n}  days={days}  smoke={args.smoke}",
        "Day5 unused. Real RX2 unused. No training.",
        "",
        "## Scale invariance (relative L2; cosine is scale-blind, recorded only)",
        "",
        "| Rep | rel-L2 | cosine |",
        "| --- | -----: | -----: |",
    ]
    for name in ("C0", "C1", "C2"):
        lines.append(f"| {name} | {invariance_l2[name]:.4f} | {invariance_cos[name]:.4f} |")
    lines.extend(["", "## Device separability (file/day means)", "", "| Rep | d_same | d_diff | rho | Day4→D123 acc |", "| --- | -----: | -----: | --: | ------------: |"])
    for name in ("C0", "C1", "C2"):
        r = rho[name]
        acc = probe[name]["acc"]
        acc_s = "?" if acc is None else f"{100 * acc:.1f}%"
        lines.append(f"| {name} | {r['d_same']:.4f} | {r['d_diff']:.4f} | {r['rho']} | {acc_s} |")
    lines.extend(
        [
            "",
            "C1/C2 are not chosen from a real target receiver.",
            "This file does not start seed 0/1 training.",
            "",
        ]
    )
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    if 5 in days or payload["day5_used"] or payload["real_rx2_used"]:
        raise SystemExit("protocol leak")
    if args.smoke:
        if set(days) != {1, 2, 3, 4}:
            raise SystemExit(f"smoke must cover Day1–4, got {days}")
        if invariance_l2["C0"] < 0.05:
            raise SystemExit(f"C0 rel-L2 {invariance_l2['C0']} too small; OOB-scale not visible")
        if invariance_l2["C1"] >= invariance_l2["C0"] or invariance_l2["C2"] >= invariance_l2["C0"]:
            raise SystemExit("C1/C2 rel-L2 must be below C0 on smoke")
        if any(math.isnan(rho[name]["d_same"]) for name in ("C0", "C1", "C2")):
            raise SystemExit("smoke d_same is nan; day coverage failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
