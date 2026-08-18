#!/usr/bin/env python3
"""Paper 1 Audit 1B: OOB preprocessing spectral audit (no training, Day1-4 only).

Does not read Day5 for selection. Does not write outputs/paper_ready_v3/.
Does not modify src/rfhstu/features.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-6
SAMPLE_RATE = 1_000_000.0
LORA_BW = 125_000.0
WINDOW = 8192
NORMS = ("legacy_zscore", "corrected_zscore", "oob_only_zscore", "ratio", "log_ratio")
WINDOWS_FNS = ("rectangular", "hann")
GUARDS_HZ = (0.0, 12_500.0, 25_000.0)


def load_day14_rows(manifest: Path, data_root: Path) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            day = int(item["day"])
            if day == 5 or item["split"] == "test":
                continue
            if day not in {1, 2, 3, 4}:
                continue
            path = Path(item["path"])
            if not path.is_absolute():
                path = data_root / path
            rows.append(
                {
                    "path": path,
                    "device": int(item["device"]),
                    "label": int(item["label"]),
                    "day": day,
                    "split": item["split"],
                }
            )
    if not rows:
        raise SystemExit("no Day1-4 rows loaded")
    days = sorted({r["day"] for r in rows})
    if 5 in days:
        raise SystemExit("Day5 leaked into 1B loader")
    return rows


def file_offsets(n_samples: int, n_windows: int) -> list[int]:
    max_off = max(0, n_samples - WINDOW)
    if n_windows <= 1 or max_off == 0:
        return [0]
    stride = max(1, max_off // max(1, n_windows - 1))
    return [min(i * stride, max_off) for i in range(n_windows)]


def iq_rms(iq: np.ndarray) -> np.ndarray:
    power = np.sqrt(np.mean(np.abs(iq) ** 2) + EPS)
    return (iq / power).astype(np.complex64, copy=False)


def masks(guard_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.fft.fftshift(np.fft.fftfreq(WINDOW, d=1.0 / SAMPLE_RATE))
    inband = np.abs(freq) <= (LORA_BW / 2.0)
    oob = np.abs(freq) > (LORA_BW / 2.0 + guard_hz)
    return inband, oob, freq


def normalize_oob(
    abs_spec: np.ndarray,
    inband: np.ndarray,
    oob: np.ndarray,
    kind: str,
) -> np.ndarray:
    mag = np.log1p(abs_spec)
    if kind == "legacy_zscore":
        mu = mag.mean()
        sd = mag.std()
        sd = max(sd, EPS)
        full = (mag * oob.astype(np.float64) - mu) / sd
        return full[oob]
    if kind == "corrected_zscore":
        mu = mag.mean()
        sd = max(mag.std(), EPS)
        full = ((mag - mu) / sd) * oob.astype(np.float64)
        return full[oob]
    if kind == "oob_only_zscore":
        vals = mag[oob]
        mu = vals.mean()
        sd = max(vals.std(), EPS)
        return (vals - mu) / sd
    n_in = max(int(inband.sum()), 1)
    inband_rms = np.sqrt(np.sum((abs_spec ** 2) * inband) / n_in + EPS)
    if kind == "ratio":
        return (abs_spec[oob] / inband_rms)
    if kind == "log_ratio":
        return np.log(abs_spec[oob] + EPS) - np.log(inband_rms + EPS)
    raise ValueError(kind)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    sim = float(np.dot(a, b) / (na * nb))
    sim = min(1.0, max(-1.0, sim))
    return 1.0 - sim


def pairwise_stats(file_vecs: list[dict]) -> dict:
    same = []
    diff_same_day = []
    by_key = {(r["device"], r["day"]): r["vec"] for r in file_vecs}
    devices = sorted({r["device"] for r in file_vecs})
    days = sorted({r["day"] for r in file_vecs})
    for d in devices:
        for i, day_a in enumerate(days):
            for day_b in days[i + 1 :]:
                same.append(cosine_distance(by_key[(d, day_a)], by_key[(d, day_b)]))
    for day in days:
        vecs = [r["vec"] for r in file_vecs if r["day"] == day]
        labs = [r["device"] for r in file_vecs if r["day"] == day]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if labs[i] != labs[j]:
                    diff_same_day.append(cosine_distance(vecs[i], vecs[j]))
    d_same = float(np.mean(same)) if same else float("nan")
    d_diff = float(np.mean(diff_same_day)) if diff_same_day else float("nan")
    rho = d_same / d_diff if d_diff and d_diff > 0 else float("nan")
    return {
        "d_same_cross_day": d_same,
        "d_diff_same_day": d_diff,
        "rho_day": rho,
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff_same_day),
    }


def day4_centroid_probe(file_vecs: list[dict]) -> dict:
    train = [r for r in file_vecs if r["day"] in {1, 2, 3}]
    test = [r for r in file_vecs if r["day"] == 4]
    by_dev: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in train:
        by_dev[r["device"]].append(r["vec"])
    centroids = {d: np.mean(vs, axis=0) for d, vs in by_dev.items()}
    correct = 0
    for r in test:
        pred = min(centroids, key=lambda d: cosine_distance(r["vec"], centroids[d]))
        correct += int(pred == r["device"])
    n = len(test)
    acc = correct / n if n else float("nan")
    return {"n_test_files": n, "n_correct": correct, "probe_acc": acc}


def pick_candidates(rows: list[dict]) -> dict:
    """Select two training candidates using Day4 probe + rho only."""
    viable = [
        r
        for r in rows
        if r["rho_day"] < 1.0 and r["probe_acc"] >= (2.0 / 24.0)
    ]
    p0_ref = next(
        r
        for r in rows
        if r["norm"] == "legacy_zscore" and r["fft_window"] == "rectangular" and r["guard_hz"] == 0.0
    )
    p0_hann_guard = [
        r
        for r in rows
        if r["norm"] == "legacy_zscore" and not (r["fft_window"] == "rectangular" and r["guard_hz"] == 0.0)
    ]
    leakage_flag = False
    if p0_hann_guard:
        best_p0_robust = max(p0_hann_guard, key=lambda r: r["probe_acc"])
        if p0_ref["probe_acc"] - best_p0_robust["probe_acc"] >= 0.15 and best_p0_robust["probe_acc"] < (2.0 / 24.0):
            leakage_flag = True

    corrected = [r for r in viable if r["norm"] != "legacy_zscore"]
    if corrected:
        best = max(corrected, key=lambda r: (r["probe_acc"], -r["rho_day"]))
    elif viable:
        best = max(viable, key=lambda r: (r["probe_acc"], -r["rho_day"]))
    else:
        best = max(rows, key=lambda r: (r["probe_acc"], -r["rho_day"]))

    decision = "YELLOW"
    if leakage_flag:
        decision = "RED_LEAKAGE"
    elif corrected and best["rho_day"] < 1.0 and best["probe_acc"] >= (2.0 / 24.0):
        decision = "GO_TWO_CANDIDATES"
    elif p0_ref["rho_day"] < 1.0 and not corrected:
        decision = "YELLOW_ONLY_LEGACY"

    return {
        "candidate_legacy": {
            "norm": p0_ref["norm"],
            "fft_window": p0_ref["fft_window"],
            "guard_hz": p0_ref["guard_hz"],
            "rho_day": p0_ref["rho_day"],
            "probe_acc": p0_ref["probe_acc"],
        },
        "candidate_corrected": {
            "norm": best["norm"],
            "fft_window": best["fft_window"],
            "guard_hz": best["guard_hz"],
            "rho_day": best["rho_day"],
            "probe_acc": best["probe_acc"],
        },
        "n_viable": len(viable),
        "leakage_flag": leakage_flag,
        "decision": decision,
    }


def maybe_plot(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{r['norm']}\n{r['fft_window'][:4]} g{int(r['guard_hz']//1000)}" for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    axes[0].bar(range(len(rows)), [r["rho_day"] for r in rows], color="#4C78A8")
    axes[0].axhline(1.0, color="red", ls="--", lw=1)
    axes[0].set_ylabel("rho_day")
    axes[0].set_title("1B Day1-4 OOB audit (Day5 unused)")
    axes[1].bar(range(len(rows)), [100.0 * r["probe_acc"] for r in rows], color="#F58518")
    axes[1].axhline(100.0 / 24.0, color="red", ls="--", lw=1, label="chance")
    axes[1].set_ylabel("Day4 centroid probe %")
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_and_probe.png", dpi=140)
    plt.close(fig)


def write_report(out_dir: Path, rows: list[dict], picked: dict, n_files: int, n_windows: int) -> None:
    lines = [
        "# Paper 1 Audit 1B — OOB spectral audit",
        "",
        "No training. Day5 was not loaded. Frozen paper numbers were not rewritten.",
        "",
        f"- files (Day1-4): {n_files}",
        f"- windows/file: {n_windows}",
        f"- configs: {len(rows)}",
        f"- decision: **{picked['decision']}**",
        f"- leakage_flag: {picked['leakage_flag']}",
        "",
        "## Candidates for 1C (Day4 selection only)",
        "",
        "### Legacy control",
        "",
        f"- `{picked['candidate_legacy']}`",
        "",
        "### Corrected candidate",
        "",
        f"- `{picked['candidate_corrected']}`",
        "",
        "## Full table",
        "",
        "| norm | window | guard_kHz | rho_day | d_same | d_diff | Day4 probe |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['norm']} | {r['fft_window']} | {r['guard_hz']/1000:.1f} | "
            f"{r['rho_day']:.3f} | {r['d_same_cross_day']:.3f} | {r['d_diff_same_day']:.3f} | "
            f"{100*r['probe_acc']:.1f}% |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- `rho_day < 1`: same device across days is closer than different devices on the same day.",
        "- Day4 probe is nearest-centroid using Day1-3 file means. Chance is 4.17%.",
        "- If Hann+guard wipes device probe while legacy rectangular 0 kHz stays high: leakage RED.",
        "- 1C must still retrain matched Main vs Full; this step only picks two OOB preprocesses.",
        "",
    ]
    (out_dir / "SPECTRAL_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/data1/hcc/llm4RF/new_phase/data/paper/cross_day_day1to5_source_only.csv")
    parser.add_argument("--data-root", default="/data1/hcc/llm4RF")
    parser.add_argument("--out-dir", default="/data1/hcc/llm4RF/new_phase/experiments/paper1_audit/results/spectral_audit")
    parser.add_argument("--windows", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_day14_rows(Path(args.manifest), Path(args.data_root))
    days = sorted({r["day"] for r in rows})
    print(f"loaded files={len(rows)} days={days} windows/file={args.windows}", flush=True)
    if 5 in days:
        raise SystemExit("Day5 leaked")

    hann = np.hanning(WINDOW).astype(np.float64)
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for i, row in enumerate(rows):
        mm = np.memmap(row["path"], dtype=np.complex64, mode="r")
        offs = file_offsets(mm.shape[0], args.windows)
        abs_rect = []
        abs_hann = []
        for off in offs:
            iq = iq_rms(np.asarray(mm[off : off + WINDOW]))
            abs_rect.append(np.abs(np.fft.fftshift(np.fft.fft(iq))))
            abs_hann.append(np.abs(np.fft.fftshift(np.fft.fft(iq * hann))))
        cache[(str(row["path"]), "rectangular")] = {"abs": np.mean(abs_rect, axis=0)}
        cache[(str(row["path"]), "hann")] = {"abs": np.mean(abs_hann, axis=0)}
        if (i + 1) % 16 == 0:
            print(f"fft {i+1}/{len(rows)}", flush=True)

    table = []
    for norm in NORMS:
        for win_name in WINDOWS_FNS:
            for guard in GUARDS_HZ:
                inband, oob, _ = masks(guard)
                file_vecs = []
                for row in rows:
                    abs_spec = cache[(str(row["path"]), win_name)]["abs"]
                    vec = normalize_oob(abs_spec, inband, oob, norm).astype(np.float64)
                    file_vecs.append({**row, "vec": vec})
                stats = pairwise_stats(file_vecs)
                probe = day4_centroid_probe(file_vecs)
                rec = {
                    "norm": norm,
                    "fft_window": win_name,
                    "guard_hz": guard,
                    **stats,
                    **probe,
                }
                table.append(rec)
                print(
                    f"{norm:18s} {win_name:12s} g={guard:7.0f}  rho={stats['rho_day']:.3f}  "
                    f"probe={100*probe['probe_acc']:.1f}%",
                    flush=True,
                )

    picked = pick_candidates(table)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "days": days,
                "n_files": len(rows),
                "windows_per_file": args.windows,
                "window_size": WINDOW,
                "day5_used": False,
                "training": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps({"rows": table, "picked": picked}, indent=2) + "\n")
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (out_dir / "candidates.json").write_text(json.dumps(picked, indent=2) + "\n")
    maybe_plot(out_dir, table)
    write_report(out_dir, table, picked, len(rows), args.windows)
    print(json.dumps(picked, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
