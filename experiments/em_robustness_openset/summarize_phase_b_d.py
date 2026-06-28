#!/usr/bin/env python3
"""Freeze Phase B/D full results: summary CSVs, reports, RUN_MANIFEST."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rfhstu.em_perturbations import MIXED_STRESS_PRESETS, ROBUSTNESS_SWEEPS  # noqa: E402

EM_FULL = ROOT / "experiments/em_robustness_openset/results/em_full_20260628"
OPENSET_DIR = ROOT / "experiments/em_robustness_openset/results/openset_full_20260628_1123"
FIG_DIR = ROOT / "docs/thesis_chapter5_em_openset/figures"
CKPT = "outputs/paper_ready_v3/step1_phase7_clean/runs/F_cross_attn_chirp_plain/seed_0/best.pt"
PYTHON = "/new_nfs/haiyu/anaconda3/bin/python"

SWEEP_FILES = {
    "AWGN": "awgn_snr_db_sweep.csv",
    "CFO": "cfo_norm_sweep.csv",
    "Narrowband": "narrowband_sir_db_sweep.csv",
    "Phase noise": "phase_noise_std_sweep.csv",
    "IQ imbalance (amp)": "iq_amp_db_sweep.csv",
    "Filter drift": "filter_tilt_norm_sweep.csv",
}

CLEAN_STRENGTH = {
    "awgn_snr_db": 100.0,
    "cfo_norm": 0.0,
    "narrowband_sir_db": 30.0,
    "phase_noise_std": 0.0,
    "iq_amp_db": 0.0,
    "filter_tilt_norm": 0.0,
}

EMCR_MODERATE = {
    "AWGN SNR (dB)": "30–15",
    "CFO norm": "0.001–0.01",
    "Narrowband SIR (dB)": "30–10",
    "Phase noise σ": "0.01–0.05",
    "IQ amp (dB)": "1–3",
    "IQ phase (deg)": "2–5",
    "Filter tilt norm": "0.1–0.2",
}

EMCR_FORBIDDEN = [
    "CFO norm 0.03 / 0.05 / 0.10",
    "AWGN 5 / 0 dB",
    "Extreme mixed stress presets (test-only)",
]


def read_sweep(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_clean_row(perturb_type: str, strength: float) -> bool:
    clean = CLEAN_STRENGTH.get(perturb_type)
    if clean is None:
        return strength == 0.0
    return abs(strength - clean) < 1e-6


def load_all_sweeps(em_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for label, fname in SWEEP_FILES.items():
        path = em_dir / fname
        if not path.exists():
            continue
        for r in read_sweep(path):
            ptype = r["perturb_type"]
            strength = float(r["strength"])
            file_acc = float(r["file_acc"])
            rows.append(
                {
                    "perturbation_family": label,
                    "perturb_type": ptype,
                    "strength": strength,
                    "file_acc": file_acc,
                    "file_acc_pct": round(file_acc * 100, 2),
                    "window_acc": float(r["window_acc"]),
                    "is_clean": is_clean_row(ptype, strength),
                }
            )
    return rows


def eval_mixed_stress(em_dir: Path) -> list[dict]:
    """Quick eval of MIXED_STRESS_PRESETS."""
    out_csv = em_dir / "mixed_stress_sweep.csv"
    if not out_csv.exists():
        eval_script = ROOT / "experiments/em_robustness_openset/_eval_mixed_quick.py"
        subprocess.run(
            [PYTHON, str(eval_script), "--out-csv", str(out_csv), "--device", "cuda"],
            check=True,
            cwd=str(ROOT),
        )
    if not out_csv.exists():
        return []
    return [
        {
            "perturbation_family": "Mixed stress",
            "perturb_type": f"mixed_{r.get('preset', r['perturb_type'])}",
            "strength": r.get("preset", r["strength"]),
            "file_acc": float(r["file_acc"]),
            "file_acc_pct": round(float(r["file_acc"]) * 100, 2),
            "window_acc": float(r.get("window_acc", 0)),
            "is_clean": False,
        }
        for r in read_sweep(out_csv)
    ]


def summarize_by_perturbation(all_rows: list[dict], global_clean: float = 0.8333333333333334) -> list[dict]:
    summaries: list[dict] = []
    families = sorted({r["perturbation_family"] for r in all_rows})
    for fam in families:
        subset = [r for r in all_rows if r["perturbation_family"] == fam]
        clean_rows = [r for r in subset if r["is_clean"]]
        pert_rows = [r for r in subset if not r["is_clean"]]
        clean_acc = float(clean_rows[0]["file_acc"]) if clean_rows else global_clean
        if not pert_rows:
            pert_rows = subset
        accs = [r["file_acc"] for r in pert_rows]
        avg_robust = float(np.mean(accs))
        min_acc = float(np.min(accs))
        summaries.append(
            {
                "perturbation_family": fam,
                "clean_file_acc_pct": round(clean_acc * 100, 2),
                "avg_robust_acc_pct": round(avg_robust * 100, 2),
                "min_file_acc_pct": round(min_acc * 100, 2),
                "accuracy_drop_pp": round((clean_acc - min_acc) * 100, 2),
                "mean_drop_pp": round((clean_acc - avg_robust) * 100, 2),
                "n_levels": len(pert_rows),
            }
        )
    summaries.sort(key=lambda x: -x["accuracy_drop_pp"])
    return summaries


def load_openset_summary(openset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for seed in range(3):
        csv_path = openset_dir / f"openset_seed{seed}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"split_seed": seed, **r})
    return rows


def openset_aggregate(openset_rows: list[dict]) -> list[dict]:
    if not openset_rows:
        return []
    scorers = sorted({r["scorer"] for r in openset_rows})
    out: list[dict] = []
    for scorer in scorers:
        sub = [r for r in openset_rows if r["scorer"] == scorer]
        out.append(
            {
                "scorer": scorer,
                "auroc_mean": round(float(np.mean([float(r["auroc"]) for r in sub])), 4),
                "auroc_std": round(float(np.std([float(r["auroc"]) for r in sub])), 4),
                "eer_mean": round(float(np.mean([float(r["eer"]) for r in sub])), 4),
                "far_mean": round(float(np.mean([float(r["far"]) for r in sub])), 4),
                "frr_mean": round(float(np.mean([float(r["frr"]) for r in sub])), 4),
                "known_acc_mean_pct": round(float(np.mean([float(r["known_acc"]) for r in sub])) * 100, 2),
            }
        )
    return out


def git_info() -> tuple[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def cuda_check() -> str:
    try:
        out = subprocess.check_output(
            [PYTHON, "-c", "import torch; print(torch.cuda.is_available(), torch.__version__)"],
            text=True,
        ).strip()
        return out
    except Exception as e:
        return f"error: {e}"


def write_run_manifest(em_dir: Path, openset_dir: Path) -> None:
    branch, commit = git_info()
    lines = [
        "# RUN_MANIFEST — Phase B/D EM robustness + open-set full",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Git",
        f"- branch: `{branch}`",
        f"- commit: `{commit}`",
        "",
        "## Environment",
        f"- python: `{PYTHON}`",
        f"- cuda: `{cuda_check()}`",
        "",
        "## Model & data",
        f"- checkpoint: `{CKPT}`",
        "- model: RF-HSTU / F_cross_attn_chirp_plain",
        "- model seed: 0",
        "- windows per file: 256",
        "- eval split: test (Day5)",
        "- manifest: `data/paper/cross_day_day1to5_source_only.csv`",
        "",
        "## GPU allocation (2026-06-28 run)",
        "- GPU1: AWGN curve",
        "- GPU2: CFO curve",
        "- GPU3: narrowband",
        "- GPU4: phase noise + IQ amp",
        "- GPU5: filter drift",
        "- GPU6: open-set full (3 seeds)",
        "",
        "## Perturbation grids",
        "- AWGN SNR (dB): 100, 40, 30, 25, 20, 15, 10, 5, 0",
        "- CFO norm: 0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10",
        "- Narrowband SIR (dB): 30, 20, 10, 5, 0",
        "- Phase noise σ: 0, 0.01, 0.03, 0.05, 0.10",
        "- IQ amp (dB): 0, 1, 3, 5",
        "- Filter tilt norm: 0, 0.1, 0.2, 0.4",
        "",
        "## Open-set",
        "- known devices: 20",
        "- unknown devices: 4",
        "- split seeds: 0, 1, 2",
        f"- output: `{openset_dir.relative_to(ROOT)}`",
        "",
        "## Logs",
    ]
    log_dir = em_dir / "logs"
    if log_dir.exists():
        for p in sorted(log_dir.glob("*.log")):
            if p.stat().st_size < 5_000_000:
                lines.append(f"- `{p.relative_to(ROOT)}`")
    (em_dir / "RUN_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_em_report(em_dir: Path, by_pert: list[dict], all_rows: list[dict]) -> None:
    clean = next((r for r in all_rows if r["perturb_type"] == "awgn_snr_db" and r["is_clean"]), None)
    clean_pct = clean["file_acc_pct"] if clean else "—"
    most = by_pert[0] if by_pert else {}
    lines = [
        "# EM Robustness Full Report",
        "",
        f"**Output:** `{em_dir}`",
        f"**Checkpoint:** `{CKPT}`",
        "",
        "## Clean baseline",
        f"- File-level accuracy (clean-equivalent AWGN≥100 dB): **{clean_pct}%**",
        "",
        "## Perturbation ranking (by max accuracy drop)",
        "",
        "| Family | Clean (%) | Avg robust (%) | Min (%) | Drop (pp) |",
        "|--------|-----------|----------------|---------|-----------|",
    ]
    for s in by_pert:
        lines.append(
            f"| {s['perturbation_family']} | {s['clean_file_acc_pct']} | "
            f"{s['avg_robust_acc_pct']} | {s['min_file_acc_pct']} | {s['accuracy_drop_pp']} |"
        )
    lines.extend(
        [
            "",
            f"**Most destructive:** {most.get('perturbation_family', '—')} "
            f"(drop {most.get('accuracy_drop_pp', '—')} pp).",
            "",
            "## AWGN",
            "- Clean 83.3%; 30 dB → ~70.8%; steep cliff below 25 dB.",
            "",
            "## CFO",
            "- norm≥0.003 collapses to ~4.2%; norm=0.001 still ~20.8%.",
            "",
            "## Narrowband",
            "- Relatively mild: SIR 30–10 dB stays ~83–87.5%; 0 dB → 75%.",
            "",
            "## Phase noise / IQ / Filter",
            "- Phase σ≥0.05 and IQ amp≥3 dB cause strong degradation.",
            "- Filter tilt moderate (0.1–0.2) → 75–67%.",
            "",
            "## Recommended perturbation ranges for EM-CR",
            "",
            "Use **moderate** ranges only for initial EM-CR training:",
            "",
        ]
    )
    for k, v in EMCR_MODERATE.items():
        lines.append(f"- {k}: **{v}**")
    lines.extend(["", "**Forbidden for initial EM-CR training:**", ""])
    for item in EMCR_FORBIDDEN:
        lines.append(f"- {item}")
    lines.append("")
    (em_dir / "EM_ROBUSTNESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_openset_report(em_dir: Path, openset_dir: Path, agg: list[dict], raw: list[dict]) -> None:
    lines = [
        "# Open-Set Authentication Report (clean, full)",
        "",
        f"**Output:** `{openset_dir}`",
        "",
        "## Aggregate (3 split seeds)",
        "",
        "| Scorer | AUROC (mean±std) | EER | FAR | FRR | Known acc (%) |",
        "|--------|------------------|-----|-----|-----|---------------|",
    ]
    for a in agg:
        lines.append(
            f"| {a['scorer']} | {a['auroc_mean']:.3f}±{a['auroc_std']:.3f} | "
            f"{a['eer_mean']:.3f} | {a['far_mean']:.3f} | {a['frr_mean']:.3f} | "
            f"{a['known_acc_mean_pct']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed Proto / Mahalanobis AUROC",
            "",
        ]
    )
    for seed in range(3):
        sub = [r for r in raw if int(r["split_seed"]) == seed and r["scorer"] in ("proto_dist", "mahalanobis", "msp")]
        if sub:
            parts = [f"{r['scorer']}={float(r['auroc']):.3f}" for r in sub]
            lines.append(f"- seed {seed}: " + ", ".join(parts))
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Prototype distance and Mahalanobis outperform MSP/Energy on average.",
            "- Seed 0 Proto/Maha AUROC=1.0 is likely small-sample; seeds 1–2 show 0.86–0.88.",
            "- Threshold selected on validation only (see eval_openset_auth.py).",
            "",
            "## CNN-IQ baseline",
            "CNN-IQ EM baseline pending dedicated checkpoint in EM full script; "
            "current full run uses RF-HSTU (Chapters 3–4 backbone). "
            "Candidate: `outputs/paper_ready_v3/step1_phase7_clean/runs/A_cnn_iq/seed_0/best.pt`.",
        ]
    )
    (em_dir / "OPENSET_AUTH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--em-dir", type=Path, default=EM_FULL)
    p.add_argument("--openset-dir", type=Path, default=OPENSET_DIR)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    em_dir = args.em_dir
    openset_dir = args.openset_dir

    all_rows = load_all_sweeps(em_dir)
    mixed = eval_mixed_stress(em_dir)
    all_rows.extend(mixed)
    awgn_clean = next(
        (r["file_acc"] for r in all_rows if r["perturb_type"] == "awgn_snr_db" and r["is_clean"]),
        0.8333333333333334,
    )
    by_pert = summarize_by_perturbation(all_rows, global_clean=float(awgn_clean))

    # em_robustness_summary.csv — flat all points
    summary_path = em_dir / "em_robustness_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "perturbation_family",
                "perturb_type",
                "strength",
                "file_acc",
                "file_acc_pct",
                "window_acc",
                "is_clean",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    by_path = em_dir / "em_robustness_by_perturbation.csv"
    with by_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_pert[0].keys()))
        w.writeheader()
        w.writerows(by_pert)

    openset_raw = load_openset_summary(openset_dir)
    openset_agg = openset_aggregate(openset_raw)
    open_path = em_dir / "openset_clean_summary.csv"
    with open_path.open("w", newline="", encoding="utf-8") as f:
        if openset_agg:
            w = csv.DictWriter(f, fieldnames=list(openset_agg[0].keys()))
            w.writeheader()
            w.writerows(openset_agg)

    write_run_manifest(em_dir, openset_dir)
    write_em_report(em_dir, by_pert, all_rows)
    write_openset_report(em_dir, openset_dir, openset_agg, openset_raw)

    if args.plot:
        plot_script = ROOT / "experiments/em_robustness_openset/plot_em_robustness_curves.py"
        subprocess.run([PYTHON, str(plot_script), "--em-dir", str(em_dir), "--openset-dir", str(openset_dir)], check=True)

    print(f"Wrote summaries to {em_dir}")


if __name__ == "__main__":
    main()
