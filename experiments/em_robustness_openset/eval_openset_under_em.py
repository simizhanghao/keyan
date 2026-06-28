#!/usr/bin/env python3
"""Open-set authentication under EM perturbations (Prototype + Mahalanobis)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_openset_auth import (  # noqa: E402
    build_prototypes,
    load_unknown_map,
    mahalanobis_min_dist,
    openset_labels,
    pick_threshold,
)
from common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    compute_auroc,
    compute_eer,
    collect_file_features,
    make_eval_loader,
    populate_args_from_ckpt,
    resolve_device,
    save_json,
)
from evaluate import build_model, load_checkpoint  # noqa: E402
from rfhstu.em_perturbations import EmPerturbConfig, MIXED_STRESS_PRESETS, apply_em_perturbation  # noqa: E402

# condition_name -> EmPerturbConfig or preset key
CONDITIONS: dict[str, EmPerturbConfig | str] = {
    "clean": None,
    "awgn_30db": EmPerturbConfig(awgn_snr_db=30.0),
    "awgn_20db": EmPerturbConfig(awgn_snr_db=20.0),
    "cfo_0.001": EmPerturbConfig(cfo_norm=0.001),
    "cfo_0.003": EmPerturbConfig(cfo_norm=0.003),
    "nbi_10db": EmPerturbConfig(narrowband_sir_db=10.0),
    "phase_0.03": EmPerturbConfig(phase_noise_std=0.03),
    "iq_3db_5deg": EmPerturbConfig(iq_amp_db=3.0, iq_phase_deg=5.0),
    "filter_0.2": EmPerturbConfig(filter_tilt_norm=0.2),
    "mixed_awgn_cfo": "awgn_cfo",
}

MAIN_SCORERS = ("proto_dist", "mahalanobis")


def resolve_perturb_cfg(spec: EmPerturbConfig | str | None) -> EmPerturbConfig | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        return MIXED_STRESS_PRESETS[spec]
    return spec


def eval_condition(
    model,
    args,
    ckpt_args,
    device,
    unk_map,
    condition: str,
    perturb_cfg: EmPerturbConfig | None,
    protos: np.ndarray,
    cov_inv: np.ndarray,
) -> list[dict]:
    val_loader = make_eval_loader(args, args.val_split)
    test_loader = make_eval_loader(args, args.test_split)
    val_rows = collect_file_features(model, val_loader, device, args, ckpt_args, perturb_cfg)
    test_rows = collect_file_features(model, test_loader, device, args, ckpt_args, perturb_cfg)
    y_val = openset_labels(val_rows, unk_map)
    y_test = openset_labels(test_rows, unk_map)

    test_emb = np.stack([r["embedding"] for r in test_rows])
    val_emb = np.stack([r["embedding"] for r in val_rows])
    proto_min = np.linalg.norm(test_emb[:, None, :] - protos[None, :, :], axis=2).min(axis=1)
    maha = mahalanobis_min_dist(test_emb, protos, cov_inv)
    val_proto = np.linalg.norm(val_emb[:, None, :] - protos[None, :, :], axis=2).min(axis=1)
    val_maha = mahalanobis_min_dist(val_emb, protos, cov_inv)

    scorers = {
        "proto_dist": (-proto_min, -val_proto),
        "mahalanobis": (-maha, -val_maha),
        "msp": (np.array([r["msp"] for r in test_rows]), np.array([r["msp"] for r in val_rows])),
        "energy": (-np.array([r["energy"] for r in test_rows]), -np.array([r["energy"] for r in val_rows])),
    }

    rows = []
    for name, (ts, vs) in scorers.items():
        thr = pick_threshold(y_val, vs)
        pred_known = ts >= thr
        far = float(np.mean(pred_known[y_test == 0])) if np.any(y_test == 0) else 0.0
        frr = float(np.mean(~pred_known[y_test == 1])) if np.any(y_test == 1) else 0.0
        known_idx = [i for i, r in enumerate(test_rows) if y_test[i] == 1]
        known_acc = float(np.mean([test_rows[i]["correct"] for i in known_idx])) if known_idx else float("nan")
        rows.append(
            {
                "split_seed": args.split_seed,
                "condition": condition,
                "scorer": name,
                "auroc": compute_auroc(y_test, ts),
                "eer": compute_eer(y_test, ts),
                "far": far,
                "frr": frr,
                "known_acc": known_acc,
                "threshold": thr,
            }
        )
    return rows


def plot_results(all_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    conds = sorted({r["condition"] for r in all_rows})
    x = np.arange(len(conds))

    for metric, fname in [("auroc", "fig_auroc_under_em"), ("eer", "fig_eer_under_em"), ("known_acc", "fig_known_acc_under_em")]:
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        width = 0.35
        for i, scorer in enumerate(MAIN_SCORERS):
            vals = []
            for c in conds:
                sub = [r for r in all_rows if r["condition"] == c and r["scorer"] == scorer]
                vals.append(float(np.mean([r[metric] for r in sub])) if sub else 0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=scorer.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=35, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--openset-split-dir", default="experiments/em_robustness_openset/results/openset_splits")
    p.add_argument("--split-seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples-per-file", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    ckpt_args = populate_args_from_ckpt(args, ckpt)
    model = build_model(args, ckpt, device)

    all_rows: list[dict] = []
    for seed in args.split_seeds:
        args.split_seed = seed
        manifest = f"experiments/em_robustness_openset/results/openset_splits/openset_split_seed{seed}.csv"
        args.manifest = manifest
        args.openset_manifest = manifest
        unk_map = load_unknown_map(ROOT / manifest, Path(args.root))

        train_loader = make_eval_loader(args, "train")
        train_rows = collect_file_features(model, train_loader, device, args, ckpt_args, None)
        train_emb = np.stack([r["embedding"] for r in train_rows])
        train_y = np.array([r["label"] for r in train_rows])
        num_classes = int(ckpt.get("num_classes", ckpt_args.get("num_classes", 24)))
        protos = build_prototypes(train_emb, train_y, num_classes)
        cov = np.cov(train_emb, rowvar=False) + np.eye(train_emb.shape[1]) * 1e-4
        cov_inv = np.linalg.inv(cov)

        for cond_name, spec in CONDITIONS.items():
            pcfg = resolve_perturb_cfg(spec)
            print(f"seed={seed} {cond_name}")
            all_rows.extend(
                eval_condition(model, args, ckpt_args, device, unk_map, cond_name, pcfg, protos, cov_inv)
            )

    csv_path = out_dir / "openset_under_em_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    try:
        plot_results(all_rows, out_dir)
    except Exception as e:
        print(f"plot skipped: {e}")

    save_json(out_dir / "meta.json", {"checkpoint": args.checkpoint, "conditions": list(CONDITIONS.keys())})
    write_report(all_rows, out_dir)
    print(f"Wrote {csv_path}")


def write_report(rows: list[dict], out_dir: Path) -> None:
    lines = ["# Open-Set Under EM Report", ""]
    clean_proto = [r for r in rows if r["condition"] == "clean" and r["scorer"] == "proto_dist"]
    if clean_proto:
        au = float(np.mean([r["auroc"] for r in clean_proto]))
        lines.append(f"- Clean Prototype AUROC (mean seeds): **{au:.3f}**")
    for cond in CONDITIONS:
        if cond == "clean":
            continue
        sub = [r for r in rows if r["condition"] == cond and r["scorer"] == "proto_dist"]
        if sub:
            au = float(np.mean([r["auroc"] for r in sub]))
            ka = float(np.mean([r["known_acc"] for r in sub]))
            lines.append(f"- {cond}: Proto AUROC={au:.3f}, known_acc={ka:.1%}")
    lines.append("")
    lines.append("## Main scorer recommendation: **Prototype distance** (slightly more stable than Mahalanobis in clean full).")
    (out_dir / "OPENSET_UNDER_EM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
