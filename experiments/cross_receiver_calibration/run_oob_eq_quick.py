#!/usr/bin/env python3
"""OOB representation equalization quick evaluation."""
from __future__ import annotations

import argparse
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CAL_ROOT))

from lib.oob_equalization import apply_equalization, build_representation, fit_stats
from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE, ROLE_SUPPORT, sample_k_support_indices

from evaluate import build_model, load_checkpoint
from run_rcpa_prototypes import (
    build_class_prototypes,
    eval_prototype_file_acc,
    file_level_vote,
    macro_f1,
)


def build_eval_args(ckpt: dict, batch_size: int) -> argparse.Namespace:
    ckpt_args = ckpt.get("args", {})
    ns = argparse.Namespace(**{k: v for k, v in ckpt_args.items()})
    ns.model_type = ckpt.get("model_type", ckpt_args.get("model_type", "rf_hstu"))
    ns.cnn_input_type = ckpt_args.get("cnn_input_type", "iq")
    ns.dim = ckpt_args.get("dim", 64)
    ns.depth = ckpt_args.get("depth", 2)
    ns.dropout = ckpt_args.get("dropout", 0.1)
    ns.window_size = ckpt_args.get("window_size", 8192)
    ns.patch_size = ckpt_args.get("patch_size", 256)
    ns.sample_rate = ckpt_args.get("sample_rate", 1e6)
    ns.lora_bandwidth = ckpt_args.get("lora_bandwidth", 125e3)
    ns.spreading_factor = ckpt_args.get("spreading_factor", 7)
    ns.patch_embed_type = ckpt_args.get("patch_embed_type", "cnn_stem")
    ns.cnn_stem_dim = ckpt_args.get("cnn_stem_dim", 32)
    ns.cnn_stem_kernels = ckpt_args.get("cnn_stem_kernels", [7, 5, 3])
    ns.oob_fusion_type = ckpt_args.get("oob_fusion_type", "cross_attn_oob")
    ns.use_oob_cross_attention = ckpt_args.get("use_oob_cross_attention", True)
    ns.use_chirp_embedding = ckpt_args.get("use_chirp_embedding", True)
    ns.input_norm = ckpt_args.get("input_norm", "iq_rms")
    ns.fft_norm = ckpt_args.get("fft_norm", "log_zscore")
    ns.oob_norm = ckpt_args.get("oob_norm", "ratio")
    ns.no_oob = ckpt_args.get("no_oob", False)
    ns.cnn_hidden_dim = ckpt_args.get("cnn_hidden_dim", 128)
    ns.cnn_dropout = ckpt_args.get("cnn_dropout", 0.3)
    ns.oob_num_heads = ckpt_args.get("oob_num_heads", 4)
    ns.use_multiscale = ckpt_args.get("use_multiscale", False)
    ns.multiscale_ratios = ckpt_args.get("multiscale_ratios", [1, 2, 4])
    ns.multiscale_fusion_type = ckpt_args.get("multiscale_fusion_type", "concat")
    ns.use_cfo_feature = ckpt_args.get("use_cfo_feature", False)
    ns.batch_size = batch_size
    return ns


EQ_METHODS = ["none", "mean_shift", "std_alignment", "coral"]
REPRS = ["oob_only", "main_only", "fused", "late_fusion"]
EVAL_METHODS = ["source_classifier", "RCPA-T", "oob_eq_only", "oob_eq_RCPA-T"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOB representation equalization quick mode")
    p.add_argument("--split-csv", required=True)
    p.add_argument("--embeddings-npz", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--direction", default="rx1_to_rx2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--shot-ks", nargs="+", type=int, default=[0, 1, 3, 5])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-rx", type=int, default=1)
    p.add_argument("--target-rx", type=int, default=2)
    return p.parse_args()


def load_arrays(npz_path: Path, split_rows: list[dict]):
    data = np.load(npz_path, allow_pickle=True)
    row_indices = data["row_indices"]
    roles = np.array([split_rows[int(i)]["role"] for i in row_indices], dtype=object)
    return {
        "main": data["main"],
        "oob": data["oob"],
        "fused": data["fused"],
        "logits": data["logits"],
        "labels": data["labels"],
        "device_ids": data["device_ids"],
        "window_indices": data["window_indices"],
        "roles": roles,
    }


def role_mask(roles: np.ndarray, role: str) -> np.ndarray:
    return roles == role


def safe_cv(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(5, int(counts.min())))


def run_probe(x: np.ndarray, y: np.ndarray) -> float:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    cv = safe_cv(y)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, xs, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0), scoring="accuracy")
    return float(scores.mean())


def file_centroids(
    z: np.ndarray,
    device_ids: np.ndarray,
    rx_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return file-level centroids keyed by (device_id, receiver)."""
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in range(len(device_ids)):
        key = (int(device_ids[i]), int(rx_labels[i]))
        groups.setdefault(key, []).append(z[i])
    keys_sorted = sorted(groups.keys())
    cents = np.stack([np.mean(groups[k], axis=0) for k in keys_sorted], axis=0)
    rxs = np.array([k[1] for k in keys_sorted], dtype=np.int64)
    return cents, rxs


def probe_metrics(
    z_main: np.ndarray,
    z_oob: np.ndarray,
    z_fused: np.ndarray,
    roles: np.ndarray,
    labels: np.ndarray,
    device_ids: np.ndarray,
    repr_name: str,
    source_rx: int,
    target_rx: int,
    oob_eq: np.ndarray | None = None,
) -> tuple[float, float]:
    z = build_representation(z_main, z_oob, z_fused, repr_name, oob_eq=oob_eq)
    rx_all = np.where(roles == ROLE_SOURCE, source_rx, target_rx)
    # receiver probe: file-level across source + target windows
    z_file, rx_file = file_centroids(z, device_ids, rx_all)
    rx_probe = run_probe(z_file, rx_file)
    # device probe: window-level on query only (evaluation windows)
    qmask = roles == ROLE_QUERY
    dev_probe = run_probe(z[qmask], labels[qmask])
    return rx_probe, dev_probe


def transform_target(
    z: np.ndarray,
    roles: np.ndarray,
    method: str,
    mu_src: np.ndarray,
    std_src: np.ndarray,
    cov_src: np.ndarray,
    mu_tgt: np.ndarray,
    std_tgt: np.ndarray,
    cov_tgt: np.ndarray,
) -> np.ndarray:
    out = z.copy()
    tmask = roles != ROLE_SOURCE
    out[tmask] = apply_equalization(
        z[tmask], method, mu_src, std_src, cov_src, mu_tgt, std_tgt, cov_tgt,
    )
    return out


def classifier_acc_from_z(z_q: np.ndarray, labels_q: np.ndarray, device_q: np.ndarray, classifier: torch.nn.Module) -> tuple[float, float]:
    with torch.no_grad():
        logits = classifier(torch.from_numpy(z_q).float())
        probs = F.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    file_labels, file_preds = file_level_vote(device_q, labels_q, preds, probs)
    acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    return acc, macro_f1(file_labels, file_preds, 24)


def eval_rcpa_t(
    z_src: np.ndarray,
    y_src: np.ndarray,
    z_sup: np.ndarray,
    y_sup: np.ndarray,
    z_q: np.ndarray,
    y_q: np.ndarray,
    device_q: np.ndarray,
    k: int,
    split_seed: int,
    win_sup: np.ndarray,
) -> tuple[float, float]:
    src_protos, src_labels = build_class_prototypes(z_src, y_src, 24)
    support_indices = set(sample_k_support_indices(k, split_seed))
    if k > 0:
        mask = np.array([int(w) in support_indices for w in win_sup], dtype=bool)
        tgt_protos, tgt_labels = build_class_prototypes(z_sup[mask], y_sup[mask], 24)
    else:
        return 0.0, 0.0
    qry = {"z": z_q, "y": y_q, "device": device_q}
    return eval_prototype_file_acc(qry, tgt_protos, tgt_labels, "cosine", 24)


def eval_source_proto(
    z_src: np.ndarray,
    y_src: np.ndarray,
    z_q: np.ndarray,
    y_q: np.ndarray,
    device_q: np.ndarray,
) -> tuple[float, float]:
    protos, plabels = build_class_prototypes(z_src, y_src, 24)
    qry = {"z": z_q, "y": y_q, "device": device_q}
    return eval_prototype_file_acc(qry, protos, plabels, "cosine", 24)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.split_csv).open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    arrs = load_arrays(Path(args.embeddings_npz), split_rows)
    roles = arrs["roles"]

    src_mask = role_mask(roles, ROLE_SOURCE)
    cal_mask = role_mask(roles, ROLE_CALIBRATION)
    sup_mask = role_mask(roles, ROLE_SUPPORT)
    qry_mask = role_mask(roles, ROLE_QUERY)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    eval_args = build_eval_args(ckpt, 64)
    model = build_model(eval_args, ckpt, "cpu")
    classifier = model.classifier
    classifier.eval()

    summary_rows: list[dict] = []
    probe_rows: list[dict] = []

    # Baselines without eq (once per repr for classifier/RCPA-T reference)
    for repr_name in REPRS:
        z_main, z_oob, z_fused = arrs["main"], arrs["oob"], arrs["fused"]
        z = build_representation(z_main, z_oob, z_fused, repr_name)

        rx_before, dev_before = probe_metrics(
            z_main, z_oob, z_fused, roles, arrs["labels"], arrs["device_ids"],
            repr_name, args.source_rx, args.target_rx,
        )
        probe_rows.append({
            "repr": repr_name, "eq_method": "none", "phase": "before",
            "receiver_probe_acc": rx_before, "device_probe_acc": dev_before,
        })

        z_src, y_src = z[src_mask], arrs["labels"][src_mask]
        z_sup, y_sup = z[sup_mask], arrs["labels"][sup_mask]
        z_q, y_q = z[qry_mask], arrs["labels"][qry_mask]
        dev_q = arrs["device_ids"][qry_mask]
        win_sup = arrs["window_indices"][sup_mask]

        # source classifier baseline (fused repr only meaningful for original classifier)
        if repr_name == "fused":
            cls_acc, cls_f1 = classifier_acc_from_z(z_q, y_q, dev_q, classifier)
            summary_rows.append({
                "eq_method": "none", "repr": repr_name, "eval_method": "source_classifier",
                "shot_k": -1, "file_acc": cls_acc, "macro_f1": cls_f1,
            })

        for k in args.shot_ks:
            if k == 0:
                acc, f1 = eval_source_proto(z_src, y_src, z_q, y_q, dev_q)
                summary_rows.append({
                    "eq_method": "none", "repr": repr_name, "eval_method": "RCPA-S_ref",
                    "shot_k": k, "file_acc": acc, "macro_f1": f1,
                })
            else:
                acc, f1 = eval_rcpa_t(
                    z_src, y_src, z_sup, y_sup, z_q, y_q, dev_q, k, args.split_seed, win_sup,
                )
                summary_rows.append({
                    "eq_method": "none", "repr": repr_name, "eval_method": "RCPA-T",
                    "shot_k": k, "file_acc": acc, "macro_f1": f1,
                })

    # OOB-Eq methods
    for eq_method in EQ_METHODS:
        if eq_method == "none":
            continue
        for repr_name in REPRS:
            z_main = arrs["main"].copy()
            z_oob = arrs["oob"].copy()
            z_fused = arrs["fused"].copy()

            # stats on the path being equalized
            path_key = {"oob_only": "oob", "main_only": "main", "fused": "fused", "late_fusion": "oob"}[repr_name]
            z_path = {"main": z_main, "oob": z_oob, "fused": z_fused}[path_key]
            mu_src, std_src, cov_src = fit_stats(z_path[src_mask])
            mu_tgt, std_tgt, cov_tgt = fit_stats(z_path[cal_mask])

            if path_key == "main":
                z_main = transform_target(z_main, roles, eq_method, mu_src, std_src, cov_src, mu_tgt, std_tgt, cov_tgt)
            elif path_key == "oob":
                z_oob = transform_target(z_oob, roles, eq_method, mu_src, std_src, cov_src, mu_tgt, std_tgt, cov_tgt)
            else:
                z_fused = transform_target(z_fused, roles, eq_method, mu_src, std_src, cov_src, mu_tgt, std_tgt, cov_tgt)

            oob_eq = z_oob if repr_name in ("oob_only", "late_fusion") else None
            rx_after, dev_after = probe_metrics(
                z_main, z_oob, z_fused, roles, arrs["labels"], arrs["device_ids"],
                repr_name, args.source_rx, args.target_rx, oob_eq=oob_eq,
            )
            probe_rows.append({
                "repr": repr_name, "eq_method": eq_method, "phase": "after",
                "receiver_probe_acc": rx_after, "device_probe_acc": dev_after,
            })

            z = build_representation(z_main, z_oob, z_fused, repr_name, oob_eq=oob_eq)
            z_src, y_src = z[src_mask], arrs["labels"][src_mask]
            z_sup, y_sup = z[sup_mask], arrs["labels"][sup_mask]
            z_q, y_q = z[qry_mask], arrs["labels"][qry_mask]
            dev_q = arrs["device_ids"][qry_mask]
            win_sup = arrs["window_indices"][sup_mask]

            if repr_name == "fused":
                acc, f1 = classifier_acc_from_z(z_q, y_q, dev_q, classifier)
                summary_rows.append({
                    "eq_method": eq_method, "repr": repr_name, "eval_method": "oob_eq_only",
                    "shot_k": -1, "file_acc": acc, "macro_f1": f1,
                })

            for k in args.shot_ks:
                acc0, f10 = eval_source_proto(z_src, y_src, z_q, y_q, dev_q)
                summary_rows.append({
                    "eq_method": eq_method, "repr": repr_name, "eval_method": "oob_eq_only",
                    "shot_k": k, "file_acc": acc0, "macro_f1": f10,
                })
                if k > 0:
                    acc, f1 = eval_rcpa_t(
                        z_src, y_src, z_sup, y_sup, z_q, y_q, dev_q, k, args.split_seed, win_sup,
                    )
                    summary_rows.append({
                        "eq_method": eq_method, "repr": repr_name, "eval_method": "oob_eq_RCPA-T",
                        "shot_k": k, "file_acc": acc, "macro_f1": f1,
                    })

    meta = {
        "direction": args.direction, "seed": args.seed, "split_seed": args.split_seed,
        "shot_ks": args.shot_ks, "note": "OOB representation equalization v1 (embedding-level)",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sfields = ["eq_method", "repr", "eval_method", "shot_k", "file_acc", "macro_f1"]
    with (out_dir / "summary_oob_eq_quick.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)

    pfields = ["repr", "eq_method", "phase", "receiver_probe_acc", "device_probe_acc"]
    with (out_dir / "probe_before_after.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        w.writerows(probe_rows)

    # shot curve for best repr (fused) RCPA-T vs oob_eq_RCPA-T
    curve_fields = ["eq_method", "eval_method", "shot_k", "file_acc"]
    curve_rows = [r for r in summary_rows if r["repr"] == "fused" and r["eval_method"] in ("RCPA-T", "oob_eq_RCPA-T")]
    with (out_dir / "oob_eq_shot_curve_quick.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=curve_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(curve_rows)

    print(json.dumps({"summary_rows": len(summary_rows), "probe_rows": len(probe_rows)}, indent=2))


if __name__ == "__main__":
    main()
