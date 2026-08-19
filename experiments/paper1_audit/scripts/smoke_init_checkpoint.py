#!/usr/bin/env python3
"""2B-1: --init-checkpoint full-state smoke. No training. No Day5. No RX2.

Check A  Full load restores classifier.*; strict load_state_dict succeeds.
Check B  --pretrained leaves classifier far from ckpt (documents F0 forbid).
Check C  After init load, Day4 window_acc within 0.05 pp of frozen C' seed 2.

Does not open F0 train. Source: PHASE2B1_IDENTITY_ANCHOR_LOCK.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.train_utils import load_checkpoint, save_checkpoint

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
DATA_ROOT = Path("/data1/hcc/llm4RF")
PY = Path("/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python")
OUT_ROOT = KEYAN / "experiments/paper1_audit/results/matched_seed0"
OUT_DIR = OUT_ROOT / "init_checkpoint_smoke"
SEED = 2
CPRIME = OUT_ROOT / "runs" / "C_full_ratio" / f"seed_{SEED}" / "best.pt"
CPRIME_METRICS = OUT_ROOT / "eval_val" / "C_full_ratio" / f"seed_{SEED}" / "metrics.json"
TOL_PP = 0.05
MANIFEST = KEYAN / "data/paper/cross_day_day1to5_source_only.csv"


def build_cprime(device: torch.device) -> DeviceClassifier:
    embedder = RFPatchEmbedder(
        window_size=8192,
        patch_size=256,
        sample_rate=1_000_000.0,
        lora_bandwidth=125_000.0,
        spreading_factor=7,
        use_oob=True,
        oob_fusion_type="cross_attn_oob",
        use_oob_cross_attention=True,
        patch_embed_type="cnn_stem",
        dim=64,
        cnn_stem_dim=32,
        cnn_stem_kernels="7,5",
        fft_norm="log_zscore",
        oob_norm="ratio",
        fft_source="full",
    )
    return DeviceClassifier(
        embedder,
        num_classes=24,
        dim=64,
        depth=2,
        dropout=0.0,
        use_chirp_embedding=True,
        oob_num_heads=4,
    ).to(device)


def tensor_rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((a - b).reshape(-1))
    den = torch.linalg.vector_norm(a.reshape(-1)).clamp_min(1e-8)
    return float((num / den).item())


def weight_checks() -> dict:
    if not CPRIME.is_file():
        raise SystemExit(f"missing C' ckpt: {CPRIME}")
    device = torch.device("cpu")
    ckpt = load_checkpoint(CPRIME, map_location=device)
    state = ckpt["model"]
    cls_keys = [k for k in state if k.startswith("classifier.")]
    if len(cls_keys) < 1:
        raise SystemExit("C' best.pt has no classifier.* keys")

    full = build_cprime(device)
    missing, unexpected = full.load_state_dict(state, strict=True)
    full_cls = full.classifier.weight.detach().clone()
    ckpt_cls = state["classifier.weight"].detach().clone()
    full_rel = tensor_rel_l2(ckpt_cls, full_cls)

    enc_only = build_cprime(device)
    torch.manual_seed(12345)
    for p in enc_only.classifier.parameters():
        p.data.normal_(0, 0.02)
    before = enc_only.classifier.weight.detach().clone()
    encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    enc_only.encoder.load_state_dict(encoder_state, strict=False)
    after = enc_only.classifier.weight.detach().clone()
    pretrained_moved = tensor_rel_l2(before, after)
    pretrained_vs_ckpt = tensor_rel_l2(ckpt_cls, after)

    payload = {
        "seed": SEED,
        "cprime_ckpt": str(CPRIME),
        "classifier_keys": cls_keys,
        "full_load": {
            "missing": list(missing),
            "unexpected": list(unexpected),
            "classifier_rel_l2_vs_ckpt": round(full_rel, 12),
            "ok": full_rel <= 1e-8 and len(missing) == 0 and len(unexpected) == 0,
        },
        "pretrained_encoder_only": {
            "classifier_changed_by_load": round(pretrained_moved, 12),
            "classifier_rel_l2_vs_ckpt": round(pretrained_vs_ckpt, 6),
            "ok_documents_forbid": pretrained_moved <= 1e-8 and pretrained_vs_ckpt > 0.1,
        },
    }
    if not payload["full_load"]["ok"]:
        raise SystemExit(f"full load failed: {payload['full_load']}")
    if not payload["pretrained_encoder_only"]["ok_documents_forbid"]:
        raise SystemExit(f"pretrained contrast failed: {payload['pretrained_encoder_only']}")
    return payload


def day4_eval_smoke(gpu: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_ckpt = OUT_DIR / f"seed_{SEED}_init_reload.pt"
    device = torch.device("cuda" if gpu else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model = build_cprime(device)
    ckpt = load_checkpoint(CPRIME, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    # Mimic finetune checkpoint payload: top-level num_classes is required
    # (evaluate.py defaults missing num_classes to 25).
    fake_args = {
        "manifest": str(MANIFEST),
        "root": str(DATA_ROOT),
        "model_type": "rf_hstu",
        "input_norm": "iq_rms",
        "fft_norm": "log_zscore",
        "oob_norm": "ratio",
        "fft_source": "full",
        "paired_view": "off",
        "window_size": 8192,
        "patch_size": 256,
        "samples_per_file": 256,
        "eval_samples_per_file": 256,
        "sample_rate": 1_000_000.0,
        "lora_bandwidth": 125_000.0,
        "spreading_factor": 7,
        "use_chirp_embedding": True,
        "no_oob": False,
        "use_oob_cross_attention": True,
        "oob_fusion_type": "cross_attn_oob",
        "oob_num_heads": 4,
        "patch_embed_type": "cnn_stem",
        "cnn_stem_dim": 32,
        "cnn_stem_kernels": "7,5",
        "dim": 64,
        "depth": 2,
        "dropout": 0.0,
        "batch_size": 128,
        "seed": SEED,
        "train_split": "train",
        "val_split": "val",
        "eval_split": "val",
        "checkpoint_metric": "acc",
        "loss_type": "ce",
        "label_smoothing": 0.0,
        "weight_decay": 5e-4,
        "lr": 3e-3,
        "epochs": 80,
        "device": "cuda",
        "num_workers": 4,
        "num_classes": 24,
        "init_checkpoint": str(CPRIME),
    }
    from types import SimpleNamespace

    save_checkpoint(
        smoke_ckpt,
        model,
        SimpleNamespace(**fake_args),
        extra={"num_classes": 24, "smoke": "init_checkpoint"},
    )

    eval_dir = OUT_DIR / f"eval_val_seed_{SEED}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    py = str(PY if PY.is_file() else sys.executable)
    cmd = [
        py,
        str(KEYAN / "scripts/evaluate.py"),
        "--manifest",
        str(MANIFEST),
        "--root",
        str(DATA_ROOT),
        "--batch-size",
        "128",
        "--samples-per-file",
        "256",
        "--eval-samples-per-file",
        "256",
        "--dim",
        "64",
        "--depth",
        "2",
        "--device",
        "cuda",
        "--train-split",
        "train",
        "--val-split",
        "val",
        "--eval-split",
        "val",
        "--input-norm",
        "iq_rms",
        "--fft-norm",
        "log_zscore",
        "--fft-source",
        "full",
        "--paired-view",
        "off",
        "--window-size",
        "8192",
        "--num-workers",
        "4",
        "--seed",
        str(SEED),
        "--model-type",
        "rf_hstu",
        "--patch-embed-type",
        "cnn_stem",
        "--cnn-stem-dim",
        "32",
        "--use-chirp-embedding",
        "--oob-fusion-type",
        "cross_attn_oob",
        "--use-oob-cross-attention",
        "--oob-norm",
        "ratio",
        "--mode",
        "classifier",
        "--file-vote-mode",
        "mean_logits",
        "--checkpoint",
        str(smoke_ckpt),
        "--out-dir",
        str(eval_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{KEYAN / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    log = OUT_DIR / f"eval_seed{SEED}.log"
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(KEYAN), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"evaluate failed rc={rc}; see {log}")

    frozen = json.loads(CPRIME_METRICS.read_text())
    smoke = json.loads((eval_dir / "metrics.json").read_text())
    if smoke.get("day5_used") is True or smoke["num_files"] != 24:
        raise SystemExit("split/files check failed")
    d_pp = abs(100.0 * smoke["window_acc"] - 100.0 * frozen["window_acc"])
    return {
        "frozen_window_pct": round(100.0 * frozen["window_acc"], 4),
        "smoke_window_pct": round(100.0 * smoke["window_acc"], 4),
        "abs_pp": round(d_pp, 4),
        "tol_pp": TOL_PP,
        "ok": d_pp <= TOL_PP,
        "smoke_ckpt": str(smoke_ckpt),
        "eval_dir": str(eval_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=os.environ.get("GPU", "5"))
    parser.add_argument("--weights-only", action="store_true", help="Skip Day4 GPU eval")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = weight_checks()
    payload = {
        "day5_used": False,
        "rx2_used": False,
        "training": False,
        "seed": SEED,
        "weights": weights,
        "day4_eval": None,
        "verdict": "WEIGHTS_PASS",
    }
    if not args.weights_only:
        day4 = day4_eval_smoke(str(args.gpu))
        payload["day4_eval"] = day4
        payload["verdict"] = "SMOKE_PASS" if day4["ok"] else "SMOKE_FAIL"
        if not day4["ok"]:
            out = OUT_DIR / "init_checkpoint_smoke.json"
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2))
            raise SystemExit(2)

    out = OUT_DIR / "init_checkpoint_smoke.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("wrote", out)
    print("VERDICT", payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
