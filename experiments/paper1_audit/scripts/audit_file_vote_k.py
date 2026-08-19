#!/usr/bin/env python3
"""Paper 1 revision-reserve 1D: file-vote K sensitivity on frozen 1C ckpts.

No training. Day4 val only. Does not overwrite eval_val/{A,B,C'}/.
Does not open Day5, S1 5-seed, RX2, or per-device (next beat).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYAN = Path("/data1/hcc/llm4RF/new_phase")
DATA_ROOT = Path("/data1/hcc/llm4RF")
PY = Path("/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python")
MANIFEST = KEYAN / "data/paper/cross_day_day1to5_source_only.csv"
ROOT = KEYAN / "experiments/paper1_audit/results/matched_seed0"
LOGIT_ROOT = ROOT / "eval_val_logits"
FROZEN_EVAL = ROOT / "eval_val"
LOG_DIR = ROOT / "logs"

MODELS = ["A_cnn_iq", "B_exact_main_no_oob", "C_full_ratio"]
MODEL_LABEL = {
    "A_cnn_iq": "A CNN",
    "B_exact_main_no_oob": "B Main",
    "C_full_ratio": "C' Full ratio",
}
SEEDS = [0, 1, 2, 3, 4]
KS = [8, 16, 32, 64, 128, 256]
VOTE_MODES = ["mean_logits", "mean_prob", "majority"]
PRIMARY_MODE = "mean_logits"
SANITY_PP = 0.05
SMOOTH_DIP_PP = 2.0
STEM = [
    "--model-type",
    "rf_hstu",
    "--patch-embed-type",
    "cnn_stem",
    "--cnn-stem-dim",
    "32",
    "--use-chirp-embedding",
]
MODEL_FLAGS = {
    "A_cnn_iq": ["--model-type", "osu_cnn", "--cnn-input-type", "iq"],
    "B_exact_main_no_oob": [
        *STEM,
        "--no-oob",
        "--oob-fusion-type",
        "no_oob",
        "--oob-norm",
        "none",
    ],
    "C_full_ratio": [
        *STEM,
        "--oob-fusion-type",
        "cross_attn_oob",
        "--use-oob-cross-attention",
        "--oob-norm",
        "ratio",
    ],
}


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def vote_pred(logits: np.ndarray, mode: str) -> int:
    if mode == "mean_logits":
        return int(logits.mean(axis=0).argmax())
    if mode == "mean_prob":
        return int(softmax(logits).mean(axis=0).argmax())
    if mode == "majority":
        preds = logits.argmax(axis=-1)
        counts = np.bincount(preds, minlength=int(logits.shape[-1]))
        return int(counts.argmax())
    raise ValueError(mode)


def group_rows(payload: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    n = int(payload["logits"].shape[0])
    for i in range(n):
        grouped[str(payload["file_path"][i])].append(
            {
                "window_index": int(payload["window_index"][i]),
                "label": int(payload["label"][i]),
                "logits": payload["logits"][i],
            }
        )
    return grouped


def acc_at_k(grouped: dict[str, list[dict]], k: int, mode: str) -> tuple[float, float, int]:
    file_correct = 0
    win_correct = 0
    n_win = 0
    n_file = 0
    for rows in grouped.values():
        rows = sorted(rows, key=lambda r: r["window_index"])[:k]
        if not rows:
            continue
        logits = np.stack([r["logits"] for r in rows], axis=0)
        label = int(rows[0]["label"])
        file_correct += int(vote_pred(logits, mode) == label)
        n_file += 1
        win_correct += sum(int(int(r["logits"].argmax()) == label) for r in rows)
        n_win += len(rows)
    return file_correct / max(1, n_file), win_correct / max(1, n_win), n_file


def load_npz(model: str, seed: int) -> dict[str, np.ndarray]:
    path = LOGIT_ROOT / model / f"seed_{seed}" / "window_logits.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def frozen_file_acc(model: str, seed: int) -> float:
    path = FROZEN_EVAL / model / f"seed_{seed}" / "metrics.json"
    return float(json.loads(path.read_text())["file_acc"])


def mean_std(values: list[float]) -> str:
    if not values:
        return "?"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def smoke() -> int:
    # Unsorted window_index must be reordered before taking first K.
    payload = {
        "logits": np.array(
            [
                [0.0, 8.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [4.0, 1.0, 0.0],
                [0.0, 6.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 0.0, 4.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        "label": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        "window_index": np.array([2, 0, 3, 1, 1, 0, 2, 3], dtype=np.int64),
        "file_path": np.array(["f0", "f0", "f0", "f0", "f1", "f1", "f1", "f1"]),
    }
    grouped = group_rows(payload)
    assert [int(r["window_index"]) for r in sorted(grouped["f0"], key=lambda r: r["window_index"])] == [0, 1, 2, 3]
    file_acc, _, n_file = acc_at_k(grouped, 2, "mean_logits")
    assert n_file == 2
    # f0 first-2 = class 0,0 label 0; f1 first-2 = class 1,1 label 1
    assert abs(file_acc - 1.0) < 1e-6
    maj_k4, _, _ = acc_at_k(grouped, 4, "majority")
    mean_k4, _, _ = acc_at_k(grouped, 4, "mean_logits")
    # f0 majority 2/2 split → class 0; f1 majority class 1
    assert abs(maj_k4 - 1.0) < 1e-6
    assert mean_k4 >= 0.5
    print("SMOKE_PASS")
    return 0


def dump_all(gpus: list[str], num_workers: int) -> None:
    jobs = [(model, seed) for seed in SEEDS for model in MODELS]
    pending = [job for job in jobs if not (LOGIT_ROOT / job[0] / f"seed_{job[1]}" / "window_logits.npz").is_file()]
    if not pending:
        print("all 15 logit dumps present; skip GPU")
        return
    if len(gpus) < 1:
        raise SystemExit("GPUS is empty")
    # sequential waves of len(gpus) via subprocess parallelism
    i = 0
    while i < len(pending):
        wave = pending[i : i + len(gpus)]
        procs: list[tuple] = []
        for (model, seed), gpu in zip(wave, gpus):
            ckpt = ROOT / "runs" / model / f"seed_{seed}" / "best.pt"
            out_dir = LOGIT_ROOT / model / f"seed_{seed}"
            logits_path = out_dir / "window_logits.npz"
            log = LOG_DIR / f"file_vote_k_{model}_seed{seed}.log"
            if not ckpt.is_file():
                raise SystemExit(f"missing frozen checkpoint: {ckpt}")
            out_dir.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(PY if PY.is_file() else sys.executable),
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
                "--window-size",
                "8192",
                "--num-workers",
                str(num_workers),
                "--seed",
                str(seed),
                "--mode",
                "classifier",
                "--file-vote-mode",
                "mean_logits",
                "--checkpoint",
                str(ckpt),
                "--out-dir",
                str(out_dir),
                "--save-window-logits",
                str(logits_path),
                *MODEL_FLAGS[model],
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = f"{KEYAN / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
            print(f"=== DUMP {model} seed={seed} GPU={gpu} (Day4, not Day5) ===")
            fh = log.open("w", encoding="utf-8")
            procs.append(
                (
                    subprocess.Popen(cmd, cwd=str(KEYAN), env=env, stdout=fh, stderr=subprocess.STDOUT),
                    fh,
                    model,
                    seed,
                )
            )
        status = 0
        for proc, fh, model, seed in procs:
            rc = proc.wait()
            fh.close()
            if rc != 0:
                status = 1
                print(f"FAIL dump {model} seed={seed}")
        if status != 0:
            raise SystemExit(f"a dump failed; see {LOG_DIR}/file_vote_k_*.log")
        i += len(gpus)


def sweep() -> dict:
    rows = []
    sanity = []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            path = LOGIT_ROOT / model / f"seed_{seed}" / "window_logits.npz"
            if not path.is_file():
                missing.append(f"{model}/seed_{seed}")
                continue
            grouped = group_rows(load_npz(model, seed))
            frozen = frozen_file_acc(model, seed)
            k256, _, n_file = acc_at_k(grouped, 256, PRIMARY_MODE)
            if n_file != 24:
                raise SystemExit(f"{model} seed {seed}: expected 24 files, got {n_file}")
            delta_pp = abs(100.0 * k256 - 100.0 * frozen)
            sanity.append(
                {
                    "model": model,
                    "seed": seed,
                    "frozen_file_acc_pct": round(100.0 * frozen, 4),
                    "k256_mean_logits_pct": round(100.0 * k256, 4),
                    "abs_pp": round(delta_pp, 4),
                    "ok": delta_pp <= SANITY_PP,
                }
            )
            for mode in VOTE_MODES:
                for k in KS:
                    file_acc, win_acc, _ = acc_at_k(grouped, k, mode)
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "k": k,
                            "vote_mode": mode,
                            "file_acc_pct": round(100.0 * file_acc, 4),
                            "window_acc_pct": round(100.0 * win_acc, 4),
                        }
                    )
    if missing:
        raise SystemExit(f"missing logits: {missing}")
    sanity_ok = all(item["ok"] for item in sanity)

    def series(model: str, mode: str, k: int) -> list[float]:
        return [
            row["file_acc_pct"]
            for row in rows
            if row["model"] == model and row["vote_mode"] == mode and row["k"] == k
        ]

    c_means = [statistics.mean(series("C_full_ratio", PRIMARY_MODE, k)) for k in KS]
    a_means = [statistics.mean(series("A_cnn_iq", PRIMARY_MODE, k)) for k in KS]
    dips = [c_means[i] - c_means[i + 1] for i in range(len(KS) - 1)]
    h4a_smooth = max(dips, default=0.0) <= SMOOTH_DIP_PP
    k64_c = statistics.mean(series("C_full_ratio", PRIMARY_MODE, 64))
    k64_a = statistics.mean(series("A_cnn_iq", PRIMARY_MODE, 64))
    c_le_a_low = all(
        statistics.mean(series("C_full_ratio", PRIMARY_MODE, k))
        <= statistics.mean(series("A_cnn_iq", PRIMARY_MODE, k))
        for k in (8, 16, 32, 64)
    )
    c_gt_a_256 = statistics.mean(series("C_full_ratio", PRIMARY_MODE, 256)) > statistics.mean(
        series("A_cnn_iq", PRIMARY_MODE, 256)
    )
    h4b_rank64 = k64_c > k64_a
    spike_only = c_le_a_low and c_gt_a_256
    if not sanity_ok:
        verdict = "SANITY_FAIL"
    elif spike_only:
        verdict = "H4_SPIKE_ONLY"
    elif h4a_smooth and h4b_rank64:
        verdict = "H4_PASS"
    elif h4a_smooth:
        verdict = "H4_SMOOTH_ONLY"
    elif h4b_rank64:
        verdict = "H4_RANK64_ONLY"
    else:
        verdict = "H4_FAIL"

    means: dict[str, dict[str, dict[str, str]]] = {}
    for model in MODELS:
        means[model] = {}
        for mode in VOTE_MODES:
            means[model][mode] = {str(k): mean_std(series(model, mode, k)) for k in KS}

    payload = {
        "day5_used": False,
        "training": False,
        "models": MODELS,
        "seeds": SEEDS,
        "k": KS,
        "vote_modes": VOTE_MODES,
        "primary_mode": PRIMARY_MODE,
        "prefix": "first_k_by_window_index",
        "sanity_pp_tol": SANITY_PP,
        "smooth_dip_pp": SMOOTH_DIP_PP,
        "sanity": sanity,
        "sanity_ok": sanity_ok,
        "rows": rows,
        "mean_file_acc_pct": means,
        "h4a_cprime_mean_logits": [round(x, 2) for x in c_means],
        "h4a_cnn_mean_logits": [round(x, 2) for x in a_means],
        "h4a_smooth": h4a_smooth,
        "h4b_cprime_gt_cnn_k64": h4b_rank64,
        "h4b_spike_only": spike_only,
        "verdict": verdict,
        "note": (
            "Revision reserve. Frozen 1C table stays K=256 mean_logits. "
            "Do not open Day5 / LODO / RX2 / S1 5-seed from this table."
        ),
    }
    return payload


def write_report(payload: dict) -> None:
    out_json = ROOT / "file_vote_k.json"
    out_md = ROOT / "file_vote_k.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    def row(model: str, mode: str) -> str:
        cells = " | ".join(f"{payload['mean_file_acc_pct'][model][mode][str(k)]:>8}" for k in KS)
        return f"| {MODEL_LABEL[model]:<15} | {cells} |"

    lines = [
        "# Paper 1 file-vote K sensitivity (Day4, frozen 1C)",
        "",
        f"verdict={payload['verdict']}  sanity_ok={payload['sanity_ok']}  day5=unused",
        f"prefix=first K by window_index  primary={PRIMARY_MODE}",
        "",
        "## File-Acc % mean±std (5 seeds)",
        "",
    ]
    for mode in VOTE_MODES:
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Model           |      K=8 |     K=16 |     K=32 |     K=64 |    K=128 |    K=256 |",
                "| --------------- | -------: | -------: | -------: | -------: | -------: | -------: |",
                row("B_exact_main_no_oob", mode),
                row("C_full_ratio", mode),
                row("A_cnn_iq", mode),
                "",
            ]
        )
    lines.extend(
        [
            "## Pre-registered H4",
            "",
            f"H4a smooth (C' mean_logits, max step drop ≤ {SMOOTH_DIP_PP} pp): {payload['h4a_smooth']}  curve={payload['h4a_cprime_mean_logits']}",
            f"H4b C' > CNN at K=64 mean_logits: {payload['h4b_cprime_gt_cnn_k64']}",
            f"SPIKE_ONLY (lose all K≤64, win only at 256): {payload['h4b_spike_only']}",
            "",
            "Frozen 1C claim table is still K=256 / mean_logits. This does not open Day5, LODO, or RX2.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print("wrote", out_json)
    print("wrote", out_md)
    print("VERDICT", payload["verdict"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "4,5"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    do_dump = args.dump or not args.sweep
    do_sweep = args.sweep or not args.dump
    if args.dump and args.sweep:
        do_dump = True
        do_sweep = True
    if do_dump and not args.smoke:
        gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
        dump_all(gpus, args.num_workers)
    if do_sweep:
        write_report(sweep())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
