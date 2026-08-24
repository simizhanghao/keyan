#!/usr/bin/env python3
"""Launch the frozen 630-run X6 evaluation queue across four GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECEIVERS = ("b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2")
MODELS = ("Shen-CIS", "Shen-RA", "B1-OOB", "C'-OOB", "C'-TrueIB")
INTERVENTIONS = (
    "scale_0.5", "scale_0.70710678", "scale_1.41421356", "scale_2.0",
    "shuffle", "neutral", "left_scale_0.5", "right_scale_0.5",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--neutral", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--manifest-only", action="store_true")
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    runner = here / "run_x6_confirmatory.py"
    tasks = []
    for receiver in RECEIVERS:
        data = a.data_root / f"{receiver}_test.h5"
        for model in MODELS:
            for seed in range(5):
                tasks.append((receiver, data, model, seed, "clean"))
        for model in ("B1-OOB", "C'-OOB"):
            for seed in range(5):
                for condition in INTERVENTIONS:
                    tasks.append((receiver, data, model, seed, condition))
    assert len(tasks) == 630 and len(set((r, m, s, c) for r, _, m, s, c in tasks)) == 630
    manifest = [{"receiver": r, "data_file": str(d), "model": m, "seed": s, "condition": c} for r, d, m, s, c in tasks]
    a.out_root.mkdir(parents=True, exist_ok=True)
    (a.out_root / "X6_TASK_MANIFEST.json").write_text(json.dumps({"count": 630, "tasks": manifest}, indent=2) + "\n")
    if a.manifest_only:
        print(json.dumps({"tasks": len(tasks), "blind_signal_opened": False}))
        return
    missing = [str(d) for _, d, _, _, _ in tasks if not d.is_file()]
    if missing:
        raise SystemExit(f"missing blind files: {sorted(set(missing))}")
    gpus = [int(x) for x in a.gpus.split(",")]
    if len(gpus) != 4:
        raise SystemExit("X6 is frozen to four workers")
    lanes = [tasks[i::4] for i in range(4)]
    env = os.environ.copy()
    repo = here.parents[2]
    env["PYTHONPATH"] = f"{repo / 'src'}:{here}:{env.get('PYTHONPATH', '')}"

    def worker(gpu, lane):
        for receiver, data, model, seed, condition in lane:
            stem = model.replace("'", "prime")
            out = a.out_root / receiver / stem / f"seed_{seed}" / f"{condition}.json"
            if out.exists():
                continue
            cmd = [str(a.python), str(runner), "--phase", "blind-confirmatory", "--snapshot", str(a.snapshot),
                   "--data-file", str(data), "--model", model, "--seed", str(seed), "--condition", condition,
                   "--out", str(out), "--device", f"cuda:{gpu}", "--batch-size", "128"]
            if condition == "neutral":
                cmd.extend(["--neutral", str(a.neutral)])
            subprocess.run(cmd, check=True, env=env)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu, lane) for gpu, lane in zip(gpus, lanes)]
        for future in futures:
            future.result()
    print(json.dumps({"status": "complete", "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
