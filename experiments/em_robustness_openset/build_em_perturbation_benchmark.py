#!/usr/bin/env python3
"""Build EM perturbation benchmark config and audit report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/em_robustness_openset/results"

from rfhstu.em_perturbations import (  # noqa: E402
    MIXED_STRESS_PRESETS,
    PERTURBATION_PHYSICS,
    ROBUSTNESS_SWEEPS,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "version": "chapter5_v1",
        "sweeps": ROBUSTNESS_SWEEPS,
        "mixed_stress": {k: vars(v) for k, v in MIXED_STRESS_PRESETS.items()},
        "physics": PERTURBATION_PHYSICS,
        "train_sampling": [
            "awgn_snr_db moderate (5–25 dB)",
            "cfo_norm moderate (0.01–0.05)",
            "phase_noise_std moderate (0.01–0.05)",
            "narrowband_sir_db moderate (5–20 dB)",
            "iq_amp_db / iq_phase_deg light",
        ],
    }
    (OUT / "em_benchmark_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    lines = [
        "# EM Perturbation Audit",
        "",
        "Physical interpretation of LoRa IQ perturbations for thesis Chapter 5.",
        "",
        "| Perturbation | Physical meaning | Strength range | Train | Test |",
        "|--------------|------------------|----------------|-------|------|",
    ]
    for key, physics in PERTURBATION_PHYSICS.items():
        sweep = ROBUSTNESS_SWEEPS.get(key, [])
        lines.append(
            f"| {key} | {physics} | {sweep} | yes (moderate) | yes (full sweep) |"
        )
    lines += [
        "",
        "## Mixed stress presets",
        "",
    ]
    for name, preset in MIXED_STRESS_PRESETS.items():
        lines.append(f"- **{name}**: {vars(preset)}")
    lines += [
        "",
        "## Notes",
        "",
        "- AWGN: thermal / environmental noise.",
        "- Narrowband interference: adjacent-channel or co-channel interferer.",
        "- CFO: oscillator frequency offset / sync error.",
        "- Phase noise: oscillator phase jitter.",
        "- IQ imbalance: receiver front-end orthogonality and gain mismatch.",
        "- Filter drift: receiver filter / cable frequency response tilt.",
    ]
    (OUT / "em_perturbation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'em_benchmark_config.json'}")
    print(f"Wrote {OUT / 'em_perturbation_audit.md'}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    main()
