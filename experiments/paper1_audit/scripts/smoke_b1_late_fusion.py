#!/usr/bin/env python3
"""Static B1 contract smoke; runtime forward is run in the torch environment."""
from __future__ import annotations

import ast
import argparse
from pathlib import Path


FORBIDDEN = ("Attention", "HSTU", "GradReverse", "chirp")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source = (Path(__file__).parents[3] / "src/rfhstu/b1_late_fusion.py").read_text()
    ast.parse(source)
    checks = {
        "source_receivers_14": len(list(args.source_root.glob("*_train.h5"))) == 14,
        "blind_paths_absent": not any(x in str(args.source_root) for x in ("b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2")),
        "four_views": all(name in source for name in ("iq", "fft", "amp_phase", "oob")),
        "late_fusion": "torch.cat" in source and "self.fusion" in source,
        "no_forbidden_components": not any(token in source for token in FORBIDDEN),
        "ten_class_contract_documented": "num_classes" in source,
    }
    payload = {"checks": checks, "all_pass": all(checks.values()), "runtime_forward": "requires torch environment"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(__import__("json").dumps(payload, indent=2) + "\n")
    print(__import__("json").dumps(payload, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
