#!/usr/bin/env python3
"""OOB representation equalization (embedding-level v1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.oob_equalization import apply_equalization, fit_stats  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOB representation equalization (embedding-level)")
    p.add_argument("--method", choices=["none", "mean_shift", "std_alignment", "coral"], default="mean_shift")
    p.add_argument("--help-methods", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.help_methods:
        print("Methods: none, mean_shift, std_alignment, coral")
        print("Stats from source receiver + target calibration Block A only.")
        return
    print("Use run_oob_eq_quick.py for evaluation. This module provides lib/oob_equalization.")


if __name__ == "__main__":
    main()
