"""Block-disjoint window split protocol for RCPA calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_WINDOWS = 256
BLOCK_SIZE = 64
WINDOW_SIZE = 8192

BLOCK_ORDER = ["A", "B", "C", "D"]

BLOCK_RANGES = {
    "A": (0, 64),
    "B": (64, 128),
    "C": (128, 192),
    "D": (192, 256),
}

ROLE_CALIBRATION = "calibration"
ROLE_SUPPORT = "support"
ROLE_QUERY = "query"
ROLE_SOURCE = "source_train"


def block_of(window_index: int) -> str:
    if window_index < 64:
        return "A"
    if window_index < 128:
        return "B"
    if window_index < 192:
        return "C"
    return "D"


def window_offset(window_index: int, file_length: int, window_size: int = WINDOW_SIZE) -> int:
    """Deterministic offset matching SigMFIQDataset with samples_per_file=256."""
    max_offset = max(0, file_length - window_size)
    if max_offset == 0:
        return 0
    stride = max(1, max_offset // max(1, NUM_WINDOWS - 1))
    return min(window_index * stride, max_offset)


def split_config(split_id: int) -> dict[str, str | tuple[str, str]]:
    """Rotate block roles: cal, support, query (two blocks)."""
    n = len(BLOCK_ORDER)
    sid = split_id % n
    return {
        ROLE_CALIBRATION: BLOCK_ORDER[sid],
        ROLE_SUPPORT: BLOCK_ORDER[(sid + 1) % n],
        ROLE_QUERY: (BLOCK_ORDER[(sid + 2) % n], BLOCK_ORDER[(sid + 3) % n]),
    }


def role_for_window(window_index: int, split_id: int = 0) -> str:
    block = block_of(window_index)
    cfg = split_config(split_id)
    if block == cfg[ROLE_CALIBRATION]:
        return ROLE_CALIBRATION
    if block == cfg[ROLE_SUPPORT]:
        return ROLE_SUPPORT
    return ROLE_QUERY


def support_block_range(split_id: int = 0) -> tuple[int, int]:
    sup_block = str(split_config(split_id)[ROLE_SUPPORT])
    return BLOCK_RANGES[sup_block]


def sample_k_support_indices(k: int, split_id: int = 0) -> list[int]:
    """Uniformly sample K window indices from the support block for split_id."""
    lo, hi = support_block_range(split_id)
    block_w = list(range(lo, hi))
    if k <= 0:
        return []
    if k >= len(block_w):
        return block_w
    positions = np.linspace(0, len(block_w) - 1, k, dtype=int)
    return [block_w[int(p)] for p in positions]


@dataclass(frozen=True)
class SplitKey:
    direction: str
    device_id: int
    window_index: int

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.direction, self.device_id, self.window_index)


def assert_disjoint_roles(rows: list[dict]) -> None:
    """Assert support/query/calibration window sets are disjoint per device."""
    by_device: dict[int, dict[str, set[int]]] = {}
    for row in rows:
        if row["role"] == ROLE_SOURCE:
            continue
        dev = int(row["device_id"])
        role = row["role"]
        wi = int(row["window_index"])
        by_device.setdefault(dev, {}).setdefault(role, set()).add(wi)

    for dev, roles in by_device.items():
        cal = roles.get(ROLE_CALIBRATION, set())
        sup = roles.get(ROLE_SUPPORT, set())
        qry = roles.get(ROLE_QUERY, set())
        assert cal.isdisjoint(qry), f"device {dev}: calibration ∩ query = {cal & qry}"
        assert sup.isdisjoint(qry), f"device {dev}: support ∩ query = {sup & qry}"
        assert cal.isdisjoint(sup), f"device {dev}: calibration ∩ support = {cal & sup}"
