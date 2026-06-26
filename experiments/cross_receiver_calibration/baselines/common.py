"""Shared helpers for same-protocol baseline comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from lib.oob_equalization import apply_equalization, fit_stats
from lib.split_protocol import ROLE_CALIBRATION, ROLE_QUERY, ROLE_SOURCE, ROLE_SUPPORT, sample_k_support_indices

# Reuse RCPA eval utilities from parent package.
import sys

CAL_ROOT = Path(__file__).resolve().parents[1]
if str(CAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CAL_ROOT))

from run_rcpa_prototypes import (  # noqa: E402
    build_class_prototypes,
    eval_prototype_file_acc,
    file_level_vote,
    load_role_data,
    macro_f1,
    predict_prototype,
)


def load_classifier_head(checkpoint: Path, device: torch.device) -> nn.Linear:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"]
    w = state["classifier.weight"]
    b = state["classifier.bias"]
    head = nn.Linear(w.shape[1], w.shape[0])
    head.weight.data = w.clone()
    head.bias.data = b.clone()
    return head


def eval_linear_classifier_file_acc(
    qry: dict,
    clf: LogisticRegression | nn.Module,
    *,
    torch_mode: bool = False,
    device: torch.device | None = None,
    num_classes: int = 24,
) -> tuple[float, float]:
    file_labels, file_preds = [], []
    for dev in sorted(set(qry["device"].tolist())):
        mask = qry["device"] == dev
        z_mean = qry["z"][mask].mean(axis=0, keepdims=True)
        if torch_mode:
            assert device is not None
            with torch.no_grad():
                logits = clf(torch.from_numpy(z_mean).float().to(device))
                pred = int(logits.argmax(dim=-1).cpu().item())
        else:
            pred = int(clf.predict(z_mean)[0])
        file_labels.append(int(qry["y"][mask][0]))
        file_preds.append(pred)
    acc = sum(a == b for a, b in zip(file_labels, file_preds)) / max(len(file_labels), 1)
    return acc, macro_f1(file_labels, file_preds, num_classes)


def train_linear_probe(support_z: np.ndarray, support_y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", C=1.0)
    clf.fit(support_z, support_y)
    return clf


def finetune_linear_head(
    support_z: np.ndarray,
    support_y: np.ndarray,
    *,
    init: str,
    checkpoint: Path | None,
    seed: int,
    epochs: int = 50,
    lr: float = 0.05,
    device: torch.device,
    num_classes: int = 24,
) -> nn.Linear:
    dim = support_z.shape[1]
    head = nn.Linear(dim, num_classes).to(device)
    if init == "source":
        if checkpoint is None:
            raise ValueError("checkpoint required for source init")
        src_head = load_classifier_head(checkpoint, device)
        head.load_state_dict(src_head.state_dict())
    else:
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)

    x = torch.from_numpy(support_z).float().to(device)
    y = torch.from_numpy(support_y).long().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(x), y)
        loss.backward()
        opt.step()
    head.eval()
    return head


def align_features(
    z: np.ndarray,
    method: str,
    src_z: np.ndarray,
    cal_z: np.ndarray,
) -> np.ndarray:
    mu_s, std_s, cov_s = fit_stats(src_z)
    mu_t, std_t, cov_t = fit_stats(cal_z)
    return apply_equalization(z, method, mu_s, std_s, cov_s, mu_t, std_t, cov_t)


def eval_source_classifier_on_z(
    qry: dict,
    head: nn.Linear,
    device: torch.device,
    num_classes: int = 24,
) -> tuple[float, float]:
    qry_aligned = {**qry, "z": qry["z"]}
    return eval_linear_classifier_file_acc(
        qry_aligned, head, torch_mode=True, device=device, num_classes=num_classes
    )


def support_subset(sup: dict, k: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    support_indices = set(sample_k_support_indices(k, split_seed))
    mask = np.array([int(w) in support_indices for w in sup["win"]], dtype=bool)
    return sup["z"][mask], sup["y"][mask]


def read_split_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
