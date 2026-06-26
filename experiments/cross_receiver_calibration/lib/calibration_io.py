"""Shared dataset and checkpoint helpers for calibration experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rfhstu.data import complex_iq_to_channels, normalize_iq


class SplitWindowDataset(Dataset):
    def __init__(self, rows: list[dict], input_norm: str = "iq_rms", window_size: int = 8192):
        self.rows = rows
        self.input_norm = input_norm
        self.window_size = window_size
        self._memmaps: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, path: str) -> np.memmap:
        if path not in self._memmaps:
            self._memmaps[path] = np.memmap(path, dtype=np.complex64, mode="r")
        return self._memmaps[path]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        mm = self._open(row["file_path"])
        off = int(row["sample_offset"])
        iq = np.asarray(mm[off : off + self.window_size])
        channels = complex_iq_to_channels(iq)
        if self.input_norm == "iq_rms":
            channels = normalize_iq(channels)
        return {
            "iq": torch.from_numpy(channels.copy()),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "device_id": int(row["device_id"]),
            "window_index": int(row["window_index"]),
            "role": row["role"],
            "row_idx": idx,
        }


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
