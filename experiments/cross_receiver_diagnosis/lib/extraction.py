"""Embedding extraction helpers for cross-receiver diagnosis."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rfhstu.cnn_baseline import OSUCNNBaseline, build_cnn_input
from rfhstu.models import DeviceClassifier


def _add_chirp(enc, x: torch.Tensor) -> torch.Tensor:
    if not enc.use_chirp_embedding:
        return x
    chirp = enc.chirp_id_embedding(enc.chirp_ids).unsqueeze(0)
    patch_in_chirp = enc.patch_in_chirp_embedding(enc.patch_in_chirp_ids).unsqueeze(0)
    return x + chirp + patch_in_chirp


def _pool_tokens(enc, tokens: torch.Tensor) -> torch.Tensor:
    for block in enc.blocks:
        tokens = block(tokens)
    tokens = enc.final_norm(tokens)
    return tokens.mean(dim=1) if enc.pool == "mean" else tokens[:, 0]


@torch.no_grad()
def encode_hybrid_paths(model: DeviceClassifier, iq: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return pooled embeddings for main-only, OOB-only, and fused paths."""
    features, views = model.embedder(iq)
    enc = model.encoder
    x_main = _add_chirp(enc, enc.input_proj(features) + enc.position)

    oob = views.get("oob")
    if oob is None or enc.oob_proj is None or enc.oob_fusion is None:
        z = _pool_tokens(enc, x_main)
        return {"main": z, "oob": z, "fused": z}

    x_oob = _add_chirp(enc, enc.oob_proj(oob) + enc.position)
    x_fused = enc.oob_fusion(x_main, x_oob)
    return {
        "main": _pool_tokens(enc, x_main),
        "oob": _pool_tokens(enc, x_oob),
        "fused": _pool_tokens(enc, x_fused),
    }


@torch.no_grad()
def extract_batch(
    model: torch.nn.Module,
    iq: torch.Tensor,
    ckpt_args: dict[str, Any],
    model_type: str,
) -> dict[str, torch.Tensor]:
    if model_type == "osu_cnn":
        x = build_cnn_input(
            iq,
            input_type=ckpt_args.get("cnn_input_type", "iq"),
            sample_rate=ckpt_args.get("sample_rate", 1e6),
            lora_bandwidth=ckpt_args.get("lora_bandwidth", 125e3),
        )
        out = model(x)
        z = out["embedding"]
        logits = out["logits"]
        return {
            "fused": z,
            "main": z,
            "oob": z,
            "logits": logits,
        }

    paths = encode_hybrid_paths(model, iq)
    logits = model.classifier(paths["fused"])
    paths["logits"] = logits
    return paths


def l2_normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, dim=-1)
