#!/usr/bin/env python3
"""Runtime smoke for audited C': chirp-off OOB cross-attention backbone."""
from __future__ import annotations

import argparse
import json
import torch

from rfhstu.models import DeviceClassifier, RFPatchEmbedder


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0"); a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    embedder = RFPatchEmbedder(window_size=8192, patch_size=256, dim=64, cnn_stem_dim=32,
                               patch_embed_type="cnn_stem", use_oob=True, oob_fusion_type="cross_attn_oob",
                               use_oob_cross_attention=True,
                               fft_norm="log_zscore", oob_norm="ratio")
    model = DeviceClassifier(embedder, num_classes=10, dim=64, depth=2, use_chirp_embedding=False).to(dev)
    x = torch.randn(2, 2, 8192, device=dev)
    out = model(x, return_features=True)
    params = sum(v.numel() for v in model.parameters())
    payload = {"verdict": bool(out["logits"].shape == (2, 10) and out["embedding"].shape == (2, 64)),
               "chirp": False, "oob_fusion": "cross_attn_oob", "main_queries_oob_memory": True,
               "parameters": params, "logits_shape": list(out["logits"].shape), "embedding_shape": list(out["embedding"].shape),
               "blind_opened": False, "note": "runtime smoke only; no accuracy"}
    open(a.out, "w").write(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2)); return 0 if payload["verdict"] else 1


if __name__ == "__main__": raise SystemExit(main())
