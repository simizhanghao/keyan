from __future__ import annotations

import math

import torch
from torch import nn

from .features import build_patch_features
from .losses import DomainAdversarialHead


class RFHSTUBlock(nn.Module):
    """Minimal RF-HSTU block for the first implementation pass."""

    def __init__(self, dim: int, num_patches: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.qkvu = nn.Linear(dim, dim * 4)
        self.relative_bias = nn.Parameter(torch.zeros(2 * num_patches - 1))
        self.av_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.residual_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        idx = torch.arange(num_patches)
        rel = idx[:, None] - idx[None, :] + num_patches - 1
        self.register_buffer("relative_index", rel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, u = self.qkvu(x).chunk(4, dim=-1)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        score = score + self.relative_bias[self.relative_index].unsqueeze(0)
        attn = torch.sigmoid(score)
        av = attn @ v
        y = self.out(self.av_norm(av) * self.act(u))
        y = self.dropout(y)
        return self.residual_norm(x + y)


class RFHSTUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_patches: int = 32,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.pool = pool
        self.input_proj = nn.Linear(input_dim, dim)
        self.position = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.ModuleList([RFHSTUBlock(dim, num_patches, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)

    def forward_tokens(self, patch_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(patch_features) + self.position
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(patch_features)
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "first":
            return tokens[:, 0]
        raise ValueError(f"Unknown pool mode: {self.pool}")


class RFPatchEmbedder(nn.Module):
    def __init__(
        self,
        window_size: int = 8192,
        patch_size: int = 256,
        sample_rate: float = 1_000_000.0,
        lora_bandwidth: float = 125_000.0,
        use_oob: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.lora_bandwidth = lora_bandwidth
        self.use_oob = use_oob

    @property
    def num_patches(self) -> int:
        return self.window_size // self.patch_size

    @property
    def input_dim(self) -> int:
        channels = 2 + 1 + 2
        if self.use_oob:
            channels += 1
        return channels * self.patch_size

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return build_patch_features(
            iq,
            patch_size=self.patch_size,
            sample_rate=self.sample_rate,
            lora_bandwidth=self.lora_bandwidth,
            use_oob=self.use_oob,
        )


class RFMAE(nn.Module):
    def __init__(
        self,
        embedder: RFPatchEmbedder,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        mask_ratio: float = 0.4,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedder.input_dim))
        self.encoder = RFHSTUEncoder(
            input_dim=embedder.input_dim,
            num_patches=embedder.num_patches,
            dim=dim,
            depth=depth,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {
                "iq": nn.Linear(dim, 2 * embedder.patch_size),
                "fft": nn.Linear(dim, embedder.patch_size),
                "amp_phase": nn.Linear(dim, 2 * embedder.patch_size),
            }
        )
        if embedder.use_oob:
            self.heads["oob"] = nn.Linear(dim, embedder.patch_size)

    def make_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        patches = self.embedder.num_patches
        num_mask = max(1, int(round(patches * self.mask_ratio)))
        noise = torch.rand(batch_size, patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(batch_size, patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask

    def forward(self, iq: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        features, targets = self.embedder(iq)
        mask = self.make_mask(features.shape[0], features.device)
        masked_features = torch.where(mask.unsqueeze(-1), self.mask_token.to(features.dtype), features)
        tokens = self.encoder.forward_tokens(masked_features)
        pred = {name: head(tokens) for name, head in self.heads.items()}
        return pred, targets, mask


class DeviceClassifier(nn.Module):
    def __init__(
        self,
        embedder: RFPatchEmbedder,
        num_classes: int,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        domain_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = RFHSTUEncoder(
            input_dim=embedder.input_dim,
            num_patches=embedder.num_patches,
            dim=dim,
            depth=depth,
            dropout=dropout,
        )
        self.classifier = nn.Linear(dim, num_classes)
        self.domain_head = DomainAdversarialHead(dim, domain_sizes) if domain_sizes else None

    def forward(
        self,
        iq: torch.Tensor,
        adv_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        features, _ = self.embedder(iq)
        z = self.encoder(features)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "embedding": z,
            "logits": self.classifier(z),
        }
        if self.domain_head is not None:
            out["domain_logits"] = self.domain_head(z, adv_lambda)
        return out

