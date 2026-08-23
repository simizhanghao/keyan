from __future__ import annotations

import torch
from torch import nn


class _ViewCNN(nn.Module):
    """Small independent 1-D encoder for one physical view."""

    def __init__(self, in_channels: int, width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, padding=3), nn.BatchNorm1d(width), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2), nn.BatchNorm1d(width * 2), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width * 2, width * 4, 3, padding=1), nn.BatchNorm1d(width * 4), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(width * 4, embedding_dim), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class MultiViewLateFusionCNN(nn.Module):
    """Input-and-capacity-matched CNN control for audited C'.

    Views are encoded independently and fused only as global embeddings. This
    intentionally contains only independent convolutions and late embedding
    fusion, with no sequence-attention or receiver-style augmentation.
    """

    VIEW_CHANNELS = {"iq": 2, "fft": 2, "amp_phase": 2, "oob": 1}

    def __init__(self, num_classes: int, width: int = 32, embedding_dim: int = 128,
                 fusion_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.views = nn.ModuleDict({
            name: _ViewCNN(channels, width, embedding_dim, dropout)
            for name, channels in self.VIEW_CHANNELS.items()
        })
        self.fusion = nn.Sequential(
            nn.Linear(4 * embedding_dim, fusion_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, embedding_dim), nn.SiLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, views: dict[str, torch.Tensor], return_features: bool = False) -> dict[str, torch.Tensor]:
        missing = set(self.VIEW_CHANNELS) - set(views)
        if missing:
            raise ValueError(f"B1 requires views: {sorted(missing)}")
        z = self.fusion(torch.cat([self.views[name](views[name]) for name in self.VIEW_CHANNELS], dim=-1))
        out = {"embedding": z, "logits": self.classifier(z)}
        if return_features:
            out["features"] = z
        return out
