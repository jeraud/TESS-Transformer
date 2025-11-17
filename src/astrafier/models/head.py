"""Classification head for Astrafier embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(normalized_shape=128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

