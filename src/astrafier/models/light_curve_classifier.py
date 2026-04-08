"""Neural encoder that produces embeddings for the Astrafier head."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .blocks.lsatt_conv import LCC


class LightCurveEncoder(nn.Module):
    def __init__(
        self,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        transformer_kwargs = transformer_kwargs or {
            "emb": 64,
            "heads": 8,
            "layers": 3,
            "dropout_p": 0.2,
            "hidden": 256,
            "num_classes": 8,
        }
        self.classifier = LCC(
            emb_d=transformer_kwargs["emb"],
            num_heads=transformer_kwargs["heads"],
            layers=transformer_kwargs["layers"],
            dropout_p=transformer_kwargs["dropout_p"],
            ffn_d=transformer_kwargs["hidden"],
            num_classes=8,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return sequence embeddings for downstream classification."""

        return self.classifier(x, t, mask, umap=True)

