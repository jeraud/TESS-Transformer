"""State Space Model encoder for light curve classification."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .blocks.state_space import MultiSectorS4Encoder


class StateSpaceLightCurveEncoder(nn.Module):
    """State Space Model encoder for light curves.
    
    Supports both single-sector and multi-sector processing.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 64,
        num_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        num_sectors: int = 1,
        sector_combine: str = "mean",
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        
        # Allow override via encoder_kwargs
        if encoder_kwargs:
            d_model = encoder_kwargs.get("d_model", d_model)
            d_state = encoder_kwargs.get("d_state", d_state)
            num_layers = encoder_kwargs.get("num_layers", num_layers)
            d_ff = encoder_kwargs.get("d_ff", d_ff)
            dropout = encoder_kwargs.get("dropout", dropout)
            num_sectors = encoder_kwargs.get("num_sectors", num_sectors)
            sector_combine = encoder_kwargs.get("sector_combine", sector_combine)
        
        self.encoder = MultiSectorS4Encoder(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            num_sectors=num_sectors,
            sector_combine=sector_combine,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Flux values
                - Single sector: (batch, seq_len)
                - Multi-sector: (batch, num_sectors, seq_len)
            t: Time values (same shape as x)
            mask: Mask for valid values (same shape as x)
        
        Returns:
            Embedding tensor of shape (batch, d_model)
        """
        return self.encoder(x, t, mask)
