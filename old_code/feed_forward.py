import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout_p=0.3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim*expansion_factor, dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        # print(x.device)
        # print(self.model.device)
        return self.model(x)
