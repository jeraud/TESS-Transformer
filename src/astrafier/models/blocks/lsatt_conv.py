"""Low-level LSATT convolutional transformer blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_prod(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    d_k = q.size()[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        m = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(m == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    if mask is not None:
        m_q = mask[:, None, :, None]
        attn = attn * m_q
    out = torch.matmul(attn, v)
    if mask is not None:
        out = out * m_q
    return out


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_emb: int) -> None:
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        batch_size = t.shape[0]
        max_len = t.shape[1]
        pe = torch.zeros(batch_size, max_len, self.d_emb, device=t.device)
        div_term = torch.exp(
            torch.arange(0, self.d_emb, 2, device=t.device).float() * (-math.log(10000.0) / self.d_emb)
        )[None, None, :]
        t = t.unsqueeze(2)
        pe[:, :, 0::2] = torch.sin((t / div_term) * (self.d_emb / max_len))
        pe[:, :, 1::2] = torch.cos((t / div_term) * (self.d_emb / max_len))
        return pe


class MHSA(nn.Module):
    def __init__(self, emb_d: int, num_heads: int) -> None:
        super().__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.head_d = emb_d // num_heads
        self.QKV = nn.Linear(emb_d, 3 * emb_d)
        self.ffn = nn.Linear(emb_d, emb_d)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, max_len, _ = x.shape
        qkv = self.QKV(x)
        qkv = qkv.reshape(batch_size, max_len, self.num_heads, 3 * self.head_d)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attention = scaled_dot_prod(q, k, v, mask, t)
        attention = attention.reshape(batch_size, max_len, self.num_heads * self.head_d)
        return self.ffn(attention)


class ConvFeedForward(nn.Module):
    def __init__(self, emb_d: int, hidden: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.convo1 = nn.Conv1d(emb_d, hidden, kernel_size=1, bias=False)
        self.convo2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding="same")
        self.convo3 = nn.Conv1d(hidden, emb_d, kernel_size=1, bias=False)
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.norm = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convo1(x.transpose(1, 2))
        x = self.swish1(x)
        x = self.convo2(x)
        x = self.norm(x)
        x = self.swish2(x)
        x = self.dropout(x)
        x = self.convo3(x)
        return x.transpose(1, 2)


class TransformerLayer(nn.Module):
    def __init__(self, emb_d: int, num_heads: int, ffn_d: int, dropout_p: float) -> None:
        super().__init__()
        self.time_encoding = TimePositionalEncoding(emb_d)
        self.attention = MHSA(emb_d=emb_d, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.cffn = ConvFeedForward(emb_d, ffn_d, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            t = t.clone()
            t[~mask] = t[:, :1].expand_as(t)[~mask]
        t = t - t[:, 0].unsqueeze(1)
        x = x + self.time_encoding(t)
        x = self.dropout1(x)
        residual_x = x
        x = self.attention(x, t, mask)
        x = self.dropout2(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.cffn(x)
        x = self.dropout3(x)
        return self.norm2(x + residual_x)



class glu_convolution_layer(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(dims,4*dims,kernel_size=1, bias=False)
        self.GLU = nn.GLU(dim=1)
        self.convolution = nn.Conv1d(2*dims,2*dims, kernel_size=3, stride=1, padding='same')
        # self.BatchNorm = nn.BatchNorm1d(2*dims)
        self.BatchNorm = nn.GroupNorm(1, 2*dims)
        self.swish = nn.SiLU()
        self.swish0 = nn.SiLU()
        self.convolution1 = nn.Conv1d(2*dims,2*dims, kernel_size=7, stride=1, padding='same')
        # self.BatchNorm1 = nn.BatchNorm1d(2*dims)
        self.BatchNorm1 = nn.GroupNorm(1, 2*dims)
        self.swish01 = nn.SiLU()
        self.convolution2 = nn.Conv1d(2*dims,2*dims, kernel_size=15, stride=1, padding='same')
        # self.BatchNorm2 = nn.BatchNorm1d(2*dims)
        self.BatchNorm2 = nn.GroupNorm(1, 2*dims)
        self.swish02 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(2*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.longrange = nn.Conv1d(2*dims,2*dims,kernel_size=111, stride=1, padding='same')
        # self.batchNorm2 = nn.BatchNorm1d(2*dims)
        self.batchNorm2 = nn.GroupNorm(1, 2*dims)

    def forward(self, x):
        x =  self.dropout1(x)
        x = self.pointwise1(x.transpose(1,2))
        x = self.GLU(x)
        x = self.convolution(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.convolution1(x)
        x = self.BatchNorm1(x)
        x = self.swish01(x)
        x = self.convolution2(x)
        x = self.BatchNorm2(x)
        x = self.swish02(x)
        x = self.longrange(x)
        x = self.batchNorm2(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.pointwise2(x)
        x = x.transpose(1,2)
        return x
        
class convolution_layer_lstm(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(2*dims,4*dims,kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.convolution = nn.Conv1d(4*dims,4*dims, kernel_size=3, stride=1, padding='same')
        # self.BatchNorm = nn.BatchNorm1d(4*dims)
        self.BatchNorm = nn.GroupNorm(1, 4*dims)
        self.swish0 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(4*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        

    def forward(self, x):
        x =  self.dropout1(x)
        x = self.pointwise1(x.transpose(1,2))
        x = self.relu(x)
        x = self.convolution(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.dropout(x)
        x = self.pointwise2(x)
        x = x.transpose(1,2)
        return x



class LSATT_CONV(nn.Module):
    def __init__(self, emb_d: int, num_heads: int, dropout_p: float, ffn_d: int) -> None:
        super().__init__()
        self.MHSA = TransformerLayer(emb_d=emb_d, ffn_d=ffn_d, num_heads=num_heads, dropout_p=dropout_p)
        self.conv = glu_convolution_layer(emb_d, dropout_p=dropout_p)
        self.conv2 = convolution_layer_lstm(emb_d, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.lstm = nn.LSTM(emb_d, emb_d, batch_first=True, dropout=dropout_p, num_layers=2, bidirectional=True)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        resid_x = x
        x = self.lstm(x)[0]
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = self.conv2(x)
        x = x + resid_x
        x = self.norm2(x)
        x = self.MHSA(x, t, mask)
        resid_x = x
        x = self.conv(x)
        x = x + resid_x
        return self.norm1(x)


class LCC(nn.Module):
    def __init__(self, emb_d: int, num_heads: int, layers: int, dropout_p: float, ffn_d: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lsatt_conv_layers = nn.ModuleList(
            [LSATT_CONV(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d) for _ in range(layers)]
        )
        self.embed1 = nn.Linear(1, emb_d)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor], umap: bool = False) -> torch.Tensor:
        x = self.embed1(x.unsqueeze(2))
        for lsatt_conv in self.lsatt_conv_layers:
            x = lsatt_conv(x, t, mask)
        if mask is not None:
            m = mask.unsqueeze(-1)
            x_sum = (x * m).sum(dim=1)
            denom = m.sum(dim=1).clamp_min(1)
            x = x_sum / denom
        else:
            x = x.mean(dim=1)
        return x

