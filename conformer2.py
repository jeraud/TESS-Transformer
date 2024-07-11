import torch
import torch.nn as nn
from multi_head_attention import TransformerLayer
from convolution import gated_convolution_layer
import torch.nn.functional as F


class Conformer(nn.Module):
    def __init__(self, emb_d, num_heads, dropout_p, ffn_d):
        super(Conformer, self).__init__()
        self.MHSA = TransformerLayer(emb_d=emb_d, ffn_d=ffn_d, num_heads=num_heads, dropout_p=dropout_p)
        self.conv = gated_convolution_layer(emb_d, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.lstm = nn.LSTM(emb_d,int(emb_d/2), batch_first=True, dropout=dropout_p, num_layers=2, bidirectional=True)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)
        
    def forward(self, x, t, mask=None):
        old = x
        old = self.lstm(old)[0]
        x = self.MHSA(x,t, mask)
        x = x + old
        x = self.norm2(x)
        resid_x = x
        x = self.conv(x)
        x = x + resid_x
        x = self.norm1(x)
        return x



class LCC(nn.Module):
    def __init__(self, emb_d, num_heads, layers, dropout_p, ffn_d):
        super(LCC,self).__init__()
        self.conformers  = nn.ModuleList([Conformer(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d) for _ in range(layers)])
        self.fc1 = nn.Linear(emb_d, 20)
        self.fc2 = nn.Linear(20,8)
        self.embed1 = nn.Linear(1, emb_d)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.swish = nn.SiLU()

    def forward(self, x,t, mask=None):
        x = self.embed1(x.unsqueeze(2))
        x = self.norm1(x)
        for conformer in self.conformers:
            x = conformer(x, t)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1).squeeze()
        return x