import torch
import torch.nn as nn
from multi_head_attention import Transformer, TransformerLayer
from convolution import gated_colvolution_layer
import torch.nn.functional as F

class Conformer(nn.Module):
    def __init__(self, emb_d, num_heads, dropout_p, ffn_d):
        super(Conformer, self).__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.ffn_d = ffn_d
        self.MHSA = TransformerLayer(emb_d=emb_d, ffn_d=ffn_d, num_heads=num_heads, dropout_p=dropout_p)
        self.conv = gated_colvolution_layer(emb_d, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)

    def forward(self, x, t, mask=None):
        # print(x.shape)
        x = self.MHSA(x,t, mask)
        # print(x.shape)
        # x = self.norm1(x)
        x = self.conv(x)
        # print(x.shape)
        x = self.norm1(x)
        # print('good',x.shape)
        return x



class LCC(nn.Module):
    def __init__(self, emb_d, num_heads, layers, dropout_p, ffn_d):
        super(LCC,self).__init__()
        self.conformers  = nn.ModuleList([Conformer(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d) for _ in range(layers)])
        # self.conformer = Conformer(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d)
        self.fc1 = nn.Linear(emb_d, 32)
        self.fc2 = nn.Linear(32,8)
        self.ReLU = nn.ReLU()
        self.embed1 = nn.Linear(1, 32)
        self.embed2 = nn.Linear(32, emb_d)
        self.ReLU2 = nn.ReLU()
        self.projection = nn.Linear(emb_d,1)
    def forward(self, x,t, mask=None):
        x = self.embed1(x.unsqueeze(2))
        x = self.ReLU(x)
        x = self.embed2(x)
        # print('1',x.shape)
        # print(self.conformers)
        # x = self.conformer(x,t)
        for conformer in self.conformers:
            # print('here', x.shape)
            x = conformer(x, t)
        # print('huh',x.shape)
        # x = self.projection(x)
        x = x.mean(dim=1)
        # x= x.squeeze()
        # print('bb',x.shape)
        x = self.fc1(x)
        # print('2',x.shape)
        x = self.ReLU(x)
        x = self.fc2(x)
        # print('3',x.shape)
        x = F.softmax(x, dim=-1).squeeze()
        # print('a',x.shape)
        return x