import torch
import torch.nn as nn
import torch.nn.functional as F
from waveletscat import WaveletScatteringTransform
from feed_forward import FeedForward
from convolution import simple_convolution_layer, gated_colvolution_layer
# from multi_head_attention import PosEncMHSA
from conformer_block import ConformerBlock
# print(torch.cuda.is_available())
class WaveScatFormer(nn.Module):
    def __init__(self, transformer_kwargs, dropout_p=0.03, *args,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = transformer_kwargs['emb']
        self.dropout = nn.Dropout(dropout_p)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(transformer_kwargs) for _ in range(5)])
        self.fc1 = nn.Linear(1350,self.emb_dim)
        self.fc2 = nn.Linear(512,50)
        self.fc3 = nn.Linear(50,8)
        self.reLU = nn.ReLU()
       

    def forward(self, x, t, mask=None):
        # print(x.shape)
        # x = WaveletScatteringTransform('cpu')
        # print(x.shape)
        x = self.fc1(x)
        t = self.fc1(t)
        # print(x.shape)
        x = self.dropout(x).unsqueeze(0)
        # print(x.shape)
        for block in self.conformer_blocks:
            x = block(x,t)
            # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.reLU(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = F.softmax(x,dim=1).squeeze()
        # print('here',x.shape)
        return x
