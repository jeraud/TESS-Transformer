import torch
import torch.nn as nn
from feed_forward import FeedForward
from convolution import simple_convolution_layer, gated_colvolution_layer
from multi_head_attention import Transformer
from alternative_transformer import TransformerWithTimeEmbeddings

class post_normalize(nn.Module):
    def __init__(self, dims, module, half=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = nn.LayerNorm(dims).to('mps')
        self.module = module
        self.halfs = half

    def forward(self, x):
        if self.halfs:
            return self.layer_norm((0.5*(self.module(x))+x))
        return self.layer_norm((self.module(x)+x))
    
class post_normalize2(nn.Module):
    def __init__(self, dims, module, half=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = nn.LayerNorm(dims).to('mps')
        self.module = module
        self.halfs = half

    def forward(self, x, t):
        if self.halfs:
            return self.layer_norm((0.5*(self.module(x))+x))
        return self.layer_norm((self.module(x,t)+x))
    
class ConformerBlock(nn.Module):
    def __init__(self, transformerargs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dims = transformerargs['emb']
        self.feed_forward = FeedForward(transformerargs['emb']).to('mps')
        self.mhsa = TransformerWithTimeEmbeddings(**transformerargs).to('mps')
        self.gated_conv = gated_colvolution_layer(transformerargs['emb']).to('mps')
        self.norm_ff = post_normalize(self.dims, self.feed_forward, True).to('mps')
        self.norm_conv = post_normalize(self.dims, self.gated_conv).to('mps')
        self.norm_mhsa = post_normalize2(self.dims, self.mhsa).to('mps')
        self.LayerNorm = nn.LayerNorm(self.dims).to('mps')

    def forward(self, x,t):
        # print(x)
        x = self.norm_ff(x)
        # print('1')
        x = self.norm_mhsa(x,t)
        # print(2)
        x = self.norm_conv(x)
        # print(3)
        x = self.norm_ff(x)
        x = self.LayerNorm(x)
        return x

