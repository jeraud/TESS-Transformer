import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

def scaled_dot_prod(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2))
    if mask is not None:
        print('error')
        scaled += mask
    soft = F.softmax(scaled, dim=-1)
    attention = torch.matmul(soft, v)
    return attention

class TimePositionalEncoding(nn.Module):
    """ Time encodings for Transformer. 
    """

    def __init__(self, d_emb):
        """
        Inputs
            d_emb - Dimensionality when projecting to the fourier feature basis.
        """
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t):
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(torch.arange(0, self.d_emb, 2).float() * (-math.log(10000.0) / self.d_emb))[None, None, :].to(t.device)  # (1, 1, D / 2)
        t = t.unsqueeze(2)  # (B, 1, T)
        pe[:, :, 0::2] = torch.sin(t * div_term)  # (B, T, D / 2)
        pe[:, :, 1::2] = torch.cos(t * div_term)  # (B, T, D / 2)
        return pe  # (B, T, D)

class MHSA(nn.Module):
    def __init__(self, emb_d, num_heads):
        super(MHSA,self).__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.head_d = emb_d // num_heads
        self.QKV = nn.Linear(emb_d, 3 * emb_d)
        self.ffn = nn.Linear(emb_d, emb_d)
    def forward(self, x, t, mask=None):
        batch_size, max_len, dims = x.shape
        qkv = self.QKV(x)
        qkv = qkv.reshape(batch_size, max_len, self.num_heads, 3 * self.head_d)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3, dim=-1)
        attention = scaled_dot_prod(q,k,v,mask)
        attention = attention.reshape(batch_size, max_len, self.num_heads * self.head_d)
        output = self.ffn(attention)
        return output

class ConvFeedForward(nn.Module):
    def __init__(self, emb_d, hidden, dropout_p=0.1):
        super(ConvFeedForward, self).__init__()
        self.convo1 = nn.Conv1d(emb_d, hidden, kernel_size=1, bias=False)
        self.convo2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding='same')
        self.convo3 = nn.Conv1d(hidden, emb_d, kernel_size=1, bias=False)
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.norm = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self,x):
        x = self.convo1(x.transpose(1,2))
        x = self.swish1(x)
        x = self.convo2(x)
        x = self.norm(x)
        x = self.swish2(x)
        x = self.convo3(x)
        x = x.transpose(1,2)
        x = self.dropout(x)
        return x
        
class TransformerLayer(nn.Module):
    def __init__(self, emb_d, num_heads, ffn_d, dropout_p):
        super(TransformerLayer,self).__init__()
        self.time_encoding = TimePositionalEncoding(emb_d)
        self.attention = MHSA(emb_d=emb_d, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.cffn = ConvFeedForward(emb_d, ffn_d, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)
    def forward(self,x,t, mask=None):
        t = t - t[:, 0].unsqueeze(1)
        x = x + self.time_encoding(t)
        residual_x = x
        x = self.attention(x, t, mask=None)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.cffn(x)
        x = self.norm2(x + residual_x)
        return x
