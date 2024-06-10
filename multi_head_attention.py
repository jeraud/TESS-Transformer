import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

def scaled_dot_prod(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2))
    if mask is not None:
        print('fuck')
        scaled += mask
    soft = F.softmax(scaled, dim=-1)
    attention = torch.matmul(soft, v)
    return attention

# class PositionalEncoding(nn.Module):
#     def __init__(self, dims, max_len):
#         super().__init__()
#         self.dims = dims
#         self.max_len = max_len

#     def forward(self):
#         even_ix = torch.arrange(0, self.dims, 2).float()
#         denominator = torch.pow(10000, even_ix/self.dims)
#         position = torch.arrange(self.max_len).reshape(self.max_len, 1)
#         even_PE = torch.sin(position/denominator)
#         odd_PE = torch.cos(position/denominator)
#         combined = torch.stack([even_PE, odd_PE], dim=2)
#         pe = torch.flatten(combined, start_dim=1, end_dim=2)
#         return pe

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
        self.time_encoding = TimePositionalEncoding(emb_d)
    def forward(self, x, t, mask=None):
        batch_size, max_len, dims = x.shape
        t = t - t[:, 0].unsqueeze(1)
        x = x + self.time_encoding(t)
        qkv = self.QKV(x)
        qkv = qkv.reshape(batch_size, max_len, self.num_heads, 3 * self.head_d)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3, dim=-1)
        attention = scaled_dot_prod(q,k,v,mask)
        attention = attention.reshape(batch_size, max_len, self.num_heads * self.head_d)
        output = self.ffn(attention)
        return output

class FeedForward(nn.Module):
    def __init__(self, emb_d, hidden, dropout_p=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_d, hidden)
        self.fc2 = nn.Linear(hidden, emb_d)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, emb_d, num_heads, ffn_d, dropout_p):
        super(TransformerLayer,self).__init__()
        self.attention = MHSA(emb_d=emb_d, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.ffn = FeedForward(emb_d=emb_d, hidden=ffn_d, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout2 = nn.Dropout(p=dropout_p)
    def forward(self,x,t, mask=None):
        residual_x = x
        x = self.attention(x, t, mask=None)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class Transformer(nn.Module):
    def __init__(self, emb_d, ffn_d, num_heads, dropout_p, num_layers):
        super(Transformer,self).__init__()
        self.model = nn.Sequential(*[TransformerLayer(emb_d=emb_d, num_heads=num_heads, ffn_d=ffn_d, dropout_p=dropout_p) 
                                     for _ in range(num_layers)]) 
    def forward(self, x):
        x = self.model(x)
        return x
    
# class Encoder_Layer(nn.Module):
#     def __init__(self, emb_d, ffn_d, num_heads, dropout_p):
#         super(Encoder_Layer, self).__init__()
#         self.mhsa = 
#         self.norm = nn.LayerNorm(normalized_shape=[emb_d])
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.ffn = 

# class PositionalEncoding(nn.Module):
#     def __init__(self, dims, max_len):
#         super().__init__()
#         self.dims = dims
#         self.max_len = max_len

#     def forward(self, x):
#         even_ix = torch.arrange(0, self.dims, 2).float()
#         denominator = torch.pow(10000, even_ix/self.dims)
#         position = torch.arrange(self.max_len).reshape(self.max_len, 1)
#         even_PE = torch.sin(position/denominator)
#         odd_PE = torch.cos(position/denominator)
#         combined = torch.stack([even_PE, odd_PE], dim=2)
#         pe = torch.flatten(combined, start_dim=1, end_dim=2)
#         return pe



# class PosEncMHSA(nn.Module):
#     def __init__(self, emb_dim, max_len, dropout_p=0.3, num_heads=8, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.emb_dim = emb_dim
#         self.max_len = max_len
#         self.pe = PositionalEncoding(emb_dim, max_len=max_len)
#         self.attention = Attention(emb_dim, num_heads)
#         self.dropout = nn.Dropout(dropout_p)
#         self.LayerNorm = nn.LayerNorm(emb_dim)
#     def forward(self,x):
#         x = self.LayerNorm(x)
#         # x = self.pe.forward(x)
#         x = self.attention.forward(x)
#         x = self.dropout(x)
#         return x
    

# class PositionalEncoding(nn.Module):
#     def __init__(self, dims, max_len):
#         super().__init__()
#         self.dims = dims
#         self.max_len = max_len

#     def forward(self, x):
#         even_ix = torch.arrange(0, self.dims, 2).float()
#         denominator = torch.pow(10000, even_ix/self.dims)
#         position = torch.arrange(self.max_len).reshape(self.max_len, 1)
#         even_PE = torch.sin(position/denominator)
#         odd_PE = torch.cos(position/denominator)
#         combined = torch.stack([even_PE, odd_PE], dim=2)
#         pe = torch.flatten(combined, start_dim=1, end_dim=2)
#         return pe
    

# class Attention(nn.Module):
#     def __init__(self, emb_dim, num_heads, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.emb_dim = emb_dim
#         self.num_heads = num_heads

#         self.query_weights = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.key_weights = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.value_weights = nn.Linear(emb_dim, emb_dim, bias=False)

#         self.add_heads = nn.Linear(emb_dim, emb_dim)

#     def forward(self, x):
#         b, t, e = x.size()
#         h = self.num_heads
#         assert e == self.emb_dim, f"Input embedding dim ({e}) should match layer embedding dim ({self.emb})"

#         s = e // h

#         keys = self.key_weights(x)
#         queries = self.query_weights(x)
#         values = self.value_weights(x)

#         keys = keys.view(b, t, h, s)
#         queries = queries.view(b, t, h, s)
#         values = values.view(b, t, h, s)

#         # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
#         #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

#         # Compute scaled dot-product self-attention

#         # - fold heads into the batch dimension
#         keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
#         queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
#         values = values.transpose(1, 2).contiguous().view(b * h, t, s)

#         queries = queries / (e ** (1 / 4))
#         keys = keys / (e ** (1 / 4))
#         # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
#         #   This should be more memory efficient

#         # - get dot product of queries and keys, and scale
#         dot = torch.bmm(queries, keys.transpose(1, 2))

#         # print(dot)
#         dot = F.softmax(dot, dim=2)

#         # - dot now has row-wise self-attention probabilities

#         # apply the self attention to the values
#         out = torch.bmm(dot, values).view(b, h, t, s)

#         # swap h, t back, unify heads
#         out = out.transpose(1, 2).contiguous().view(b, t, s * h)

#         return self.add_heads(out).squeeze()

        
