import torch
import torch.nn as nn
from Model.multi_head_attention import TransformerLayer
from Model.convolution import glu_convolution_layer, convolution_layer_lstm
import torch.nn.functional as F

class LSATT_CONV(nn.Module):
    def __init__(self, emb_d, num_heads, dropout_p, ffn_d):
        super(LSATT_CONV, self).__init__()
        self.MHSA = TransformerLayer(emb_d=emb_d, ffn_d=ffn_d, num_heads=num_heads, dropout_p=dropout_p)
        self.conv = glu_convolution_layer(emb_d, dropout_p=dropout_p)
        self.conv2 = convolution_layer_lstm(emb_d, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.lstm = nn.LSTM(emb_d,emb_d, batch_first=True, dropout=dropout_p, num_layers=2, bidirectional=True)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)   

    def forward(self, x, t, mask=None):
        resid_x = x
        x = self.lstm(x)[0]
        x = self.conv2(x)
        x = x + resid_x
        x = self.norm2(x)
        x = self.MHSA(x,t, mask)
        resid_x = x
        x = self.conv(x)
        x = x + resid_x
        x = self.norm1(x)
        return x



class LCC(nn.Module):
    def __init__(self, emb_d, num_heads, layers, dropout_p, ffn_d, num_classes):
        super(LCC,self).__init__()
        self.lsatt_conv_layers  = nn.ModuleList([LSATT_CONV(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d) for _ in range(layers)])
        self.fc1 = nn.Linear(emb_d, 20)
        self.fc2 = nn.Linear(20,num_classes)
        self.embed1 = nn.Linear(1, emb_d)
        self.swish = nn.SiLU()
        self.norm0 = nn.LayerNorm(normalized_shape=20)
        self.norm1 = nn.LayerNorm(normalized_shape=num_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout0 = nn.Dropout(dropout_p)

    def forward(self, x,t, mask=None):
        x = self.embed1(x.unsqueeze(2))
        for lsatt_conv in self.lsatt_conv_layers:
            x = lsatt_conv(x, t)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.norm0(x)
        x = self.swish(x)
        x = self.dropout0(x)
        x = self.fc2(x)
        x = self.norm1(x)
        x = F.softmax(x, dim=-1).squeeze()
        return x
