import torch
import torch.nn as nn
import torch.nn.functional as F

class gated_colvolution_layer(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(dims,4*dims,kernel_size=1, bias=False)
        self.GLU = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(2*dims,2*dims, kernel_size=3, stride=1, padding='same')
        self.BatchNorm = nn.BatchNorm1d(2*dims)
        self.swish = nn.SiLU()
        self.swish0 = nn.SiLU()
        self.depthwise1 = nn.Conv1d(2*dims,2*dims, kernel_size=7, stride=1, padding='same')
        self.BatchNorm1 = nn.BatchNorm1d(2*dims)
        self.swish01 = nn.SiLU()
        self.depthwise2 = nn.Conv1d(2*dims,2*dims, kernel_size=15, stride=1, padding='same')
        self.BatchNorm2 = nn.BatchNorm1d(2*dims)
        self.swish02 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(2*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.longrange = nn.Conv1d(2*dims,2*dims,kernel_size=111, stride=1, padding='same')
        self.batchNorm2 = nn.BatchNorm1d(2*dims)

    def forward(self, x):
        x = self.pointwise1(x.transpose(1,2))
        x = self.GLU(x)
        x = self.depthwise(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.depthwise1(x)
        x = self.BatchNorm1(x)
        x = self.swish01(x)
        x = self.depthwise2(x)
        x = self.BatchNorm2(x)
        x = self.swish02(x)
        x = self.longrange(x)
        x = self.batchNorm2(x)
        x = self.swish(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1,2)
        return x
        
class gated_colvolution_layer_lstm(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(2*dims,4*dims,kernel_size=1, bias=False)
        self.GLU = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(2*dims,2*dims, kernel_size=3, stride=1, padding='same')
        self.BatchNorm = nn.BatchNorm1d(2*dims)
        self.swish0 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(2*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        

    def forward(self, x):
        x = self.pointwise1(x.transpose(1,2))
        x = self.GLU(x)
        x = self.depthwise(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1,2)
        return x
        