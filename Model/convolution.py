import torch
import torch.nn as nn
import torch.nn.functional as F

class glu_convolution_layer(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(dims,4*dims,kernel_size=1, bias=False)
        self.GLU = nn.GLU(dim=1)
        self.convolution = nn.Conv1d(2*dims,2*dims, kernel_size=3, stride=1, padding='same')
        self.BatchNorm = nn.BatchNorm1d(2*dims)
        self.swish = nn.SiLU()
        self.swish0 = nn.SiLU()
        self.convolution1 = nn.Conv1d(2*dims,2*dims, kernel_size=7, stride=1, padding='same')
        self.BatchNorm1 = nn.BatchNorm1d(2*dims)
        self.swish01 = nn.SiLU()
        self.convolution2 = nn.Conv1d(2*dims,2*dims, kernel_size=15, stride=1, padding='same')
        self.BatchNorm2 = nn.BatchNorm1d(2*dims)
        self.swish02 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(2*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.longrange = nn.Conv1d(2*dims,2*dims,kernel_size=111, stride=1, padding='same')
        self.batchNorm2 = nn.BatchNorm1d(2*dims)

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
        self.BatchNorm = nn.BatchNorm1d(4*dims)
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
        