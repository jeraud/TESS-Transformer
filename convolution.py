import torch
import torch.nn as nn
import torch.nn.functional as F

class gated_colvolution_layer(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dims = dims
        self.LayerNorm = nn.LayerNorm(dims)
        self.pointwise1 = nn.Conv1d(dims,dims,1,1)
        self.GLU = nn.GLU()
        self.depthwise = nn.Conv1d(int(dims/2),int(dims/2), kernel_size=kernel_size, stride=stride, padding='same')
        self.BatchNorm = nn.BatchNorm1d(int(dims/2))
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(int(dims/2),int(dims/2),1,1)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(int(dims/2), dims)
        self.model = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Conv1d(dims,dims,1,1),
            nn.GLU(),
            nn.Conv1d(dims,dims, kernel_size=kernel_size, stride=stride, padding='same'),
            nn.BatchNorm1d(dims),
            nn.SiLU(),
            nn.Conv1d(dims,dims,1,1),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        x = self.LayerNorm(x)
        # print(1,x.shape)
        x = self.pointwise1(x.transpose(1,2))
        # print(2,x.shape)
        x = self.GLU(x.transpose(1,2))
        # print(3,x.shape)
        x = self.depthwise(x.transpose(1,2))
        # print(4,x.shape)
        x = self.BatchNorm(x)
        # print(5,x.shape)
        x = self.swish(x.transpose(1,2))
        # print(6,x.shape)
        x = self.pointwise2(x.transpose(1,2))
        # print(3,x.shape)
        x = self.dropout(x.transpose(1,2))
        x = self.linear(x)
        return x

class simple_convolution_layer(nn.Module):
    def __init__(self, dims, dropout_p, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dims = dims
        self.LayerNorm = nn.LayerNorm(dims)
        self.conv = nn.Conv1d(dims, dims, kernel_size=kernel_size, stride=stride, padding='same', bias=False)
        self.norm = nn.BatchNorm1d(dims)
        self.siLU = nn.SiLU
        self.dropout = nn.Dropout(dropout_p)
        self.model = nn.Sequential(
            nn.Conv1d(dims, dims, kernel_size=kernel_size, stride=stride, padding='same', bias=False),
            nn.BatchNorm1d(dims),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.conv(x.transpose(1,2))
        x = self.norm(x.transpose(1,2))
        x = self.siLU(x)
        x = self.dropout(x)
        return x
