import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self, inp_dims, num_classes, dropout_p):
        super(Classification,self).__init__()
        self.inp_dims = inp_dims
        self.classes = num_classes
        self.norm = nn.LayerNorm(inp_dims)
        self.pool1 = nn.AvgPool1d(kernel_size=5, stride=5)
        self.conv1 = nn.Conv1d(inp_dims, inp_dims, kernel_size=3, stride=3,padding=1)
        self.swish = nn.SiLU()
        self.pool2 = nn.AvgPool1d(kernel_size=5, stride=5)
        self.fc1 = nn.Linear(inp_dims*18, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32,8)
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.swish4 = nn.SiLU()
    def forward(self, x):
        print(x.shape)
        # x = self.norm(x)
        x = x.transpose(1,2)
        x = self.pool1(x)
        print('1',x.shape)
        x = self.conv1(x)
        x = self.dropout1(x)
        print('2',x.shape)
        x = self.swish(x)
        x = self.pool2(x)
        print('3',x.shape)
        # x = x.transpose(1,2)
        x = x.flatten(start_dim=1)
        print('4',x.shape)
        x = self.fc1(x)
        x = self.dropout2(x)
        print('5',x.shape)
        x = self.swish2(x)
        x = self.fc2(x)
        x = self.swish3(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.swish4(x)
        print('6',x.shape)
        x = F.softmax(x, dim=-1).squeeze()
        print('out',x.shape)
        return x



class Output(nn.Module):
    def __init__(self, emb_d, num_classes, dropout_p):
        super(Output, self).__init__()
        self.conv1 = nn.Conv1d(emb_d, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(32,32,kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(32,8, kernel_size=3, padding='same')
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        x = x.transpose(0,1)
        # print('11', x.shape)
        x = self.conv1(x)
        # print('12', x.shape)
        x = self.swish1(x)
        x = self.conv2(x)
        # print('13', x.shape)
        x = self.swish2(x)
        x = self.conv3(x)
        # print('14', x.shape)
        x = self.softmax(x.transpose(0,1))
        return x