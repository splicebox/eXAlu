# this model is for EAD, containing the context of Alu Seqs
from torch.nn.functional import leaky_relu, relu
from torch import nn, tanh
import torch
from torch.nn.modules import conv

MAX_LEN = 400

class FCN_v1(nn.Module):
    def __init__(self, batch_size, output_dim=1):
        super(FCN_v1, self).__init__()
        self.batch_size = batch_size
        # self.conv1 = nn.Conv1d(21, 8, kernel_size=1)
        self.conv = nn.ModuleList()
        self.bnc = nn.ModuleList()

        self.conv.append(nn.Conv1d(4, 128, kernel_size=20))
        self.bnc.append(nn.BatchNorm1d(128))

        self.conv.append(nn.Conv1d(128, 256, kernel_size=16))
        self.bnc.append(nn.BatchNorm1d(256))
        
        self.conv.append(nn.Conv1d(256, 512, kernel_size=12))
        self.bnc.append(nn.BatchNorm1d(512))

        self.conv.append(nn.Conv1d(512, 256, kernel_size=8))
        self.bnc.append(nn.BatchNorm1d(256))

        self.conv.append(nn.Conv1d(256, 128, kernel_size=4))
        self.bnc.append(nn.BatchNorm1d(128))

        self.fc = nn.Linear(128, output_dim)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dp2 = nn.Dropout(0.2)
        self.dp5 = nn.Dropout(0.5)
    
    def forward(self, x):
        for i in range(5):
            x = self.conv[i](x)
            x = self.bnc[i](x)
            x = relu(x)
            x = self.dp2(x)
        # global pooling
        x = x.mean([-1])
        # x = self.pool(x)
        # x = x.squeeze()
        # fc
        x = self.fc(x)
        x = self.dp5(x)
        return x
