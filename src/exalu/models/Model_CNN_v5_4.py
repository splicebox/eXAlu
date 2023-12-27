# this model is for EAD, containing the context of Alu Seqs
from torch.nn.functional import leaky_relu, relu
from torch import nn, tanh
import torch
from torch.nn.modules import conv

MAX_LEN = 400

class CNet_v5_4(nn.Module):
    def __init__(self, batch_size, output_dim=1):
        super(CNet_v5_4, self).__init__()
        self.batch_size = batch_size
        # self.conv1 = nn.Conv1d(21, 8, kernel_size=1)
        self.conv = nn.ModuleList()
        self.bnc = nn.ModuleList()
        self.pool = nn.ModuleList()

        self.conv.append(nn.Conv1d(4, 64, kernel_size=11))
        self.bnc.append(nn.BatchNorm1d(64))
        self.pool.append(nn.AvgPool1d(2))

        self.conv.append(nn.Conv1d(64, 128, kernel_size=4))
        self.bnc.append(nn.BatchNorm1d(128))
        self.pool.append(nn.AvgPool1d(4))
        
        self.conv.append(nn.Conv1d(128, 256, kernel_size=4))
        self.bnc.append(nn.BatchNorm1d(256))
        self.pool.append(nn.AvgPool1d(5))

        self.conv.append(nn.Conv1d(256, 512, kernel_size=3))
        self.bnc.append(nn.BatchNorm1d(512))
        self.pool.append(nn.AvgPool1d(7))
        # self.pool.append(nn.AvgPool1d(5))

        self.dp5 = nn.Dropout(0.5)
        self.dp3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 32)
        self.bnf1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.dp5(relu(self.bnc[0](self.conv[0](x))))
        x = self.pool[0](x)
        x = self.dp5(relu(self.bnc[1](self.conv[1](x))))
        x = self.pool[1](x)
        x = self.dp5(relu(self.bnc[2](self.conv[2](x))))
        x = self.pool[2](x)
        x = self.dp5(relu(self.bnc[3](self.conv[3](x))))
        x = self.pool[3](x)

        x = x.view(-1, 512)
        x = self.dp5(relu(self.bnf1(self.fc1(x))))
        x = self.fc2(x)
        return x