# this model is for EAD, containing the context of Alu Seqs
from torch.nn.functional import leaky_relu, relu
from torch import nn, tanh
import torch
from torch.nn.modules import conv

MAX_LEN = 400

class CNet_v6(nn.Module):
    def __init__(self, batch_size, output_dim=1):
        super(CNet_v6, self).__init__()
        self.batch_size = batch_size
        # self.conv1 = nn.Conv1d(21, 8, kernel_size=1)
        self.conv = nn.ModuleList()
        self.bnc = nn.ModuleList()
        self.pool = nn.ModuleList()

        self.conv.append(nn.Conv1d(4, 128, kernel_size=5))
        
        self.bnc.append(nn.BatchNorm1d(128))
        self.pool.append(nn.AvgPool1d(4))

    # TODO: decrease channel
        self.conv.append(nn.Conv1d(128, 512, kernel_size=5)) 
        self.bnc.append(nn.BatchNorm1d(512))
        self.pool.append(nn.AvgPool1d(5))
        
        self.conv.append(nn.Conv1d(512, 1024, kernel_size=5))
        self.bnc.append(nn.BatchNorm1d(1024))
        self.pool.append(nn.AvgPool1d(15))

        # self.conv.append(nn.Conv1d(1024, 2048, kernel_size=4))
        # self.bnc.append(nn.BatchNorm1d(2048))

        # self.conv.append(nn.Conv1d(512, 512, kernel_size=3))
        # self.bnc.append(nn.BatchNorm1d(512))
        # self.pool.append(nn.MaxPool1d(1))

        self.dp2 = nn.Dropout(0.2)
        self.dp5 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.bnf1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_dim)
        torch.nn.init.uniform_(self.fc1.weight)

    
    def forward(self, x):
        '''
        >>>>>>>
        torch.Size([2048, 1024, 1])
        torch.Size([2048, 1024])
        torch.Size([2048, 512])
        torch.Size([2048, 1])
        '''
        # x = torch.cat((a, x), dim=2)
        # print(x.shape)
        for i in range(3):
            x = self.bnc[i](relu(self.conv[i](x)))
            x = self.pool[i](x)
            # print(x.shape)
        x = x.view(-1, 1024)
        # x = tanh(self.bnf1(self.fc1(x)))
        # print(torch.max(x[0]).data, torch.min(x[0]).data, torch.mean(x[0]).data)
        x = self.fc1(x)
        torch.set_printoptions(profile="full")
        # print(x.max(), x.min(), x.mean())
        x = relu(x)
        # print(x.max(), x.min(), x.mean())
        # print(torch.max(x[0]).data, torch.min(x[0]).data, torch.mean(x[0]).data)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x


#         # this model is for EAD, containing the context of Alu Seqs
# from torch.nn.functional import leaky_relu, relu
# from torch import nn, tanh
# import torch
# from torch.nn.modules import conv

# MAX_LEN = 400

# class CNet_v6(nn.Module):
#     def __init__(self, batch_size, output_dim=1):
#         super(CNet_v6, self).__init__()
#         self.batch_size = batch_size
#         # self.conv1 = nn.Conv1d(21, 8, kernel_size=1)
#         self.conv = nn.ModuleList()
#         self.bnc = nn.ModuleList()
#         self.pool = nn.ModuleList()

#         self.conv.append(nn.Conv1d(4, 32, kernel_size=5))
        
#         self.bnc.append(nn.BatchNorm1d(32))
#         self.pool.append(nn.AvgPool1d(4))

#     # TODO: decrease channel
#         self.conv.append(nn.Conv1d(32, 64, kernel_size=5)) 
#         self.bnc.append(nn.BatchNorm1d(128))
#         self.pool.append(nn.AvgPool1d(5))
        
#         self.conv.append(nn.Conv1d(64, 128, kernel_size=5))
#         self.bnc.append(nn.BatchNorm1d(128))
#         self.pool.append(nn.AvgPool1d(15))

#         # self.conv.append(nn.Conv1d(1024, 2048, kernel_size=4))
#         # self.bnc.append(nn.BatchNorm1d(2048))

#         # self.conv.append(nn.Conv1d(512, 512, kernel_size=3))
#         # self.bnc.append(nn.BatchNorm1d(512))
#         # self.pool.append(nn.MaxPool1d(1))

#         self.dp2 = nn.Dropout(0.2)
#         self.dp5 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(128, 1)
#         # self.bnf1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(512, output_dim)
#         torch.nn.init.uniform_(self.fc1.weight)

    
#     def forward(self, x):
#         '''
#         >>>>>>>
#         torch.Size([2048, 1024, 1])
#         torch.Size([2048, 1024])
#         torch.Size([2048, 512])
#         torch.Size([2048, 1])
#         '''
#         # x = torch.cat((a, x), dim=2)
#         # print(x.shape)
#         for i in range(3):
#             x = self.bnc[i](relu(self.conv[i](x)))
#             x = self.pool[i](x)
#             # print(x.shape)
#         x = x.view(-1, 1024)
#         # x = tanh(self.bnf1(self.fc1(x)))
#         # print(torch.max(x[0]).data, torch.min(x[0]).data, torch.mean(x[0]).data)
#         x = self.fc1(x)
#         x = relu(x)
#         # print(torch.max(x[0]).data, torch.min(x[0]).data, torch.mean(x[0]).data)
#         # x = self.fc2(x)
#         # x = self.softmax(x)
#         return x