#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn

from nn.layer.spatialgraphconv import SpatialGraphConv
from nn.config import register_model

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, num_nodes, time_points, feats = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, feats, time_points)
        x = self.relu(self.conv(x))
        x = x.view(batch_size, num_nodes, x.shape[-2], x.shape[-1])
        x = x.permute(0, 1, 3, 2)
        return x

class TemporalMaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        batch_size, num_nodes, time_points, feats = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, feats, time_points)
        x = self.pool(x)
        x = x.view(batch_size, num_nodes, feats, -1)
        x = x.permute(0, 1, 3, 2)
        return x

@register_model
class MTSCN(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.tconv1 = nn.Sequential(
            TemporalConv(1, 16, 3, 1, 1),
            TemporalMaxPool(2, 2),
        )
        self.tconv2 = nn.Sequential(
            TemporalConv(16, 8, 3, 1, 1),
            TemporalMaxPool(3, 3),
        )
        
        self.gcn1 = SpatialGraphConv(8, 8, activation=nn.ReLU())
        # self.gcn2 = SpatialGraphConv(16, 8, activation=nn.ReLU())
        
        self.mlp = nn.Sequential(
            nn.Linear(256000, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )
                                 
    def forward(self, adjacency, h):
        h = h.unsqueeze(-1)
        
        h = self.tconv1(h)
        h = self.tconv2(h)
        
        h = self.gcn1(adjacency, h)
        # h = self.gcn2(adjacency, h)
        
        h = h.contiguous().view(h.shape[0], -1)
        h = torch.sigmoid(self.mlp(h))
        return h

if __name__ == "__main__":
    from data.utils.adjacency import abspearson
    
    x = torch.randn(4, 64, 3000)
    adj = []
    for sub in x:
        adj.append(abspearson(sub))
    adj = torch.stack(adj)
    print(adj.shape)
    adj = (adj > 0.8).float()
    mtscn = MTSCN(2)
    # print(mtscn)
    pred = mtscn(adj, x)
    print(pred.shape)