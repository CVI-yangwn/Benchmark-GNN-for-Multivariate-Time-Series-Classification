#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn

from nn.layer.diffgraphlearn import DiffGraphLearn
from nn.layer.spatialgraphconv import SpatialGraphConv
from nn.layer.temporalconv import TemporalConv

from nn.config import register_model

@register_model
class STConv(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, out_feat: int, 
                 kernel_size: int, dropout: float, residual: bool = False):
        super().__init__()
        self.tconv1 = TemporalConv(in_feat, hidden_feat, kernel_size)
        self.sconv = SpatialGraphConv(hidden_feat, hidden_feat,
                                      activation=nn.ReLU())
        self.tconv2 = TemporalConv(hidden_feat, out_feat, kernel_size)
        self.dropout = nn.Dropout(dropout)
        # self.residual = residual
        # if in_feat != out_feat:
        #     self.residual = False

    def forward(self, adj: torch.Tensor, x: torch.Tensor):
        # h_in = x
        x = self.tconv1(x)
        x = self.sconv(adj, x)
        x = self.tconv2(x)
        # if self.residual:
        #     x = x + h_in
        return self.dropout(x)

@register_model
class STGCN(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 t_embedding: int,
                 mlp_dim: int,
                 k: int,
                 n_classes: int,
                 n_layers: int = 2,
                 dropout: float = 0.5,
                 num_nodes: int = 0,
                 residual: bool = False,
                 readout: str = "sum",
                 graphlearn: bool = False):
        super().__init__()
        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        self.t_embed = nn.Linear(in_dim, t_embedding)
        convs = [STConv(1, hidden_dim // 4, hidden_dim, k, dropout, residual)]
        for _ in range(1, n_layers):
            convs.append(STConv(
                hidden_dim, hidden_dim // 4, hidden_dim, k, dropout, residual))
        self.convs = nn.ModuleList(convs)
        # self.stconv1 = \
        #     STConv(1, hidden_dim // 2, hidden_dim, k, dropout, residual)
        # self.stconv2 = STConv(
        #     hidden_dim, hidden_dim // 2, hidden_dim, k, dropout, residual)
        
        # t_embedding -= n_layers * 2 * (k - 1)
        self.tlinear = nn.Linear(t_embedding, 1)

        # self.mlp = nn.Sequential(
        #     nn.Linear(out_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, n_classes)
        # )
        self.mlp = nn.Linear(hidden_dim, n_classes)
        self.readout = readout
                                 
    def forward(self, adj: torch.Tensor, h: torch.Tensor, raw: torch.Tensor):
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)
        
        h = self.t_embed(h)

        h = h.unsqueeze(-1)
        
        # h = self.stconv1(adj, h)
        # h = self.stconv2(adj, h)
        for conv in self.convs:
            h = conv(adj, h)

        h = self.tlinear(h.permute(0, 1, 3, 2))
        b, v, f, _ = h.shape
        h = h.reshape(b, v, f)

        if self.readout == "sum":
            h = torch.sum(h, dim=1)
        elif self.readout == "mean":
            h = torch.mean(h, dim=1)
        # h = h.sum(dim=1)
        # h = h.mean(dim=1)

        h = self.mlp(h)
        return h

if __name__ == "__main__":
    from data.utils.adjacency import abspearson
    
    x = torch.randn(4, 9, 144)
    adj = []
    for sub in x:
        adj.append(abspearson(sub))
    adj = torch.stack(adj)
    print(adj.shape)
    adj = (adj > 0.8).float()
    stgcn = STGCN(144, 144, 128, 128, 128, 3, 2, 3, 0.5, True, "mean")
    print(stgcn)
    pred = stgcn(adj, x, x)
    print(pred.shape)