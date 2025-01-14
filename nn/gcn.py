#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn
from nn.layer.graphconv import GraphConv
from nn.layer.diffgraphlearn import DiffGraphLearn
from typing import Literal, Callable
from nn.config import register_model

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 dropout: float = 0.5,
                 residual: bool = False,
                 norm: Literal["both", "none"] = "both",
                 bias: bool = True,
                 activation: Callable = nn.ReLU()) -> None:
        super(GCNLayer, self).__init__()
        self.gconv = \
            GraphConv(in_feat, out_feat, residual, norm, bias, activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, adj, x):
        return self.dropout(self.gconv(adj, x))
        

@register_model
class GCN(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 n_classes: int,
                 t_embedding: int = 0,
                 n_layers: int = 1,
                 dropout: float = 0.5,
                 num_nodes: int = 0,
                 residual: bool = False,
                 norm: bool = False,
                 mlp: bool = False,
                 readout = "sum",
                 graphlearn: bool = False) -> None:
        super(GCN, self).__init__()
        print("in_dim: ", in_dim)

        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        self.t_embedding = t_embedding
        if t_embedding != 0:
            self.t_embed = nn.Linear(in_dim, t_embedding)
            in_dim = t_embedding
        # self.embedding = nn.Linear(in_dim, hidden_dim)
        print("in_dim: ", in_dim)
        convs = [GCNLayer(in_dim, hidden_dim, dropout, residual)]
        for _ in range(1, n_layers):
            convs.append(GCNLayer(hidden_dim, hidden_dim, dropout, residual))
        self.convs = nn.ModuleList(convs)
        self.norm = norm
        if norm:
            bns = []
            for _ in range(n_layers):
                bns.append(nn.BatchNorm1d(num_nodes))
            self.bns = nn.ModuleList(bns)
        self.readout = readout
        if mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, n_classes)
            )
        else:
            self.mlp = nn.Linear(hidden_dim, n_classes)

    def forward(self, adj: torch.Tensor, x: torch.Tensor,
                raw: torch.Tensor) -> torch.Tensor:
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)

        
        if self.t_embedding != 0:
            x = self.t_embed(x)
        # x = self.embedding(x)
        # for conv in self.convs:
        #     x = conv(adj, x)
        for idx, conv in enumerate(self.convs):
            x = conv(adj, x)
            if self.norm:
                x = self.bns[idx](x)
        # x = torch.sum(x, dim=-2)
        if self.readout == "sum":
            x = torch.sum(x, dim=-2)
        elif self.readout == "mean":
            x = torch.mean(x, dim=-2)
        
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 6, 32)
    adj = torch.randn(2, 6, 6)
    adj = (adj > 0).type(torch.float)
    model = GCN(32, 32, 8, 2, 2, t_embedding=128,mlp=True, graphlearn=True)
    y = model(adj, x, x)
    print(model.save_adj.shape)
