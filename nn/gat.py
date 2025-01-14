#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn
from nn.layer.gatconv import GATConv
from nn.layer.diffgraphlearn import DiffGraphLearn
from typing import Callable, Optional
from nn.config import register_model

class GATLayer(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 num_heads: int,
                 dropout: float,
                 thred: float,
                 residual: bool = False,
                 bias: bool = True,
                 activation: Optional[Callable] = nn.ELU()) -> None:
        super(GATLayer, self).__init__()
        self.gconv = GATConv(in_feat,
                             out_feat // num_heads,
                             num_heads,
                             dropout,
                             thred,
                             residual=residual,
                             bias=bias,
                             activation=activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, adj, x):
        return self.dropout(self.gconv(adj, x))
        

@register_model
class GAT(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 num_heads: int,
                 thred: float,
                 n_classes: int,
                 n_layers: int = 1,
                 dropout: float = 0.5,
                 num_nodes:int = 0,
                 residual: bool = False,
                 readout: str = "sum",
                 graphlearn: bool = False,
                 self_loop: bool = False) -> None:
        super(GAT, self).__init__()
        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        # self.embedding = nn.Linear(in_dim, hidden_dim)
        convs = \
            [GATLayer(in_dim, hidden_dim, num_heads, dropout, thred, residual)]
        for _ in range(1, n_layers):
            convs.append(GATLayer(
                hidden_dim, hidden_dim, num_heads, dropout, thred, residual))
        self.convs = nn.ModuleList(convs)
        # convs = [GATLayer(hidden_dim,
        #                   hidden_dim,
        #                   num_heads,
        #                   dropout,
        #                   thred)
        #             for _ in range(n_layers-1)]
        # self.convs = nn.ModuleList(convs)
        # self.convs.append(
        #     GATLayer(hidden_dim, out_dim, num_heads, dropout, thred))
        # self.mlp = nn.Sequential(
        #     nn.Linear(out_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, n_classes)
        # )
        self.mlp = nn.Linear(hidden_dim, n_classes)
        self.readout = readout
        self.self_loop = self_loop

    def forward(self, adj: torch.Tensor, x: torch.Tensor,
                raw: torch.Tensor) -> torch.Tensor:
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)
        # x = self.embedding(x)
        if self.self_loop:
            adj_d = adj.diagonal(dim1=-2, dim2=-1).diag_embed().to(adj)
            adj = adj - adj_d + torch.eye(adj.shape[-1]).to(adj)

        for conv in self.convs:
            x = conv(adj, x)
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
    model = GAT(32, 32, 16, 2, 2, 0.8, 2, 1, 0.5, True, "mean")
    print(model)
    y = model(adj, x, x)
    print(y.shape)