#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn
from nn.layer.chebconv import ChebConv
from nn.layer.diffgraphlearn import DiffGraphLearn
from typing import Callable, Optional
from nn.config import register_model

class ChebLayer(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 k: int,
                 dropout: float = 0.5,
                 residual: bool = False,
                 activation: Callable = nn.ReLU()) -> None:
        super(ChebLayer, self).__init__()
        self.gconv = ChebConv(in_feat, out_feat, k)
        # self.norm = nn.BatchNorm1d(out_feat)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if in_feat != out_feat:
            self.residual = False
        
    def forward(self, adj, x, lambda_max = None):
        h_in = x
        x = self.gconv(adj, x, lambda_max)
        x = self.activation(x)
        if self.residual:
            x = x + h_in
        x = self.dropout(x)
        return x
        # return self.dropout(self.activation(self.gconv(adj, x, lambda_max)))

@register_model
class ChebNet(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 n_classes: int,
                 k: int = 2,
                 n_layers: int = 1,
                 dropout: float = 0.5,
                 num_nodes: int = 0,
                 residual: bool = False,
                 graphlearn: nn.Module = None) -> None:
        super(ChebNet, self).__init__()
        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        # self.embedding = nn.Linear(in_dim, hidden_dim)
        # self.norm = nn.BatchNorm1d(hidden_dim)
        # convs = [ChebLayer(hidden_dim, hidden_dim, k, dropout, residual)
        #          for _ in range(n_layers-1)]
        # self.convs = nn.ModuleList(convs)
        # self.convs.append(ChebLayer(hidden_dim, out_dim, k, dropout))
        convs = [ChebLayer(in_dim, hidden_dim, k, dropout, residual)]
        for _ in range(1, n_layers):
            convs.append(ChebLayer(hidden_dim, hidden_dim, k, dropout, residual))
        self.convs = nn.ModuleList(convs)
        # self.mlp = nn.Sequential(
        #     nn.Linear(out_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, n_classes)
        # )
        self.mlp = nn.Linear(hidden_dim, n_classes)

    def forward(self, adj: torch.Tensor, x: torch.Tensor,
                raw: torch.Tensor,
                lambda_max: Optional[float] = None) -> torch.Tensor:
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)
        # x = self.embedding(x)
        # b, v, t = x.shape
        # x = self.norm(x.view(b*v, t)).view(b, v, t)
        # x = torch.relu(x)

        for conv in self.convs:
            x = conv(adj, x, lambda_max)
        x = torch.sum(x, dim=-2)
        
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 6, 32)
    adj = torch.randn(2, 6, 6)
    adj = (adj > 0).type(torch.float)
    model = ChebNet(32, 32, 16, 8, 8, 2)
    print(model)
    y = model(adj, x, x)
    print(y.shape)