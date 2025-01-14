#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn
from nn.layer.megatconv import MEGATConv
from nn.layer.diffgraphlearn import DiffGraphLearn
from nn.layer.graph_edge_model import GEM, GlobalProj
from typing import Callable, Optional
from nn.config import register_model

class MEGATLayer(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 num_heads: int,
                 num_nodes: int,
                 dropout: float,
                 thred: float,
                 residual: bool = True,
                 bias: bool = True,
                 activation: Optional[Callable] = nn.ELU()) -> None:
        super(MEGATLayer, self).__init__()
        self.gconv = MEGATConv(in_feat,
                             out_feat // num_heads,
                             num_heads,
                             num_nodes,
                             thred,
                             residual=residual)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, adj, x, e):
        x, e = self.gconv(adj, x, e)
        return self.dropout(x), e

@register_model
class MEGAT(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 t_embedding: int,
                 mlp_dim: int,
                 num_heads: int,
                 thred: float,
                 n_classes: int,
                 n_layers: int = 1,
                 dropout: float = 0.5,
                 num_nodes: int = 0,
                 residual: bool = True,
                 readout: str = "sum",
                 graphlearn: bool = False,
                 self_loop: bool = False) -> None:
        super(MEGAT, self).__init__()
        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        # self.embedding = nn.Linear(in_dim, hidden_dim)
        if t_embedding > 0:
            self.t_embed = nn.Linear(in_dim, t_embedding)
            in_dim = t_embedding
        else:
            self.t_embed = nn.Identity()
        # if in_dim == len:
        #     self.t_embed = nn.Linear(in_dim, t_embedding)
        # else:
        #     self.t_embed = nn.Identity()
        # self.norm = nn.BatchNorm1d(hidden_dim)
        # self.edge_extractor = GEM(in_dim)
        self.gproj = GlobalProj(num_nodes)
        self.edge_extractor = GEM(in_dim, num_nodes)
        
        convs = [MEGATLayer(in_dim,
                            hidden_dim,
                            num_heads,
                            num_nodes,
                            dropout,
                            thred,
                            residual)]
        for _ in range(1, n_layers):
            convs.append(MEGATLayer(
                hidden_dim,
                hidden_dim,
                num_heads,
                num_nodes,
                dropout,
                thred,
                residual))
        self.convs = nn.ModuleList(convs)
        # convs = [MEGATLayer(in_dim,
        #                   hidden_dim,
        #                   num_heads,
        #                   dropout,
        #                   thred)
        #             for _ in range(n_layers-1)]
        # self.convs = nn.ModuleList(convs)
        # self.convs.append(
        #     MEGATLayer(hidden_dim, out_dim, num_heads, dropout, thred))
        # self.mlp = nn.Sequential(
        #     nn.Linear(out_dim, mlp_dim),
        #     # nn.BatchNorm1d(mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     # nn.BatchNorm1d(mlp_dim),
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

        # b, v, t = x.shape
        # x = self.norm(x.view(b*v, t)).view(b, v, t)
        # x = torch.relu(x)
        x = self.t_embed(x)  # [B,V,T]
        gx = self.gproj(x)  # GlobalProj [B,T]
        e = self.edge_extractor(x, gx)  # [B,V*V,T]
        
        # print(e.shape)
        # e = torch.relu(e)
        
        for conv in self.convs:
            x, e = conv(adj, x, e)
            # print("===", e.shape)

        # # x = torch.sum(x, dim=-2)
        # x = torch.mean(x, dim=-2)
        if self.readout == "sum":
            x = torch.sum(x, dim=-2)
        elif self.readout == "mean":
            x = torch.mean(x, dim=-2)
        
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    cuda = 3
    dim = 4
    length = 5
    raw = torch.randn(8, dim, length).cuda(cuda)
    x = torch.randn(8, dim, length).cuda(cuda)
    adj = torch.randn(8, dim, dim).cuda(cuda)
    megat = MEGAT(length, length, 64, 0, 128, 1, 0.2, 7, 1, 0.5, dim, graphlearn=True).cuda(cuda)
    crit = nn.CrossEntropyLoss()
    gd = torch.ones(8, 7).cuda(cuda)

    optimizer=torch.optim.Adam(megat.parameters(),lr=0.005)
    for _ in range(100):
        y = megat(adj, x, raw)
        loss = crit(y, gd)
        loss.backward()
        optimizer.step()