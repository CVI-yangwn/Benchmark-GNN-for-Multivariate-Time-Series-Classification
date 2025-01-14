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
        
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end
from torch.autograd import Variable
import numpy as np
def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge

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


import math
class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x
class MEGAT_DEBUG(nn.Module):
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
        super().__init__()
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
        self.mlp = nn.Linear(hidden_dim, n_classes)
        self.readout = readout
        self.self_loop = self_loop

        self.gnn = GNN(length, num_nodes)

        class_linear_layers = []
        for i in range(num_nodes):
            layer = LinearBlock(in_dim, in_dim)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)

    def forward(self, adj: torch.Tensor, x: torch.Tensor,
                raw: torch.Tensor) -> torch.Tensor:
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)
        # x = self.embedding(x)
        if self.self_loop:
            adj_d = adj.diagonal(dim1=-2, dim2=-1).diag_embed().to(adj)
            adj = adj - adj_d + torch.eye(adj.shape[-1]).to(adj)

        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        # x = self.t_embed(x)
        # gx = self.gproj(x)  # GlobalProj
        print("===============================")
        print("f_u shape:", f_u.shape)  # torch.Size([8, 4, 4, 5])
        print("x shape:", x.shape)  # torch.Size([8, 4, 5])
        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        print("f_e shape:", f_e.shape)  # torch.Size([8, 16, 5])
        f_v, f_e = self.gnn(f_v, f_e)
        print("f_v shape:", f_v.shape)  # torch.Size([8, 4, 5])
        print("f_e shape:", f_e.shape)  # torch.Size([8, 16, 5])

        # e = torch.relu(e)
        
        for conv in self.convs:
            x, f_e = conv(adj, x, f_e)
            # print("===", e.shape)
        print("x shape:", x.shape)  # torch.Size([8, 4, 64])
        print("f_e shape:", f_e.shape)  # torch.Size([8, 16, 64])
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
    megat = MEGAT_DEBUG(length, length, 64, 0, 128, 1, 0.2, 7, 1, 0.5, dim, graphlearn=True).cuda(cuda)
    crit = nn.CrossEntropyLoss()
    gd = torch.ones(8, 7).cuda(cuda)

    optimizer=torch.optim.Adam(megat.parameters(),lr=0.005)
    for _ in range(100):
        y = megat(adj, x, raw)
        loss = crit(y, gd)
        loss.backward()
        optimizer.step()