#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()

# in_channels: 节点特征, num_classes: 节点数
class MEGATHeadConv(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_nodes: int,
        thred: float,
        residual: bool = True
    ):
        super(MEGATHeadConv, self).__init__()
        self.num_nodes = num_nodes
        self.thred = thred
        start, end = create_e_matrix(self.num_nodes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = in_feats
        dim_out = out_feats

        self.fc_h1 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_e1 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_proj1 = nn.Linear(3 * dim_out, dim_out, bias=False)
        self.attn_fc1 = nn.Linear(3 * dim_out, 1, bias=False)

        self.softmax = nn.Softmax(2)

        self.bnv1 = nn.BatchNorm1d(num_nodes)
        self.bne1 = nn.BatchNorm1d(num_nodes * num_nodes)

        self.act = nn.ELU()
        self.leaky_relu = nn.LeakyReLU()

        self.residual = residual
        if residual:
            if dim_in != dim_out:
                self.res_proj = nn.Linear(
                    dim_in, dim_out, False)
            else:
                self.res_proj = nn.Identity()
        

        self.init_weights_linear(dim_in, dim_out, 1)

    def init_weights_linear(self, dim_in, dim_out, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        scale_out = gain * np.sqrt(2.0 / dim_out)
        self.fc_h1.weight.data.normal_(0, scale)
        self.fc_e1.weight.data.normal_(0, scale)
        self.fc_proj1.weight.data.normal_(0, scale_out)
        #
        # bn_init(self.bnv1, 1e-6)
        # bn_init(self.bne1, 1e-6)
        # bn_init(self.bnv2, 1e-6)
        # bn_init(self.bne2, 1e-6)
        bn_init(self.bnv1)
        bn_init(self.bne1)

    def forward(self, adj, x, edge):
        start = self.start.to(x)
        end = self.end.to(x)
        res = x

        z_h = self.fc_h1(x)  # V x d_out
        z_e = self.fc_e1(edge)  # E x d_out
        # print("=====")
        # print(z_h.shape)
        # print(x.shape)
        # print(z_e.shape)
        # print(edge.shape)

        z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
                      dim=-1)
        # print(z.shape)
        z_e = self.fc_proj1(z)
        attn = self.leaky_relu(self.attn_fc1(z))
        b, _, _ = attn.shape
        attn = attn.view(b, self.num_nodes, self.num_nodes, 1)
        connectivity_mask = torch.where(
            torch.lt(adj, self.thred), -torch.inf, 0.)
        attn = self.softmax(attn + connectivity_mask.unsqueeze(-1))
        attn = attn.view(b, -1, 1)

        source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
        z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))

        # TODO: 移到外部去
        if self.residual:
            res = self.res_proj(res)
        x = self.act(res + self.bnv1(z_h))
        edge = self.act(self.bne1(z_e))

        return x, edge


class MEGATConv(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int,
        num_nodes: int,
        thred: float,
        residual: bool = True,
    ):
        super(MEGATConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.residual = residual

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                MEGATHeadConv(in_feats, out_feats, num_nodes, thred, residual))
    
    def forward(self, adj, x, edge):

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(adj, x, edge)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)
        
        x = torch.cat(head_outs_h, dim=2)
        edge = torch.cat(head_outs_e, dim=2)

        return x, edge
    
    def __repr__(self):
        return '{}(in_feats={}, out_feats={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_feats,
                                             self.out_feats, self.num_heads, self.residual)


if __name__ == "__main__":
    adj = torch.randn(4, 12, 12).cuda(1)
    e = torch.randn(4, 144, 64).cuda(1)
    x = torch.randn(4, 12, 64).cuda(1)
    gconv = MEGATHeadConv(in_feats=64, out_feats=64, num_nodes=12, thred=0.2).cuda(1)
    x, e = gconv(adj, x, e)
