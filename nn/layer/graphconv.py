import torch
from torch import nn
from torch.nn import init
from typing import Literal, Callable

class GraphConv(nn.Module):

    def __init__(
        self,
        in_feats: torch.Tensor,
        out_feats: torch.Tensor,
        residual: bool = False,
        norm: Literal["both", "none"] = "both",
        bias: bool = True,
        activation: Callable = None
    ):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.residual = residual
        self._norm = norm
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)
        if in_feats != out_feats:
            self.residual = False

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, adj: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        h_in = feat
        # src_degrees = adj.sum(dim=0).clamp(min=1)
        src_degrees = adj.sum(dim=1).clamp(min=1)
        # dst_degrees = adj.sum(dim=1).clamp(min=1)
        dst_degrees = adj.sum(dim=2).clamp(min=1)
        feat_src = feat

        if self._norm == "both":
            norm_src = torch.pow(src_degrees, -0.5)
            # shp = norm_src.shape + (1,) * (feat.dim() - 1)
            shp = norm_src.shape + (1,)
            norm_src = torch.reshape(norm_src, shp).to(feat.device)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = torch.matmul(feat_src, self.weight)
            rst = adj @ feat_src
        else:
            # aggregate first then mult W
            rst = adj @ feat_src
            rst = torch.matmul(rst, self.weight)

        if self._norm != "none":
            if self._norm == "both":
                norm_dst = torch.pow(dst_degrees, -0.5)
            else:  # right
                norm_dst = 1.0 / dst_degrees
            # shp = norm_dst.shape + (1,) * (feat.dim() - 1)
            shp = norm_dst.shape + (1,)
            norm_dst = torch.reshape(norm_dst, shp).to(feat.device)
            rst = rst * norm_dst

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
        
        if self.residual:
            rst = h_in + rst 

        return rst

    def __repr__(self):
        custom = f"GraphConv(in_feats={self._in_feats}, " \
                 f"out_feats={self._out_feats}, " \
                 f"residual={self.residual}, " \
                 f"norm={self._norm}, " \
                 f"bias={hasattr(self, 'bias')}, " \
                 f"activation={self._activation})"
        return custom

if __name__ == "__main__":
    # x = torch.randn(2, 6, 10)
    # adj = torch.randn(2, 6, 6)
    # adj = (adj > 0).type(torch.float)
    model = GraphConv(10, 5)
    print(model)
    # y = model(adj, x)
    # print(y.shape)