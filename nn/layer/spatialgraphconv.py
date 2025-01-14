import torch as torch
from torch import nn
from torch.nn import init

class SpatialGraphConv(nn.Module):
    def __init__(
        self, in_feats, out_feats, norm="both", bias=True, activation=None
    ):
        super(SpatialGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, adj, feat):
        adj = adj.to(feat)
        feat = feat.permute(0, 2, 1, 3)
        src_degrees = adj.sum(dim=1).clamp(min=1).unsqueeze(1)
        dst_degrees = adj.sum(dim=2).clamp(min=1).unsqueeze(1)
        feat_src = feat

        if self._norm == "both":
            norm_src = torch.pow(src_degrees, -0.5)
            # shp = norm_src.shape + (1,) * (feat.dim() - 1)
            shp = norm_src.shape + (1,)
            norm_src = torch.reshape(norm_src, shp).to(feat.device)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce torche feature size for aggregation.
            feat_src = torch.matmul(feat_src, self.weight)
            rst = adj.unsqueeze(1) @ feat_src
        else:
            # aggregate first torchen mult W
            rst = adj.unsqueeze(1) @ feat_src
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
            rst = self._activation(rst + feat)

        rst = rst.permute(0, 2, 1, 3)
        return rst

    def __repr__(self):
        custom = f"SpatialGraphConv(in_feats={self._in_feats}, " \
                 f"out_feats={self._out_feats}, " \
                 f"norm={self._norm}, " \
                 f"bias={hasattr(self, 'bias')}, " \
                 f"activation={self._activation})"
        return custom

if __name__ == "__main__":
    a = torch.randn(4, 64, 3000, 16)
    adj = torch.randn(4, 64, 64)
    adj = (adj > 0).float()
    gconv = SpatialGraphConv(16, 8, activation=nn.ReLU())
    print(gconv)
    # pred = gconv(adj, a)
    # print(pred.shape)